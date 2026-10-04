"""End-to-end: mix file → recognised tracks → transitions → MP4 per transition."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .audio import decode_mono, probe_duration
from .covers import fetch_cover
from .features import ANALYSIS_SR, compute_features
from .recognize import DEFAULT_URL, BudgetExhausted, Recognizer, ShazamClient
from .render import FORMATS, render_card
from .segment import find_boundaries, plan_probes
from .tracklist import apply_tracklist
from .transitions import assign_unknowns, find_transitions, group_occurrences
from .util import mmss, slug
from .video import make_video

log = logging.getLogger(__name__)


@dataclass
class Options:
    api_key: str | None = None
    api_url: str = DEFAULT_URL
    api_field: str = "file"
    cache_dir: Path = Path.home() / ".cache" / "djclips"
    dry_run: bool = False
    max_calls: int | None = None
    probe_seconds: float = 10
    min_track: float = 60
    max_segment: float = 360
    pad: float = 10
    max_clip: float = 60
    formats: tuple[str, ...] = ("9x16", "4x5")
    tracklist: Path | None = None  # editable overrides; default <out>/tracks.json


def run(input_path: Path, out_dir: Path, opts: Options) -> dict:
    input_path = input_path.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    total = probe_duration(input_path)
    log.info("Analysing %s (%s)…", input_path.name, mmss(total))
    F = compute_features(decode_mono(input_path, ANALYSIS_SR))

    boundaries = find_boundaries(F, opts.min_track)
    probes = plan_probes(len(F), boundaries, opts.probe_seconds, opts.max_segment)
    client = ShazamClient(opts.api_key, opts.api_url, opts.api_field) if opts.api_key else None
    recognizer = Recognizer(input_path, opts.cache_dir / "shazam.json", client, opts.max_calls, opts.dry_run)
    needed = sum(not recognizer.is_cached(p) for p in probes)
    log.info("Found %d candidate boundaries → %d probes (%d cached, %d need API calls)",
             len(boundaries), len(probes), len(probes) - needed, needed)

    if opts.dry_run:
        for p in probes:
            state = "cached" if recognizer.is_cached(p) else "API call"
            print(f"  probe {mmss(p.start):>8} – {mmss(p.end):<8} [{state}]")
        print(f"Dry run: {needed} API call(s) would be made.")
        return {"probes": [[p.start, p.end] for p in probes], "api_calls_needed": needed}

    tracks = []
    for p in probes:
        try:
            tracks.append(recognizer.recognize(p))
        except BudgetExhausted as exc:
            log.warning("%s; remaining probes skipped", exc)
            tracks.append(None)
            recognizer.dry_run = True  # serve the rest from cache only
        t = tracks[-1]
        log.info("  %s → %s", mmss(p.start), f"{t.artist} – {t.title}" if t else "no match")

    tracklist = opts.tracklist or out_dir / "tracks.json"
    tracks = apply_tracklist(tracklist, probes, tracks)
    log.info("Track names/covers: edit %s and re-run to fix any mistakes", tracklist)
    resolved = assign_unknowns(F, probes, tracks)
    for p, before, after in zip(probes, tracks, resolved):
        if before is None:
            log.info("  %s unmatched → %s", mmss(p.start), f"sounds like {after.title}" if after else "unknown track")
    occurrences = group_occurrences(probes, resolved)
    transitions = find_transitions(F, occurrences, total, opts.pad, opts.max_clip)
    log.info("Identified %d tracks, %d transitions", len(occurrences), len(transitions))

    covers_dir = opts.cache_dir / "covers"
    work = out_dir / ".frames"
    for fmt in opts.formats:
        (work / fmt).mkdir(parents=True, exist_ok=True)
        (out_dir / fmt).mkdir(exist_ok=True)
        for stale in (out_dir / fmt).glob("*.mp4"):  # names change when tracks are edited
            stale.unlink()
    videos = []
    for i, tr in enumerate(transitions, 1):
        name = f"{i:02d}_{slug(tr.a.title)}_x_{slug(tr.b.title)}"
        cover_a, cover_b = fetch_cover(tr.a, covers_dir), fetch_cover(tr.b, covers_dir)
        made = {}
        for fmt in opts.formats:
            card = render_card(tr.a, tr.b, cover_a, cover_b, work / fmt / f"{name}.png", FORMATS[fmt])
            video = make_video(card, input_path, tr.clip_start, tr.clip_end, out_dir / fmt / f"{name}.mp4")
            made[fmt] = f"{fmt}/{video.name}"
        videos.append(made)
        log.info("  [%d/%d] %s  (%s–%s, %.0fs)", i, len(transitions), name,
                 mmss(tr.clip_start), mmss(tr.clip_end), tr.clip_end - tr.clip_start)

    report = {
        "input": str(input_path),
        "api_calls": recognizer.calls,
        "tracks": [{"id": o.track.uid, "title": o.track.title, "artist": o.track.artist, "isrc": o.track.isrc,
                    "heard_at": [mmss(p.start) for p in o.anchors]} for o in occurrences],
        "transitions": [dict(tr.to_dict(), videos=v) for tr, v in zip(transitions, videos)],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report
