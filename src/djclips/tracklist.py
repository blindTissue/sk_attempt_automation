"""The editable tracks.json: fix a title, artist or cover, or say which track a
segment really is. Manual edits always win over Shazam; re-running is free
because recognitions are cached."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .recognize import Track
from .segment import Probe
from .util import mmss, slug

log = logging.getLogger(__name__)

HELP = [
    "Edit and re-run the same command to re-render (no new API calls).",
    "tracks: change title / artist; cover may be an image URL or a local file path (relative to this file).",
    "segments: set 'track' to a track id to fix a wrong or missing match, or null for 'unknown'.",
    "You can add new track ids under tracks and reference them from segments.",
]


def apply_tracklist(path: Path, probes: list[Probe], recognised: list[Track | None]) -> list[Track | None]:
    """Merge Shazam results into `path` (creating it on first run) and return
    the per-probe tracks as the file describes them."""
    data = json.loads(path.read_text()) if path.exists() else {}
    entries: dict = data.get("tracks", {})
    assigned = {round(float(s["probe"]), 1): s.get("track") for s in data.get("segments", [])}
    id_by_key = {e["shazam_key"]: tid for tid, e in entries.items() if e.get("shazam_key")}

    segments, out = [], []
    for probe, track in zip(probes, recognised):
        at = round(probe.start, 1)
        if at in assigned:
            tid = assigned[at]
        elif track is None:
            tid = None
        else:
            tid = id_by_key.get(track.key)
            if tid is None:
                tid = _unique_id(slug(f"{track.artist}-{track.title}").lower(), entries)
                entries[tid] = {"title": track.title, "artist": track.artist,
                                "cover": track.cover_url, "shazam_key": track.key}
                id_by_key[track.key] = tid
        if tid is not None and tid not in entries:
            log.warning("tracks.json: segment %s refers to unknown track id %r; treating as unknown", mmss(at), tid)
            tid = None
        segments.append({"probe": at, "time": mmss(at),
                         "shazam": f"{track.artist} – {track.title}" if track else "no match", "track": tid})
        out.append(_to_track(tid, entries[tid], path.parent) if tid else None)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"_help": HELP, "tracks": entries, "segments": segments}, indent=2, ensure_ascii=False))
    return out


def _unique_id(base: str, taken: dict) -> str:
    tid, n = base, 2
    while tid in taken:
        tid, n = f"{base}-{n}", n + 1
    return tid


def _to_track(tid: str, e: dict, base: Path) -> Track:
    cover = e.get("cover")
    if cover and not cover.startswith(("http://", "https://")):
        p = Path(cover).expanduser()
        cover = str(p if p.is_absolute() else base / p)
    return Track(title=e.get("title") or tid, artist=e.get("artist") or "", cover_url=cover, uid=tid)
