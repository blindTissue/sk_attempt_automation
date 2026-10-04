"""Locate each transition precisely by comparing every second of audio to
reference material from the outgoing (A) and incoming (B) track."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import smooth, unit_rows
from .recognize import Track
from .segment import Probe


@dataclass
class Occurrence:
    """A stretch of the mix identified as one track, via one or more probes."""
    track: Track
    anchors: list[Probe]
    after_gap: bool = False  # an unidentified track sits between this and the previous occurrence

    @property
    def first(self) -> Probe:
        return self.anchors[0]

    @property
    def last(self) -> Probe:
        return self.anchors[-1]


@dataclass
class Transition:
    a: Track
    b: Track
    start: float       # where B starts to be audible / A stops being "pure"
    crossover: float   # where B takes over from A
    end: float         # where only B remains
    clip_start: float
    clip_end: float

    def to_dict(self) -> dict:
        return {
            "from": self.a.to_dict(), "to": self.b.to_dict(),
            "transition": [round(self.start, 1), round(self.end, 1)],
            "crossover": round(self.crossover, 1),
            "clip": [round(self.clip_start, 1), round(self.clip_end, 1)],
        }


def group_occurrences(probes: list[Probe], tracks: list[Track | None]) -> list[Occurrence]:
    """Collapse consecutive probes that recognised the same track. An
    unidentified probe between two different tracks marks a gap: there is no
    direct transition across it."""
    occ: list[Occurrence] = []
    gap = False
    for probe, track in zip(probes, tracks):
        if track is None:
            gap = bool(occ)
            continue
        if occ and occ[-1].track.key == track.key:
            occ[-1].anchors.append(probe)
        else:
            occ.append(Occurrence(track, [probe], after_gap=gap))
        gap = False
    return occ


def assign_unknowns(F: np.ndarray, probes: list[Probe], tracks: list[Track | None]) -> list[Track | None]:
    """Give each unrecognised segment to a neighbouring identified track if it
    sounds like that track (e.g. a breakdown Shazam couldn't match); otherwise
    leave it unknown. Free: uses local features only."""
    Fn = unit_rows(smooth(F, 3))
    T = len(F)
    out = list(tracks)
    known = [i for i, t in enumerate(tracks) if t is not None]
    for i, t in enumerate(tracks):
        if t is not None:
            continue
        frames = _segment_frames([probes[i]], T)
        best, best_ratio = None, 1.0
        for j in (max((k for k in known if k < i), default=None), min((k for k in known if k > i), default=None)):
            if j is None:
                continue
            ref = _segment_frames([p for p, tr in zip(probes, tracks) if tr is not None and tr.key == tracks[j].key
                                   and abs(probes.index(p) - j) <= 3], T)
            if len(ref) < 5 or len(frames) < 5:
                continue
            ceiling = _noise_ceiling(_nn_distance(Fn, ref, ref))
            ratio = float(np.median(_nn_distance(Fn, frames, ref))) / ceiling
            if ratio <= best_ratio:
                best, best_ratio = tracks[j], ratio
        out[i] = best
    return out


def _nn_distance(Fn: np.ndarray, times: np.ndarray, ref: np.ndarray, k: int = 3, exclude: int = 4) -> np.ndarray:
    """Mean cosine distance of each frame in `times` to its k nearest frames in
    `ref`. Frames within `exclude` seconds of each other are not compared, so
    reference frames get a realistic (not near-zero) self-distance."""
    d = 1 - Fn[times] @ Fn[ref].T
    if len(ref) > 2 * exclude + 1 + k:  # only when enough frames remain to compare against
        d[np.abs(times[:, None] - ref[None, :]) <= exclude] = np.inf
    k = min(k, max(1, len(ref) - 1))
    return np.sort(d, axis=1)[:, :k].mean(axis=1)


EDGE = 15        # seconds trimmed off each side of a segment before using it as a reference
SPAN = 30        # how far from a segment boundary a transition may extend


def _segment_frames(probes: list[Probe], T: int, edge: float = EDGE) -> np.ndarray:
    """Frames of the probes' segments, trimmed away from the boundaries where
    the neighbouring track may already be audible."""
    frames = set()
    for p in probes:
        s, e = p.segment
        if e - s - 2 * edge < 10:
            s, e = p.start, p.end
        else:
            s, e = s + edge, e - edge
        frames.update(range(int(max(s, 0)), int(min(e, T))))
    return np.array(sorted(frames), dtype=int)


def _snippet_frames(probes: list[Probe], lo: float, hi: float, pad: float, T: int) -> np.ndarray:
    frames = set()
    for p in probes:
        frames.update(range(int(max(lo, p.start - pad, 0)), int(min(hi, p.end + pad, T))))
    return np.array(sorted(frames), dtype=int)


def locate(F: np.ndarray, a: Occurrence, b: Occurrence) -> tuple[float, float, float]:
    """Return (start, crossover, end) seconds of the A→B transition."""
    T = len(F)
    Fn = unit_rows(smooth(F, 3))
    if a.last.segment == b.first.segment:
        # Both probes fell in one segment (segmentation missed the boundary):
        # search between the probes, using audio around each probe as reference.
        w0, w1 = a.last.end, b.first.start
        ref_a = _snippet_frames(a.anchors, 0, w0, 20, T)
        ref_b = _snippet_frames(b.anchors, w1, T, 20, T)
    else:
        w0 = max(a.last.end, a.last.segment[1] - SPAN)
        w1 = min(b.first.start, b.first.segment[0] + SPAN)
        ref_a = _segment_frames(a.anchors, T)
        ref_b = _segment_frames(b.anchors, T)
    if w1 - w0 < 2 or len(ref_a) < 2 or len(ref_b) < 2:
        mid = (w0 + w1) / 2
        return mid, mid, mid
    lo_t = int(max(0, min(ref_a.min(), w0)))
    hi_t = int(min(T, max(ref_b.max() + 1, w1 + 1)))

    times = np.arange(lo_t, hi_t)
    da = _smooth1d(_nn_distance(Fn, times, ref_a))  # how unlike A each second sounds
    db = _smooth1d(_nn_distance(Fn, times, ref_b))
    in_a, in_b = np.isin(times, ref_a), np.isin(times, ref_b)
    s = da / (da + db + 1e-9)                       # 0 = sounds like A, 1 = sounds like B
    level_a, level_b = np.median(s[in_a]), np.median(s[in_b])
    if level_b - level_a < 0.05:                    # tracks indistinguishable: fall back to window centre
        mid = (w0 + w1) / 2
        return mid, mid, mid

    # Crossover: best single step fit of s inside the window.
    win = (times >= w0) & (times <= w1)
    gain = np.where(win, (level_a + level_b) / 2 - s, 0)
    before = np.concatenate([[0], np.cumsum(gain)])[:-1]
    score = before - (gain.sum() - before)          # A-likeness before minus after
    candidates = np.where(win)[0]
    ci = candidates[np.argmax(score[candidates])]

    # Start: walking back from the crossover, the point where the audio drops
    # to A's own noise floor (B no longer audible). End: symmetric for B.
    thr_a = _noise_ceiling(da[in_a])
    thr_b = _noise_ceiling(db[in_b])
    si = ci
    while si > 0 and times[si - 1] >= w0 and da[si - 1] > thr_a:
        si -= 1
    ei = ci
    while ei < len(times) - 1 and times[ei + 1] <= w1 and db[ei + 1] > thr_b:
        ei += 1
    return float(times[si]), float(times[ci]), float(times[ei] + 1)


def _smooth1d(x: np.ndarray, width: int = 5) -> np.ndarray:
    return np.convolve(np.pad(x, width // 2, mode="edge"), np.ones(width) / width, mode="valid")


def _noise_ceiling(d: np.ndarray) -> float:
    """Distance above which audio no longer sounds like the track: twice its
    typical self-distance (median, so breakdowns don't inflate it)."""
    return float(np.median(d) * 2 + 1e-3)


def clip_window(start: float, crossover: float, end: float, total: float,
                pad: float = 10, max_clip: float = 60) -> tuple[float, float, float, float]:
    """Pad the transition and cap the total length, keeping the crossover in
    view. Returns (transition_start, transition_end, clip_start, clip_end)."""
    max_trans = max_clip - 2 * pad
    if end - start > max_trans:
        s = min(max(start, crossover - max_trans / 2), end - max_trans)
        start, end = s, s + max_trans
    clip_start = max(0.0, start - pad)
    clip_end = min(total, end + pad)
    return start, end, clip_start, clip_end


def find_transitions(F: np.ndarray, occurrences: list[Occurrence], total: float,
                     pad: float = 10, max_clip: float = 60) -> list[Transition]:
    out = []
    for a, b in zip(occurrences, occurrences[1:]):
        if b.after_gap:
            continue
        start, cross, end = locate(F, a, b)
        start, end, cs, ce = clip_window(start, cross, end, total, pad, max_clip)
        out.append(Transition(a.track, b.track, start, cross, end, cs, ce))
    return out
