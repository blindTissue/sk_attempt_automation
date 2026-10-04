"""Split the mix into candidate track segments without any API calls, then
decide where to probe the recognizer (one probe per segment, a few more for
suspiciously long segments)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from .features import smooth, unit_rows


@dataclass(frozen=True)
class Probe:
    start: float
    duration: float
    seg_start: float | None = None  # the novelty segment this probe was taken from
    seg_end: float | None = None

    @property
    def end(self) -> float:
        return self.start + self.duration

    @property
    def segment(self) -> tuple[float, float]:
        if self.seg_start is None or self.seg_end is None:
            return self.start, self.end
        return self.seg_start, self.seg_end


def novelty_curve(F: np.ndarray, half: int = 16) -> np.ndarray:
    """Foote novelty: a Gaussian checkerboard kernel slid along the
    diagonal of the cosine self-similarity matrix."""
    Fn = unit_rows(smooth(F, 3))
    T = len(Fn)
    idx = np.arange(-half, half) + 0.5
    gauss = np.exp(-0.5 * (idx / (half / 2)) ** 2)
    sign = np.sign(idx)
    kernel = np.outer(sign, sign) * np.outer(gauss, gauss)
    nov = np.zeros(T)
    for t in range(half, T - half):
        block = Fn[t - half:t + half]
        nov[t] = (kernel * (block @ block.T)).sum()
    nov = np.maximum(nov - np.median(nov[half:T - half] if T > 2 * half else nov), 0)
    peak = nov.max()
    return nov / peak if peak > 0 else nov


def find_boundaries(F: np.ndarray, min_track: float = 60, half: int = 16, prominence: float = 0.15) -> list[int]:
    nov = novelty_curve(F, half)
    peaks, _ = find_peaks(nov, distance=max(1, int(min_track)), prominence=prominence)
    return [int(p) for p in peaks]


def plan_probes(total: float, boundaries: list[int], clip: float = 10, max_segment: float = 360,
                min_segment: float = 20) -> list[Probe]:
    """One probe at the centre of each segment; long segments get evenly
    spaced extra probes in case the novelty curve missed a boundary."""
    edges = [0.0, *map(float, boundaries), float(total)]
    probes = []
    for s, e in zip(edges, edges[1:]):
        length = e - s
        if length < max(min_segment, clip):
            continue
        n = max(1, math.ceil(length / max_segment))
        for k in range(n):
            centre = s + (k + 0.5) * length / n
            start = min(max(0.0, centre - clip / 2), total - clip)
            probes.append(Probe(round(start, 1), clip, s, e))
    return probes
