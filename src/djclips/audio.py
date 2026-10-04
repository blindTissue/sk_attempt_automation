"""Thin ffmpeg wrappers for decoding and cutting audio."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def decode_mono(path: Path, sr: int) -> np.ndarray:
    """Decode any ffmpeg-readable file to a mono float32 signal at `sr`."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"],
        check=True, capture_output=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def extract_mp3(path: Path, start: float, duration: float, out_path: Path) -> Path:
    """Cut a small mp3 snippet (used for recognition uploads)."""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(path),
         "-ac", "2", "-ar", "44100", "-b:a", "128k", str(out_path)],
        check=True,
    )
    return out_path
