"""Mux the still image with the audio excerpt into an Instagram-ready MP4
(H.264 High, yuv420p, 30 fps, AAC 48 kHz, faststart)."""

from __future__ import annotations

import subprocess
from pathlib import Path

FADE = 0.5


def make_video(image: Path, audio: Path, start: float, end: float, out_path: Path) -> Path:
    dur = end - start
    afilter = f"afade=t=in:d={FADE},afade=t=out:st={max(0.0, dur - FADE):.3f}:d={FADE}"
    subprocess.run([
        "ffmpeg", "-v", "error", "-y",
        "-loop", "1", "-framerate", "30", "-i", str(image),
        "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(audio),
        "-map", "0:v", "-map", "1:a", "-t", f"{dur:.3f}",
        "-c:v", "libx264", "-preset", "medium", "-tune", "stillimage", "-crf", "20",
        "-profile:v", "high", "-pix_fmt", "yuv420p", "-r", "30", "-g", "60",
        "-af", afilter, "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart", str(out_path),
    ], check=True)
    return out_path
