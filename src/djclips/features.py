"""Per-second melody (chroma) + timbre (MFCC) features of a mix."""

from __future__ import annotations

import librosa
import numpy as np

ANALYSIS_SR = 11025
N_FFT = 2048
HOP = 512
CHUNK_SECONDS = 120


def compute_features(y: np.ndarray, sr: int = ANALYSIS_SR) -> np.ndarray:
    """Return a (seconds, dims) matrix: chroma captures the melody/harmony,
    MFCCs the timbre. Each block is z-scored and weighted equally."""
    per_sec = sr
    total_secs = len(y) // per_sec
    chroma_rows, mfcc_rows = [], []
    for c0 in range(0, total_secs, CHUNK_SECONDS):
        n_secs = min(CHUNK_SECONDS, total_secs - c0)
        seg = y[c0 * per_sec:(c0 + n_secs) * per_sec]
        S = np.abs(librosa.stft(seg, n_fft=N_FFT, hop_length=HOP)) ** 2
        chroma = librosa.feature.chroma_stft(S=S, sr=sr, n_fft=N_FFT)
        mel = librosa.feature.melspectrogram(S=S, sr=sr, n_fft=N_FFT, n_mels=64)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)[1:]
        sec_of_frame = np.minimum((np.arange(S.shape[1]) * HOP) // per_sec, n_secs - 1)
        chroma_rows.append(_mean_by_second(chroma, sec_of_frame, n_secs))
        mfcc_rows.append(_mean_by_second(mfcc, sec_of_frame, n_secs))
    if not chroma_rows:
        raise ValueError("audio is shorter than one second")
    chroma = np.vstack(chroma_rows)
    mfcc = np.vstack(mfcc_rows)
    return np.hstack([_zscore(chroma) / np.sqrt(chroma.shape[1]), _zscore(mfcc) / np.sqrt(mfcc.shape[1])])


def _mean_by_second(feat: np.ndarray, sec_of_frame: np.ndarray, n_secs: int) -> np.ndarray:
    counts = np.bincount(sec_of_frame, minlength=n_secs).astype(float)
    out = np.zeros((n_secs, feat.shape[0]))
    for d in range(feat.shape[0]):
        out[:, d] = np.bincount(sec_of_frame, weights=feat[d], minlength=n_secs)
    return out / np.maximum(counts, 1)[:, None]


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)


def smooth(F: np.ndarray, width: int) -> np.ndarray:
    """Moving average along time, same length."""
    if width <= 1:
        return F
    kernel = np.ones(width) / width
    padded = np.pad(F, ((width // 2, width - 1 - width // 2), (0, 0)), mode="edge")
    return np.apply_along_axis(lambda c: np.convolve(c, kernel, mode="valid"), 0, padded)


def unit_rows(F: np.ndarray) -> np.ndarray:
    return F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
