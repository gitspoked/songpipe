"""Objective, wav-only health metrics for ranking best-of-N renders.

Deliberately NOT whisper-based: word-overlap can't rank good *singing* takes
(melisma/timing). These cheap measures catch clipping, dead air, and flatness.
"""
import numpy as np

WEIGHTS = {"clip": 1.0, "silence": 1.0, "lra": 0.01}

def _mono(x):
    x = np.asarray(x, dtype="float64")
    return x.mean(axis=1) if x.ndim > 1 else x

def clip_ratio(x) -> float:
    x = np.asarray(x)
    return 0.0 if x.size == 0 else float(np.mean(np.abs(x) >= 0.999))

def silence_ratio(x, sr, frame_ms=50, floor_db=-50.0) -> float:
    x = _mono(x)
    if x.size == 0:
        return 1.0
    n = max(1, int(sr * frame_ms / 1000))
    floor = 10.0 ** (floor_db / 20.0)
    frames = [x[i:i+n] for i in range(0, len(x), n)]
    silent = sum(1 for f in frames if np.sqrt(np.mean(f**2)) < floor)
    return float(silent / len(frames))

def loudness_range(x) -> float:
    x = _mono(x)
    if x.size == 0:
        return 0.0
    peak = float(np.max(np.abs(x))); rms = float(np.sqrt(np.mean(x**2)))
    if peak <= 0 or rms <= 0:
        return 0.0
    return float(20.0 * np.log10(peak / rms))  # crest-factor proxy in dB

def combine_scores(clip, silence, lra, weights=None) -> float:
    if weights is None:
        weights = WEIGHTS
    return (-weights["clip"] * clip
            - weights["silence"] * silence
            + weights["lra"] * lra)
