"""
utils.py — Shared helper utilities
====================================
KTU S5 DSP Project: Voice Camouflage Detection

Small, pure functions that are used across multiple modules.
No DSP logic here — only generic data-manipulation helpers.
"""

import io
import base64
import numpy as np
import soundfile as sf
from typing import Optional


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide without ZeroDivisionError. Returns `default` if denominator is near zero."""
    if abs(denominator) < 1e-10:
        return default
    return numerator / denominator


def linear_to_db(power: float, ref: float = 1.0) -> float:
    """Convert power ratio to decibels: 10·log10(power/ref)."""
    return 10.0 * np.log10(safe_divide(power, ref, default=1e-10))


def downsample_1d(arr: np.ndarray, max_points: int) -> np.ndarray:
    """Reduce a 1-D array to at most `max_points` by uniform index selection."""
    if len(arr) <= max_points:
        return arr
    idx = np.linspace(0, len(arr) - 1, max_points, dtype=int)
    return arr[idx]


def downsample_2d(arr: np.ndarray, max_rows: int, max_cols: int) -> np.ndarray:
    """Reduce a 2-D array (rows×cols) by uniform index selection on each axis."""
    rows, cols = arr.shape
    if rows > max_rows:
        r_idx = np.linspace(0, rows - 1, max_rows, dtype=int)
        arr = arr[r_idx, :]
    if cols > max_cols:
        c_idx = np.linspace(0, cols - 1, max_cols, dtype=int)
        arr = arr[:, c_idx]
    return arr


def numpy_to_wav_base64(y: np.ndarray, sr: int) -> str:
    """Encode a float32 numpy audio array as a WAV base64 data-URI.

    Useful for embedding audio directly in JSON responses or HTML without
    writing to disk.

    Args:
        y: Audio signal, float32, values expected in [-1.0, 1.0].
        sr: Sample rate in Hz.

    Returns:
        A data-URI string: ``data:audio/wav;base64,<encoded>``.
    """
    try:
        buf = io.BytesIO()
        y_clipped = np.clip(y, -1.0, 1.0).astype(np.float32)
        sf.write(buf, y_clipped, sr, format="WAV", subtype="PCM_16")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:audio/wav;base64,{b64}"
    except Exception as e:
        print(f"[utils] numpy_to_wav_base64 error: {e}")
        return ""


def frames_to_time(frame_indices: np.ndarray, hop_len: int, sr: int) -> np.ndarray:
    """Convert frame indices to time stamps in seconds."""
    return frame_indices * hop_len / sr


def next_power_of_two(n: int) -> int:
    """Return the smallest power of 2 that is >= n (used for zero-padding FFT)."""
    p = 1
    while p < n:
        p <<= 1
    return p


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a scalar to [lo, hi]."""
    return max(lo, min(hi, value))
