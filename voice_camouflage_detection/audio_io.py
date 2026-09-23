"""
audio_io.py — Audio loading, recording, and format conversion
=============================================================
KTU S5 DSP Project: Voice Camouflage Detection

Responsible for getting audio into the system as a normalised numpy float32
array. No DSP processing happens here — that belongs in preprocessing.py.

DSP Concept: Sampling theorem (Nyquist).  All audio is resampled to a
common sample rate (config.SAMPLE_RATE) so every downstream module can
assume a fixed fs.  The Nyquist frequency = SR / 2 = 11 025 Hz covers
the full human speech range (80 Hz – 8 kHz).
"""

import io
import numpy as np
import soundfile as sf
import librosa
from typing import Optional, Tuple

from config import SAMPLE_RATE, MAX_DURATION_S, MIN_DURATION_S


# ─── File Loading ────────────────────────────────────────────────────────────

def load_file(path: str, sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Load an audio file from disk and resample to `sr`.

    Supports WAV, MP3, FLAC, M4A via librosa/soundfile/audioread.

    Args:
        path: Absolute or relative path to the audio file.
        sr:   Target sample rate (Hz).  Defaults to config.SAMPLE_RATE.

    Returns:
        (y, sr) where y is float32 mono signal in [-1.0, 1.0].

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the loaded clip is shorter than MIN_DURATION_S.
    """
    y, loaded_sr = librosa.load(path, sr=sr, mono=True)
    return _validate_and_clip(y, sr)


def load_bytes(audio_bytes: bytes, sr: int = SAMPLE_RATE,
               suffix: str = ".wav") -> Tuple[np.ndarray, int]:
    """Load audio from an in-memory bytes buffer (e.g. from Streamlit uploader).

    Args:
        audio_bytes: Raw audio file content as bytes.
        sr:          Target sample rate (Hz).
        suffix:      File extension hint (e.g. ".mp3"), used by soundfile
                     when format cannot be auto-detected.

    Returns:
        (y, sr) float32 mono array.
    """
    buf = io.BytesIO(audio_bytes)
    try:
        y, loaded_sr = librosa.load(buf, sr=sr, mono=True)
    except Exception:
        # Fallback: try soundfile for WAV/FLAC
        buf.seek(0)
        data, loaded_sr = sf.read(buf, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        y = librosa.resample(data, orig_sr=loaded_sr, target_sr=sr)
    return _validate_and_clip(y, sr)


def record_mic(duration_s: float = 5.0, sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Record from the default microphone for `duration_s` seconds.

    Requires the `sounddevice` package (``pip install sounddevice``).

    Args:
        duration_s: Recording length in seconds.
        sr:         Sample rate (Hz).

    Returns:
        (y, sr) float32 mono array.
    """
    try:
        import sounddevice as sd  # optional dependency
    except ImportError as e:
        raise ImportError(
            "sounddevice is required for microphone recording: pip install sounddevice"
        ) from e

    samples = int(duration_s * sr)
    recording = sd.rec(samples, samplerate=sr, channels=1, dtype="float32")
    sd.wait()  # block until done
    y = recording.flatten()
    return _validate_and_clip(y, sr)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _validate_and_clip(y: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    """Validate length constraints and clip to MAX_DURATION_S."""
    if len(y) < int(MIN_DURATION_S * sr):
        raise ValueError(
            f"Audio too short ({len(y)/sr:.2f}s). Minimum {MIN_DURATION_S}s required."
        )
    max_samples = int(MAX_DURATION_S * sr)
    if len(y) > max_samples:
        y = y[:max_samples]
    return y.astype(np.float32), sr


def duration_seconds(y: np.ndarray, sr: int) -> float:
    """Compute signal duration in seconds from sample count."""
    return len(y) / sr
