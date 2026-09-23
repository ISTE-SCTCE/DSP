"""
preprocessing.py — Noise removal, normalisation, framing/windowing
===================================================================
KTU S5 DSP Project: Voice Camouflage Detection

DSP Concepts demonstrated (KTU S5 Unit 4 — Digital Filters):
  - IIR Filter Design via Butterworth prototype (high-pass filter)
  - Z-transform representation of the filter:  H(z) = B(z) / A(z)
  - Spectral Subtraction for noise reduction (Unit 5 — Applications)
  - Overlap-Add framing with Hamming windowing (Unit 2 — DFT properties)

Each function is independently testable.  Run from this file directly:
    python preprocessing.py
"""

import numpy as np
from scipy.signal import butter, sosfilt, get_window, tf2zpk, freqz
from typing import Tuple

from config import (
    SAMPLE_RATE, FRAME_DURATION_MS, HOP_DURATION_MS, WINDOW_TYPE,
    HPF_CUTOFF_HZ, HPF_ORDER,
    NOISE_ESTIMATION_PERCENTILE, SPECTRAL_SUBTRACTION_ALPHA, SPECTRAL_NOISE_FLOOR,
)


# ─── 1. High-Pass Filter (Z-transform / IIR Design) ─────────────────────────

def design_high_pass(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Design a Butterworth high-pass filter and return second-order sections.

    DSP Theory — KTU Unit 4 (Filter Design & Z-transform):
        The Butterworth HPF is derived from an analog prototype H_a(s) via
        the bilinear transform s = (2/T)(1 - z⁻¹)/(1 + z⁻¹).  This maps
        the s-plane unit circle to the z-plane unit circle without aliasing.

        In the z-domain the filter is described by its transfer function:
            H(z) = B(z) / A(z)
        where B(z) and A(z) are polynomials in z⁻¹ (the unit delay operator).
        The poles of H(z) lie inside the unit circle → stable IIR filter.

        We use Second-Order Sections (SOS) representation to avoid numerical
        precision issues from convolving high-order polynomial coefficients.

    Args:
        sr: Sample rate (Hz).

    Returns:
        sos: Array of shape (n_sections, 6) for use with scipy.signal.sosfilt.
    """
    sos = butter(HPF_ORDER, HPF_CUTOFF_HZ, btype="high", fs=sr, output="sos")
    return sos


def get_zpk(sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return zeros, poles, and gain of the HPF for pole-zero diagram plotting.

    Used in visualize.plot_pole_zero() to illustrate Z-transform theory.
    """
    # Get numerator / denominator in polynomial form from SOS
    from scipy.signal import sos2tf
    sos = design_high_pass(sr)
    b, a = sos2tf(sos)
    z, p, k = tf2zpk(b, a)
    return z, p, k


def get_frequency_response(sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, np.ndarray]:
    """Compute H(e^jω) — the frequency response of the HPF.

    Returns:
        freqs: Frequency axis (Hz).
        H:     Complex frequency response.
    """
    sos = design_high_pass(sr)
    from scipy.signal import sos2tf
    b, a = sos2tf(sos)
    w, H = freqz(b, a, worN=512, fs=sr)
    return w, H


def high_pass_filter(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Apply Butterworth HPF to remove DC offset and mains hum below 80 Hz.

    Args:
        y:  Input audio signal (float32).
        sr: Sample rate (Hz).

    Returns:
        Filtered signal as float32.
    """
    sos = design_high_pass(sr)
    return sosfilt(sos, y).astype(np.float32)


# ─── 2. Spectral Subtraction Noise Reduction ─────────────────────────────────

def spectral_subtraction(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Estimate and subtract background noise spectrum from the signal.

    DSP Theory — KTU Unit 5 (Applications / Adaptive Filtering):
        Spectral subtraction estimates the noise power spectral density |N(k)|²
        from the quietest frames (where speech is assumed absent), then subtracts
        it from the noisy speech magnitude spectrum:
            |Ŝ(k)| = max( |X(k)| - α·|N̂(k)|,  β·|X(k)| )
        α (over-subtraction factor) controls aggressiveness.
        β (spectral floor) prevents musical-noise artifacts.
        Phase is unchanged: signal is reconstructed via ISTFT.

    Args:
        y:  Input audio (float32).
        sr: Sample rate (Hz).

    Returns:
        Denoised signal as float32.
    """
    try:
        import librosa
        frame_len = int(sr * FRAME_DURATION_MS / 1000)
        hop_len = int(sr * HOP_DURATION_MS / 1000)

        S = librosa.stft(y, n_fft=frame_len, hop_length=hop_len)
        mag, phase = np.abs(S), np.angle(S)

        # Estimate noise from the quietest NOISE_ESTIMATION_PERCENTILE of frames
        frame_energy = np.sum(mag ** 2, axis=0)
        n_quiet = max(1, int(len(frame_energy) * NOISE_ESTIMATION_PERCENTILE))
        quiet_idx = np.argsort(frame_energy)[:n_quiet]
        noise_profile = np.mean(mag[:, quiet_idx], axis=1, keepdims=True)

        # Spectral subtraction with noise floor
        mag_clean = np.maximum(
            mag - SPECTRAL_SUBTRACTION_ALPHA * noise_profile,
            SPECTRAL_NOISE_FLOOR * mag,
        )

        S_clean = mag_clean * np.exp(1j * phase)
        y_clean = librosa.istft(S_clean, hop_length=hop_len, length=len(y))
        return y_clean.astype(np.float32)
    except Exception as e:
        print(f"[preprocessing] spectral_subtraction fallback: {e}")
        return y.astype(np.float32)


# ─── 3. Amplitude Normalisation ──────────────────────────────────────────────

def peak_normalize(y: np.ndarray) -> np.ndarray:
    """Scale amplitude so the peak is exactly ±1.0.

    Prevents clipping in downstream DSP and makes feature magnitudes
    comparable across recordings with different recording levels.

    Args:
        y: Input signal (any float dtype).

    Returns:
        Normalised signal as float32 in [-1.0, 1.0].
    """
    peak = np.max(np.abs(y))
    if peak < 1e-8:
        return y.astype(np.float32)
    return (y / peak).astype(np.float32)


# ─── 4. Framing / Windowing ──────────────────────────────────────────────────

def compute_frame_params(sr: int = SAMPLE_RATE) -> Tuple[int, int]:
    """Return (frame_length_samples, hop_length_samples) for the configured durations."""
    frame_len = int(sr * FRAME_DURATION_MS / 1000)
    hop_len = int(sr * HOP_DURATION_MS / 1000)
    return frame_len, hop_len


def apply_window(frame: np.ndarray, window_type: str = WINDOW_TYPE) -> np.ndarray:
    """Multiply a single frame by a window function.

    DSP Theory — KTU Unit 2 (DFT Properties / Windowing):
        Taking the DFT of a finite-length signal implicitly assumes the signal
        is periodic. Rectangular windowing (no window) causes spectral leakage
        because the sharp edges introduce high-frequency components.
        The Hamming window:
            w[n] = 0.54 - 0.46·cos(2πn/(N-1))
        tapers the frame edges, reducing side-lobe leakage by ~43 dB at the
        cost of a slightly wider main lobe. This is the standard choice for
        speech analysis.

    Args:
        frame:       1-D signal array of length N.
        window_type: scipy.signal.get_window identifier (default: 'hamming').

    Returns:
        Windowed frame as float32.
    """
    win = get_window(window_type, len(frame))
    return (frame * win).astype(np.float32)


def segment_frames(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Slice signal into overlapping Hamming-windowed frames.

    Frame count formula:
        N_frames = 1 + (len(y) - frame_len) // hop_len

    Args:
        y:  Clean audio signal (float32).
        sr: Sample rate (Hz).

    Returns:
        Array of shape (N_frames, frame_len) — each row is one windowed frame.
    """
    frame_len, hop_len = compute_frame_params(sr)
    window = get_window(WINDOW_TYPE, frame_len)
    n_frames = 1 + (len(y) - frame_len) // hop_len

    if n_frames <= 0:
        return np.zeros((1, frame_len), dtype=np.float32)

    frames = np.zeros((n_frames, frame_len), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_len
        frames[i] = y[start : start + frame_len] * window

    return frames


# ─── 5. Full Preprocessing Pipeline ──────────────────────────────────────────

def run(y: np.ndarray, sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, np.ndarray]:
    """Full preprocessing: HPF → spectral subtraction → peak-normalize → frame.

    Args:
        y:  Raw input audio (float32).
        sr: Sample rate (Hz).

    Returns:
        (y_clean, frames):
            y_clean — denoised, normalised signal (float32, 1-D).
            frames  — windowed frames array (float32, 2-D: N_frames × frame_len).
    """
    y_hpf = high_pass_filter(y, sr)
    y_denoised = spectral_subtraction(y_hpf, sr)
    y_clean = peak_normalize(y_denoised)
    frames = segment_frames(y_clean, sr)
    return y_clean, frames


# ─── Quick self-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    _sr = SAMPLE_RATE
    _t = np.linspace(0, 1.0, _sr, dtype=np.float32)
    _y = 0.5 * np.sin(2 * np.pi * 440 * _t) + 0.05 * np.random.randn(_sr).astype(np.float32)
    _clean, _frames = run(_y, _sr)
    print(f"Input shape : {_y.shape}")
    print(f"Clean shape : {_clean.shape}")
    print(f"Frames shape: {_frames.shape}")
    print("preprocessing.py self-test passed.")
