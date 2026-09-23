"""
transforms.py — DFT, FFT, DCT, DST, Z-transform utilities
===========================================================
KTU S5 DSP Project: Voice Camouflage Detection

This module is the mathematical heart of the project. Every transform is
implemented in a way that exposes the underlying DSP theory, not just
calls a black-box function.

KTU S5 DSP Syllabus mapping:
    Unit 2  — DFT and its properties            → dft_direct(), fft()
    Unit 3  — FFT algorithms (radix-2 DIT)      → fft() docstring
    Unit 4  — Digital filter design, Z-transform → z_transfer()
    Unit 5  — Applications (speech processing)  → dct(), dst(), compare_transforms()

Run standalone to see a self-test with timing comparison:
    python transforms.py
"""

import time
import numpy as np
from scipy.fft import dct as scipy_dct, dst as scipy_dst
from scipy.signal import tf2zpk, freqz, sos2tf
from typing import Dict, Tuple

from config import FREQZ_N_POINTS


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DFT — Direct O(N²) Definition (Educational Reference)
# ═══════════════════════════════════════════════════════════════════════════════

def dft_direct(x: np.ndarray) -> np.ndarray:
    """Compute the N-point DFT using the direct O(N²) definition.

    DSP Theory — KTU Unit 2 (DFT):
        The Discrete Fourier Transform is defined as:

            X[k] = Σ_{n=0}^{N-1}  x[n] · e^{−j2πkn/N}   for k = 0, 1, …, N−1

        where the complex exponential W_N = e^{−j2π/N} is called the twiddle
        factor.  For each of the N output bins we sum N multiplications →
        complexity is O(N²).

        For N = 512 this is ~262 144 multiplications.
        For N = 512 using FFT (below) it is only ~4 608  → 57× faster.

    This function is intentionally slow and is only used for:
        a) Teaching / verifying the FFT result
        b) The "Compare Transforms" tab to show the O(N²) timing difference

    Args:
        x: Real or complex input signal of length N.

    Returns:
        X: Complex DFT output of length N.
    """
    x = np.asarray(x, dtype=complex)
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))            # column vector for broadcasting
    W = np.exp(-2j * np.pi * k * n / N)   # N×N twiddle factor matrix
    return W @ x                     # matrix-vector product → O(N²)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FFT — Fast Fourier Transform (Radix-2 DIT, O(N log N))
# ═══════════════════════════════════════════════════════════════════════════════

def fft(x: np.ndarray) -> np.ndarray:
    """Compute the N-point FFT using numpy's radix-2 implementation.

    DSP Theory — KTU Unit 3 (FFT Algorithms):
        The Cooley-Tukey radix-2 Decimation-In-Time (DIT) FFT exploits the
        periodicity and symmetry of the twiddle factor:
            W_N^{k+N/2} = −W_N^k
        to recursively decompose an N-point DFT into two N/2-point DFTs of
        even- and odd-indexed samples.  This halves the work at each stage.

        Recursion depth = log₂(N) stages, each requiring N/2 butterfly ops:
            Complexity = (N/2)·log₂(N) complex multiplications → O(N log N)

        For N = 512: DFT needs 262 144 mults; FFT needs only 2 304 — 114× fewer.

        numpy.fft.fft implements an optimised version (mixed-radix, FFTPACK),
        which also handles non-power-of-2 sizes efficiently.

    Args:
        x: Real or complex input array of length N.

    Returns:
        X: Complex FFT output of length N.
    """
    return np.fft.fft(x)


def rfft(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the one-sided FFT of a real signal and return (magnitudes, freqs).

    For real x of length N, only N//2+1 unique frequency bins exist
    (the spectrum is conjugate-symmetric).  This is used in the feature
    pipeline for efficiency.

    Returns:
        magnitudes: |X[k]| for k = 0 … N//2
        freq_bins:  Bin index array (0 … N//2)
    """
    X = np.fft.rfft(x)
    return np.abs(X), np.arange(len(X))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. DCT — Discrete Cosine Transform (Type-II, used in MFCC / compression)
# ═══════════════════════════════════════════════════════════════════════════════

def dct(x: np.ndarray, norm: str = "ortho") -> np.ndarray:
    """Compute the Type-II Discrete Cosine Transform of x.

    DSP Theory — KTU Unit 2 (Transform Properties) / Unit 5 (Applications):
        The DCT-II is defined as:
            C[k] = Σ_{n=0}^{N-1}  x[n] · cos(π·k·(2n+1) / (2N))

        Why DCT instead of DFT for MFCCs and audio compression?
        ─────────────────────────────────────────────────────────
        1. REAL-VALUED output: No imaginary part to discard — all N coefficients
           carry meaningful energy information.
        2. ENERGY COMPACTION: For most natural signals (including speech), the
           DCT packs ~95 % of the signal energy into the first few low-frequency
           coefficients.  The higher coefficients are near zero and can be
           discarded (basis of MP3/AAC compression and MFCC truncation to 13).
        3. DECORRELATION: The DCT approximately diagonalises the autocorrelation
           matrix of speech → adjacent coefficients are nearly uncorrelated,
           which is the ideal input for a classifier.

        scipy.fft.dct implements the type-II transform with optional
        orthonormal normalisation (norm="ortho") so that the inverse
        (IDCT) is simply the transpose.

    Args:
        x:    Input signal (1-D real array).
        norm: Normalisation mode.  "ortho" gives energy-preserving transform.

    Returns:
        C: DCT-II coefficients, same length as x.
    """
    return scipy_dct(x, type=2, norm=norm)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DST — Discrete Sine Transform (Type-II, alternative/comparison transform)
# ═══════════════════════════════════════════════════════════════════════════════

def dst(x: np.ndarray, norm: str = "ortho") -> np.ndarray:
    """Compute the Type-II Discrete Sine Transform of x.

    DSP Theory — KTU Unit 2 (Transforms):
        The DST-II uses sine basis functions instead of cosine:
            D[k] = Σ_{n=0}^{N-1}  x[n] · sin(π·k·(2n+1) / (2N))

        The DST assumes odd-symmetric extension of the signal (signal = 0
        at boundary), whereas the DCT assumes even-symmetric extension
        (zero slope at boundary).  For most speech signals, the DCT
        assumption is a better match → DCT is preferred for MFCCs.

        The DST is included here as a comparison transform to demonstrate
        how basis function choice affects spectral representation.

    Args:
        x:    Input signal (1-D real array).
        norm: Normalisation mode.

    Returns:
        D: DST-II coefficients, same length as x.
    """
    return scipy_dst(x, type=2, norm=norm)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Z-transform Utilities — Filter Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def z_transfer(b: np.ndarray, a: np.ndarray, sr: int,
               n_points: int = FREQZ_N_POINTS) -> Dict:
    """Analyse a digital filter defined by transfer function H(z) = B(z)/A(z).

    DSP Theory — KTU Unit 4 (Z-transform and Filter Design):
        Every digital filter can be described by its Z-domain transfer function:
            H(z) = B(z) / A(z)
                 = (b₀ + b₁z⁻¹ + … + bₘz⁻ᴹ) / (1 + a₁z⁻¹ + … + aₙz⁻ᴺ)

        Key analysis:
        ─────────────────────────────────────────────────────────────────
        • POLES (roots of A(z)): determine stability and resonances.
          A causal IIR filter is BIBO stable iff ALL poles lie INSIDE
          the unit circle |z| = 1 in the z-plane.

        • ZEROS (roots of B(z)): determine nulls in the frequency response.
          A zero ON the unit circle creates infinite attenuation at that frequency.

        • Frequency Response H(e^{jω}): obtained by evaluating H(z) on the
          unit circle z = e^{jω}, ω ∈ [0, π).
          scipy.signal.freqz does this efficiently.

    This function is used by visualize.plot_pole_zero() and
    visualize.plot_freq_response() to show Z-transform theory directly.

    Args:
        b:        Numerator polynomial coefficients (MA part).
        a:        Denominator polynomial coefficients (AR part).
        sr:       Sample rate for frequency axis labelling.
        n_points: Number of frequency evaluation points.

    Returns:
        dict with keys:
            zeros, poles, gain  — for pole-zero plot
            freqs_hz            — frequency axis in Hz
            H_magnitude_db      — |H(e^jω)| in dB
            H_phase_deg         — ∠H(e^jω) in degrees
    """
    z, p, k = tf2zpk(b, a)
    w, H = freqz(b, a, worN=n_points, fs=sr)

    return {
        "zeros": z,
        "poles": p,
        "gain": k,
        "freqs_hz": w,
        "H_magnitude_db": 20 * np.log10(np.abs(H) + 1e-12),
        "H_phase_deg": np.angle(H, deg=True),
    }


def z_transfer_from_sos(sos: np.ndarray, sr: int,
                        n_points: int = FREQZ_N_POINTS) -> Dict:
    """Wrapper: convert SOS filter to (b, a) then call z_transfer.

    SOS (Second-Order Sections) is the preferred numerically stable
    representation; we convert to polynomial form only for analysis.
    """
    b, a = sos2tf(sos)
    return z_transfer(b, a, sr, n_points)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Compare Transforms — Side-by-side with timing
# ═══════════════════════════════════════════════════════════════════════════════

def compare_transforms(frame: np.ndarray) -> Dict:
    """Run DFT, FFT, DCT, DST on the same audio frame and record timings.

    This is used by the Streamlit "Compare Transforms" tab to give the
    professor a direct O(N²) vs O(N log N) performance comparison.

    Args:
        frame: Single audio frame (1-D float array of length N).

    Returns:
        dict with keys: dft, fft, dct, dst (each containing coefficients,
        magnitudes, and elapsed_ms).
    """
    results = {}

    for name, fn in [
        ("dft", dft_direct),
        ("fft", fft),
        ("dct", dct),
        ("dst", dst),
    ]:
        t0 = time.perf_counter()
        coeff = fn(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        results[name] = {
            "coefficients": np.real(coeff),          # real part (DCT/DST are already real)
            "magnitude": np.abs(coeff),
            "elapsed_ms": round(elapsed_ms, 4),
            "n": len(frame),
        }

    return results


# ─── Self-test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    N = 64                               # small N for DFT (keep it fast for test)
    t = np.linspace(0, 1.0, N, endpoint=False)
    x = np.sin(2 * np.pi * 5 * t) + 0.3 * np.sin(2 * np.pi * 13 * t)

    X_dft = dft_direct(x)
    X_fft = fft(x)

    max_err = np.max(np.abs(X_dft - X_fft))
    print(f"DFT vs FFT max error: {max_err:.2e}  (should be < 1e-10)")
    assert max_err < 1e-8, "DFT and FFT disagree!"

    comparison = compare_transforms(x)
    for name, res in comparison.items():
        print(f"  {name.upper():4s}  elapsed={res['elapsed_ms']:.4f} ms")

    print("transforms.py self-test passed.")
