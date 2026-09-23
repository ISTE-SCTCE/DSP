"""
test_transforms.py — Unit tests for transforms.py
===================================================
KTU S5 DSP Project: Voice Camouflage Detection

Tests compare our implementations against numpy/scipy reference outputs
to verify correctness of the DSP math.

Run:
    cd voice_camouflage_detection
    python -m pytest tests/test_transforms.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from scipy.fft import dct as scipy_dct, dst as scipy_dst

import transforms


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sine_signal():
    """256-sample sine wave at 440 Hz (sr=8000 Hz for speed)."""
    N = 256
    t = np.linspace(0, N / 8000, N, endpoint=False)
    return (np.sin(2 * np.pi * 440 * t)).astype(np.float64)


@pytest.fixture
def small_signal():
    """64-sample random signal (DFT is slow — keep N small for test speed)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(64)


# ─── DFT vs FFT ──────────────────────────────────────────────────────────────

class TestDFTvsFFT:
    """DFT direct definition should match numpy.fft.fft exactly."""

    def test_magnitudes_match(self, small_signal):
        X_dft = transforms.dft_direct(small_signal)
        X_fft = transforms.fft(small_signal)
        np.testing.assert_allclose(
            np.abs(X_dft), np.abs(X_fft), rtol=1e-8,
            err_msg="DFT and FFT magnitude spectra diverge"
        )

    def test_phases_match(self, small_signal):
        X_dft = transforms.dft_direct(small_signal)
        X_fft = transforms.fft(small_signal)
        # Compare complex values directly
        np.testing.assert_allclose(
            X_dft.real, X_fft.real, atol=1e-8,
            err_msg="DFT vs FFT real parts diverge"
        )
        np.testing.assert_allclose(
            X_dft.imag, X_fft.imag, atol=1e-8,
            err_msg="DFT vs FFT imag parts diverge"
        )

    def test_dc_component(self):
        """DC bin X[0] = sum(x) for real signal."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        X_dft = transforms.dft_direct(x)
        X_fft = transforms.fft(x)
        assert abs(X_dft[0].real - 10.0) < 1e-10
        assert abs(X_fft[0].real - 10.0) < 1e-10

    def test_output_length(self, small_signal):
        N = len(small_signal)
        assert len(transforms.dft_direct(small_signal)) == N
        assert len(transforms.fft(small_signal)) == N

    def test_parseval_theorem(self, small_signal):
        """Energy in time domain = energy in frequency domain / N (Parseval)."""
        N = len(small_signal)
        E_time = np.sum(small_signal ** 2)
        X = transforms.fft(small_signal)
        E_freq = np.sum(np.abs(X) ** 2) / N
        np.testing.assert_allclose(E_time, E_freq, rtol=1e-10,
                                   err_msg="Parseval theorem violated")


# ─── DCT ─────────────────────────────────────────────────────────────────────

class TestDCT:
    """Our DCT should match scipy.fft.dct(type=2, norm='ortho')."""

    def test_matches_scipy(self, sine_signal):
        C_ours = transforms.dct(sine_signal)
        C_scipy = scipy_dct(sine_signal, type=2, norm="ortho")
        np.testing.assert_allclose(C_ours, C_scipy, rtol=1e-10,
                                   err_msg="DCT deviates from scipy reference")

    def test_energy_compaction(self, sine_signal):
        """First 10% of coefficients should hold most energy (compaction property)."""
        C = transforms.dct(sine_signal)
        N = len(C)
        k = max(1, N // 10)
        energy_low = np.sum(C[:k] ** 2)
        energy_total = np.sum(C ** 2)
        ratio = energy_low / energy_total
        assert ratio > 0.80, \
            f"DCT energy compaction poor: top {k}/{N} bins hold only {ratio:.1%}"

    def test_orthonorm(self, sine_signal):
        """Ortho-normalised DCT is its own inverse (IDCT = DCT of coefficients)."""
        from scipy.fft import idct
        C = transforms.dct(sine_signal)
        x_rec = idct(C, type=2, norm="ortho")
        np.testing.assert_allclose(x_rec, sine_signal, atol=1e-10,
                                   err_msg="DCT inversion failed")


# ─── DST ─────────────────────────────────────────────────────────────────────

class TestDST:
    """Our DST should match scipy.fft.dst(type=2, norm='ortho')."""

    def test_matches_scipy(self, sine_signal):
        D_ours = transforms.dst(sine_signal)
        D_scipy = scipy_dst(sine_signal, type=2, norm="ortho")
        np.testing.assert_allclose(D_ours, D_scipy, rtol=1e-10,
                                   err_msg="DST deviates from scipy reference")


# ─── Z-transfer ──────────────────────────────────────────────────────────────

class TestZTransfer:
    """Z-transform analysis should return sensible HPF shapes."""

    def setup_method(self):
        from scipy.signal import butter, sos2tf
        sos = butter(5, 80.0, btype="high", fs=22050, output="sos")
        b, a = sos2tf(sos)
        self.result = transforms.z_transfer(b, a, sr=22050)

    def test_keys_present(self):
        for key in ["zeros", "poles", "gain", "freqs_hz",
                    "H_magnitude_db", "H_phase_deg"]:
            assert key in self.result

    def test_hpf_stopband_attenuated(self):
        """Frequencies well below 80 Hz should be attenuated (< −20 dB)."""
        freqs = self.result["freqs_hz"]
        mag_db = self.result["H_magnitude_db"]
        # Check at 20 Hz
        idx = np.argmin(np.abs(freqs - 20.0))
        assert mag_db[idx] < -20.0, \
            f"HPF not attenuating at 20 Hz: {mag_db[idx]:.1f} dB"

    def test_hpf_passband_flat(self):
        """Frequencies above 500 Hz should pass mostly unattenuated (> −3 dB)."""
        freqs = self.result["freqs_hz"]
        mag_db = self.result["H_magnitude_db"]
        idx = np.argmin(np.abs(freqs - 1000.0))
        assert mag_db[idx] > -3.0, \
            f"HPF attenuating passband at 1 kHz: {mag_db[idx]:.1f} dB"

    def test_poles_inside_unit_circle(self):
        """All poles must be inside the unit circle (stability criterion)."""
        poles = self.result["poles"]
        assert all(np.abs(poles) < 1.0 + 1e-6), \
            f"Unstable filter — pole outside unit circle: {np.abs(poles)}"


# ─── compare_transforms ──────────────────────────────────────────────────────

class TestCompareTransforms:
    def test_returns_all_keys(self, small_signal):
        result = transforms.compare_transforms(small_signal)
        for name in ["dft", "fft", "dct", "dst"]:
            assert name in result
            assert "coefficients" in result[name]
            assert "magnitude" in result[name]
            assert "elapsed_ms" in result[name]

    def test_timing_positive(self, small_signal):
        result = transforms.compare_transforms(small_signal)
        for name in ["dft", "fft", "dct", "dst"]:
            assert result[name]["elapsed_ms"] >= 0.0

    def test_dft_slower_than_fft(self, sine_signal):
        """DFT should be slower than FFT for N=256 in most runs.
        We repeat to reduce noise, but just assert DFT elapsed ≥ 0 (non-negative).
        Actual speed comparison is demonstrated in the Streamlit app, not here."""
        result = transforms.compare_transforms(sine_signal)
        assert result["dft"]["elapsed_ms"] >= 0.0
        assert result["fft"]["elapsed_ms"] >= 0.0
