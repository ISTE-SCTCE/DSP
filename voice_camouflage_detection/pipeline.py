"""
pipeline.py — Full DSP Pipeline Orchestrator
============================================
KTU S5 DSP Project: Voice Camouflage Detection

Wires together all modules:
    audio_io → preprocessing → transforms → features → classifier

Returns a single PipelineResult dataclass containing every intermediate
array/value so that app.py can source all numbers from one object,
guaranteeing charts and text always show the same computed values.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import audio_io
import preprocessing
import transforms as tfm
import features as feat
from classifier import HybridClassifier
from utils import downsample_1d, downsample_2d, numpy_to_wav_base64
from config import (
    SAMPLE_RATE,
    WAVEFORM_MAX_POINTS, SPEC_MAX_TIME, SPEC_MAX_FREQ, MFCC_MAX_TIME,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Result Dataclass
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineResult:
    """Single structured result from the full DSP pipeline.

    app.py reads from this object to render every plot and metric.
    No signal-processing math appears in app.py — all values come from here.
    """
    # ─ Input info ──────────────────────────────────────────────────────────
    filename: str = ""
    duration_s: float = 0.0
    sample_rate: int = SAMPLE_RATE

    # ─ Stage 1: Pre-processing ─────────────────────────────────────────────
    waveform_raw: List[float] = field(default_factory=list)    # downsampled
    waveform_clean: List[float] = field(default_factory=list)  # downsampled
    waveform_raw_full: Any = None    # full numpy array (for audio playback)
    waveform_clean_full: Any = None  # full numpy array (for audio playback)
    n_frames: int = 0

    # Base64 WAV data URIs for browser/Streamlit audio widgets
    original_wav_b64: str = ""
    clean_wav_b64: str = ""

    # HPF filter analysis (Z-transform)
    hpf_analysis: Dict = field(default_factory=dict)

    # ─ Stage 2: Transforms ─────────────────────────────────────────────────
    # Spectrogram
    spectrogram_db: List[List[float]] = field(default_factory=list)  # downsampled
    spectrogram_freqs: List[float] = field(default_factory=list)
    spectrogram_times: List[float] = field(default_factory=list)
    spectrogram_full: Any = None   # full magnitude_db array

    # MFCC
    mfcc_matrix: List[List[float]] = field(default_factory=list)  # downsampled
    mfcc_means: List[float] = field(default_factory=list)
    mfcc_stds: List[float] = field(default_factory=list)
    mfcc_delta_energy: float = 0.0
    mfcc_full: Any = None   # full (N_MFCC, T) array

    # Pitch
    pitch_f0: List[float] = field(default_factory=list)
    pitch_times: List[float] = field(default_factory=list)
    pitch_stats: Dict = field(default_factory=dict)

    # HNR
    hnr_db: float = 0.0
    hnr_2_4khz_db: float = 0.0

    # ZCR, STE, Rolloff
    zcr_mean: float = 0.0
    zcr_std: float = 0.0
    zcr_array: List[float] = field(default_factory=list)
    ste_mean: float = 0.0
    ste_std: float = 0.0
    ste_array: List[float] = field(default_factory=list)
    rolloff_mean: float = 0.0
    rolloff_array: List[float] = field(default_factory=list)

    # ─ Stage 3: Classification ─────────────────────────────────────────────
    label: str = "No Input"
    confidence: float = 0.0
    ai_percentage: float = 0.0
    human_percentage: float = 0.0
    anomalies: List[str] = field(default_factory=list)
    scores: Dict = field(default_factory=dict)
    segments: Dict = field(default_factory=dict)
    ml_prediction: Dict = field(default_factory=dict)

    # ─ Benchmarks ──────────────────────────────────────────────────────────
    prep_ms: float = 0.0
    features_ms: float = 0.0
    classifier_ms: float = 0.0
    total_ms: float = 0.0

    # ─ Raw feature dict (for SVM training / feedback) ──────────────────────
    _raw_features: Dict = field(default_factory=dict, repr=False)

    def to_app_dict(self) -> Dict:
        """Serialise to a plain dict suitable for Streamlit state / JSON."""
        return {
            "filename": self.filename,
            "duration_s": self.duration_s,
            "sample_rate": self.sample_rate,
            "waveform": {"data": self.waveform_clean, "raw_data": self.waveform_raw,
                         "sr": self.sample_rate},
            "spectrogram": {"data": self.spectrogram_db,
                            "freqs": self.spectrogram_freqs,
                            "times": self.spectrogram_times},
            "mfcc_data": {"matrix": self.mfcc_matrix, "means": self.mfcc_means},
            "pitch_contour": {"f0": self.pitch_f0, "times": self.pitch_times,
                              "stats": self.pitch_stats},
            "features": {
                "mfcc_mean": self.mfcc_means,
                "mfcc_std": self.mfcc_stds,
                "mfcc_delta_energy": self.mfcc_delta_energy,
                "pitch_stats": self.pitch_stats,
                "hnr": {"hnr_db": self.hnr_db, "hnr_2_4khz_db": self.hnr_2_4khz_db},
                "ste": {"mean": self.ste_mean, "std": self.ste_std},
                "zcr": {"mean": self.zcr_mean, "std": self.zcr_std},
                "rolloff_mean": self.rolloff_mean,
            },
            "classification": {
                "label": self.label,
                "confidence": self.confidence,
                "ai_percentage": self.ai_percentage,
                "human_percentage": self.human_percentage,
                "anomalies": self.anomalies,
                "scores": self.scores,
                "segments": self.segments,
                "ml_prediction": self.ml_prediction,
            },
            "benchmarks": {
                "prep_ms": self.prep_ms,
                "features_ms": self.features_ms,
                "classification_ms": self.classifier_ms,
                "total_ms": self.total_ms,
                "steps": [
                    {"name": "Stage 1: Pre-processing", "time_ms": self.prep_ms},
                    {"name": "Stage 2: Feature Extraction", "time_ms": self.features_ms},
                    {"name": "Stage 3: Classification", "time_ms": self.classifier_ms},
                ],
            },
            "audio_files": {
                "original_wav": self.original_wav_b64,
                "noise_reduced_wav": self.clean_wav_b64,
            },
            "hpf_analysis": self.hpf_analysis,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Runner
# ═══════════════════════════════════════════════════════════════════════════════

_classifier = HybridClassifier()   # singleton so model is loaded once


def run(y: np.ndarray, sr: int = SAMPLE_RATE,
        filename: str = "") -> PipelineResult:
    """Run the full 3-stage DSP pipeline on a raw audio array.

    Stages:
        1. Preprocessing  — HPF, spectral subtraction, normalisation, framing
        2. Feature Extraction — STFT, MFCC, Pitch, HNR, STE, ZCR, Rolloff
        3. Classification — Hybrid (rules + SVM + ExtraTrees)

    Args:
        y:        Raw audio signal (float32).
        sr:       Sample rate (Hz).
        filename: Original file name for display.

    Returns:
        PipelineResult dataclass with every intermediate value.
    """
    result = PipelineResult(filename=filename,
                            duration_s=round(len(y) / sr, 2),
                            sample_rate=sr)

    # ── Stage 1: Pre-processing ─────────────────────────────────────────────
    t0 = time.perf_counter()
    y_clean, frames = preprocessing.run(y, sr)
    result.prep_ms = round((time.perf_counter() - t0) * 1000, 1)

    result.n_frames = frames.shape[0]
    result.waveform_raw = downsample_1d(y, WAVEFORM_MAX_POINTS).tolist()
    result.waveform_clean = downsample_1d(y_clean, WAVEFORM_MAX_POINTS).tolist()
    result.waveform_raw_full = y
    result.waveform_clean_full = y_clean

    result.original_wav_b64 = numpy_to_wav_base64(y, sr)
    result.clean_wav_b64 = numpy_to_wav_base64(y_clean, sr)

    # Z-transform analysis of HPF (for filter theory tab)
    sos = preprocessing.design_high_pass(sr)
    from scipy.signal import sos2tf
    b, a = sos2tf(sos)
    result.hpf_analysis = tfm.z_transfer(b, a, sr)

    # ── Stage 2: Feature Extraction ─────────────────────────────────────────
    t1 = time.perf_counter()
    all_features = feat.extract_all(y_clean, frames, sr)
    result.features_ms = round((time.perf_counter() - t1) * 1000, 1)

    # Spectrogram
    spec = all_features["stft"]
    spec_ds = downsample_2d(spec["magnitude_db"], SPEC_MAX_FREQ, SPEC_MAX_TIME)
    result.spectrogram_db = spec_ds.tolist()
    result.spectrogram_freqs = downsample_1d(spec["freqs"], SPEC_MAX_FREQ).tolist()
    result.spectrogram_times = downsample_1d(spec["times"], SPEC_MAX_TIME).tolist()
    result.spectrogram_full = spec["magnitude_db"]

    # MFCC
    mfcc_info = all_features["mfcc"]
    mfcc_ds = downsample_2d(mfcc_info["mfcc"], 13, MFCC_MAX_TIME)
    result.mfcc_matrix = mfcc_ds.tolist()
    result.mfcc_means = mfcc_info["mfcc_means"]
    result.mfcc_stds = mfcc_info["mfcc_stds"]
    result.mfcc_delta_energy = mfcc_info["delta_energy"]
    result.mfcc_full = mfcc_info["mfcc"]

    # Pitch
    pitch_info = all_features["pitch"]
    result.pitch_f0 = pitch_info["f0"].tolist()
    result.pitch_times = pitch_info["times"].tolist()
    result.pitch_stats = pitch_info["stats"]

    # HNR
    result.hnr_db = all_features["hnr"]["hnr_db"]
    result.hnr_2_4khz_db = all_features["hnr"]["hnr_2_4khz_db"]

    # ZCR, STE, Rolloff
    zcr_info = all_features["zcr"]
    result.zcr_mean = zcr_info["zcr_mean"]
    result.zcr_std = zcr_info["zcr_std"]
    result.zcr_array = zcr_info["zcr"].tolist()

    ste_info = all_features["ste"]
    result.ste_mean = ste_info["ste_mean"]
    result.ste_std = ste_info["ste_std"]
    result.ste_array = ste_info["ste"].tolist()

    rolloff_info = all_features["rolloff"]
    result.rolloff_mean = rolloff_info["rolloff_mean"]
    result.rolloff_array = rolloff_info["rolloff"].tolist()

    result._raw_features = all_features

    # ── Stage 3: Classification ─────────────────────────────────────────────
    t2 = time.perf_counter()
    clf = _classifier.predict(all_features, y_clean, sr)
    result.classifier_ms = round((time.perf_counter() - t2) * 1000, 1)

    result.label = clf["label"]
    result.confidence = clf["confidence"]
    result.ai_percentage = clf["ai_percentage"]
    result.human_percentage = clf["human_percentage"]
    result.anomalies = clf["anomalies"]
    result.scores = clf["scores"]
    result.segments = clf["segments"]
    result.ml_prediction = clf["ml_prediction"]

    result.total_ms = round(result.prep_ms + result.features_ms + result.classifier_ms, 1)

    return result


def run_from_file(path: str, sr: int = SAMPLE_RATE) -> PipelineResult:
    """Convenience: load file then run full pipeline."""
    y, sr_ = audio_io.load_file(path, sr)
    import os
    return run(y, sr_, filename=os.path.basename(path))


def run_from_bytes(audio_bytes: bytes, filename: str = "upload.wav",
                   sr: int = SAMPLE_RATE) -> PipelineResult:
    """Convenience: load from bytes (Streamlit uploader) then run pipeline."""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ".wav"
    y, sr_ = audio_io.load_bytes(audio_bytes, sr, suffix=ext)
    return run(y, sr_, filename=filename)


def add_feedback(features_dict: Dict, label: str) -> Dict:
    """Store user feedback and retrain SVM if enough samples."""
    return _classifier.add_feedback(features_dict, label)


# ─── Self-test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sr = SAMPLE_RATE
    t = np.linspace(0, 2.0, sr * 2, dtype=np.float32)
    y = 0.6 * np.sin(2 * np.pi * 180 * t) + 0.2 * np.random.randn(sr * 2).astype(np.float32)
    res = run(y, sr, filename="test_sine.wav")
    print(f"Label      : {res.label}")
    print(f"Confidence : {res.confidence}%")
    print(f"F0 mean    : {res.pitch_stats.get('f0_mean', 0):.1f} Hz")
    print(f"HNR        : {res.hnr_db:.1f} dB")
    print(f"Timing     : prep={res.prep_ms}ms  feat={res.features_ms}ms  clf={res.classifier_ms}ms")
    print("pipeline.py self-test passed.")
