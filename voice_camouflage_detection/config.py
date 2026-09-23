"""
config.py — Central configuration constants
============================================
KTU S5 DSP Project: Voice Camouflage Detection
All tunable parameters live here. Never hard-code magic numbers in other modules.
"""

# ─── Audio ──────────────────────────────────────────────────────────────────
SAMPLE_RATE: int = 22050          # Default resampling target (Hz)
MAX_DURATION_S: float = 120.0     # Clip recordings longer than this (seconds)
MIN_DURATION_S: float = 0.3       # Reject clips shorter than this (seconds)

# ─── Framing / Windowing ────────────────────────────────────────────────────
FRAME_DURATION_MS: float = 25.0   # Frame length in milliseconds
HOP_DURATION_MS: float = 10.0     # Hop size (60 % overlap)
WINDOW_TYPE: str = "hamming"      # Hamming window — good side-lobe suppression

# ─── Pre-processing / Filtering ─────────────────────────────────────────────
HPF_CUTOFF_HZ: float = 80.0       # High-pass filter cutoff (removes DC + mains hum)
HPF_ORDER: int = 5                # Butterworth filter order
NOISE_ESTIMATION_PERCENTILE: float = 0.15   # Fraction of quietest frames for noise profile
SPECTRAL_SUBTRACTION_ALPHA: float = 1.2     # Over-subtraction factor
SPECTRAL_NOISE_FLOOR: float = 0.10          # Minimum fraction of original magnitude kept

# ─── Feature Extraction ─────────────────────────────────────────────────────
N_MFCC: int = 13                  # Number of MFCC coefficients (standard for speech)
N_FFT: int = None                 # None → auto-set from frame length in FeatureExtractor
N_MEL_FILTERS: int = 40          # Mel filterbank bands used for MFCC computation
FMIN_PITCH: float = 50.0          # Min F0 search frequency (Hz)
FMAX_PITCH: float = 500.0         # Max F0 search frequency (Hz)
SPECTRAL_ROLLOFF_PERCENT: float = 0.85  # Energy percentile for rolloff freq

# ─── Classification Thresholds (natural-speech baselines) ───────────────────
NATURAL_F0_STD_MIN: float = 6.0         # σ(F0) below this → suspiciously flat pitch
NATURAL_JITTER_MIN: float = 0.8         # Jitter (Hz) below this → robotic
NATURAL_DELTA_ENERGY_MIN: float = 0.015 # MFCC Δ energy below this → smooth vocoder
HNR_SYNTHETIC_HIGH: float = 40.0        # HNR above this → unnaturally clean
HNR_NOISE_LOW: float = 6.0              # HNR below this → too noisy
HNR_BAND_MIN: float = 3.0               # 2–4 kHz band HNR below this → vocoder band gap
ROLLOFF_NARROW_HZ: float = 1800.0       # Rolloff below this → narrow bandwidth

# Segment diarization (temporal analysis)
SEGMENT_LEN_S: float = 1.5       # Each temporal segment length
SEGMENT_STEP_S: float = 0.5      # Temporal step between segments
SEG_F0_STD_FLAT: float = 8.5     # Per-segment F0 std threshold for flatness
SEG_JITTER_FLAT: float = 1.2     # Per-segment jitter threshold
SEG_DELTA_SMOOTH: float = 0.018  # Per-segment delta energy smoothness threshold
SEG_SYN_RATIO_LOW: float = 0.15  # Warning threshold (% synthetic segments)
SEG_SYN_RATIO_HIGH: float = 0.35 # High-confidence AI threshold

# ─── SVM Classifier ─────────────────────────────────────────────────────────
SVM_MODEL_PATH: str = "model/svm_classifier.joblib"
SVM_SCALER_PATH: str = "model/svm_scaler.joblib"
EXTRATREES_MODEL_PATH: str = "model/voice_classifier.joblib"
EXTRATREES_SCALER_PATH: str = "model/scaler.joblib"
FEEDBACK_DATA_PATH: str = "model/feedback_data.json"

# ─── Z-transform / Filter visualisation ─────────────────────────────────────
FREQZ_N_POINTS: int = 512        # Number of frequency points for H(e^jω) plot

# ─── Visualisation ──────────────────────────────────────────────────────────
WAVEFORM_MAX_POINTS: int = 4000   # Downsample waveform to this for rendering
SPEC_MAX_TIME: int = 200          # Spectrogram time bins after downsampling
SPEC_MAX_FREQ: int = 128          # Spectrogram freq bins after downsampling
MFCC_MAX_TIME: int = 100          # MFCC matrix time bins after downsampling
