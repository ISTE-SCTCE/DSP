"""
DSP + Machine Learning Voice Classification Trainer
===================================================
Generates acoustic feature samples for Natural Human Voice vs AI Synthetic Voices
(e.g., Gemini Voice, ElevenLabs, WaveNet, Tacotron2, VITS), trains a Random Forest /
ExtraTrees Classifier on extracted DSP features, and exports joblib artifacts.
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

def generate_dsp_dataset(n_samples_per_class=1000, random_seed=42):
    np.random.seed(random_seed)
    X = []
    y = []

    # Features vector structure (33 features total):
    # 0: f0_mean (Hz)
    # 1: f0_std (Hz)
    # 2: f0_min (Hz)
    # 3: f0_max (Hz)
    # 4: jitter_hz (Hz)
    # 5: voiced_ratio (0-1)
    # 6: hnr_db (dB)
    # 7: hnr_2_4khz_db (dB)
    # 8-20: mfcc_mean_0 to 12
    # 21-33: mfcc_std_0 to 12
    # 34: mfcc_delta_energy
    # 35: rolloff_mean (Hz)
    # 36: zcr_mean
    # 37: ste_mean

    # Class 0: Natural Human Voice
    for _ in range(n_samples_per_class):
        # Natural pitch variation (male: 80-160Hz, female: 165-260Hz)
        is_female = np.random.rand() > 0.5
        f0_mean = np.random.normal(210, 25) if is_female else np.random.normal(125, 20)
        f0_mean = max(70, min(350, f0_mean))
        f0_std = np.random.uniform(12.0, 45.0) # Healthy natural pitch variation
        f0_min = max(50, f0_mean - f0_std * np.random.uniform(1.5, 2.5))
        f0_max = f0_mean + f0_std * np.random.uniform(1.5, 2.5)
        jitter_hz = np.random.uniform(1.2, 8.5) # Micro-pitch fluctuations
        voiced_ratio = np.random.uniform(0.35, 0.85)

        hnr_db = np.random.uniform(12.0, 32.0)
        hnr_2_4khz_db = np.random.uniform(6.0, 24.0)

        # MFCC profile for human vocal tract
        mfcc_means = [
            np.random.normal(-250, 40), # Energy
            np.random.normal(110, 20),  # Low band formant C1
            np.random.normal(-20, 15),  # C2
            np.random.normal(15, 12),   # C3
            np.random.normal(-10, 10),
            np.random.normal(5, 8),
            np.random.normal(-5, 8),
            np.random.normal(2, 6),
            np.random.normal(-2, 6),
            np.random.normal(1, 5),
            np.random.normal(-1, 5),
            np.random.normal(0, 4),
            np.random.normal(0, 4),
        ]

        mfcc_stds = [np.random.uniform(15.0, 45.0) for _ in range(13)]
        mfcc_delta_energy = np.random.uniform(0.025, 0.095) # Natural dynamic transition

        rolloff_mean = np.random.uniform(2200, 6800)
        zcr_mean = np.random.uniform(0.03, 0.12)
        ste_mean = np.random.uniform(0.01, 0.15)

        feat = [
            f0_mean, f0_std, f0_min, f0_max, jitter_hz, voiced_ratio,
            hnr_db, hnr_2_4khz_db,
            *mfcc_means, *mfcc_stds,
            mfcc_delta_energy, rolloff_mean, zcr_mean, ste_mean
        ]
        X.append(feat)
        y.append(0) # 0 = Natural Human Voice

    # Class 1: AI-Generated Synthetic Voice (Gemini Voice, ElevenLabs, WaveNet, Tacotron2)
    for _ in range(n_samples_per_class):
        type_flag = np.random.choice(["flat_tts", "vocoder_neural", "band_limited"])

        if type_flag == "flat_tts":
            # Robotic/Flat pitch contour
            f0_mean = np.random.uniform(110, 240)
            f0_std = np.random.uniform(0.5, 5.5) # Unnatural flat F0
            jitter_hz = np.random.uniform(0.05, 0.65) # Suppressed micro-jitter
            mfcc_delta_energy = np.random.uniform(0.002, 0.014) # Smooth envelope
            hnr_db = np.random.uniform(28.0, 48.0) # Unnaturally clean
            hnr_2_4khz_db = np.random.uniform(0.5, 4.5)
            rolloff_mean = np.random.uniform(3000, 7000)

        elif type_flag == "vocoder_neural":
            # Modern neural vocoder (HiFi-GAN/WaveNet/SoundStream)
            f0_mean = np.random.uniform(120, 230)
            f0_std = np.random.uniform(4.0, 12.0)
            jitter_hz = np.random.uniform(0.3, 1.1)
            mfcc_delta_energy = np.random.uniform(0.008, 0.018)
            hnr_db = np.random.choice([np.random.uniform(-25.0, 2.0), np.random.uniform(35.0, 46.0)])
            hnr_2_4khz_db = np.random.uniform(-5.0, 3.0) # Band suppression artifact
            rolloff_mean = np.random.uniform(3500, 7800)

        else: # band_limited
            f0_mean = np.random.uniform(130, 210)
            f0_std = np.random.uniform(2.0, 8.0)
            jitter_hz = np.random.uniform(0.2, 0.9)
            mfcc_delta_energy = np.random.uniform(0.005, 0.016)
            hnr_db = np.random.uniform(18.0, 38.0)
            hnr_2_4khz_db = np.random.uniform(1.0, 5.0)
            rolloff_mean = np.random.uniform(1200, 2100) # Sharp cutoff

        f0_min = max(40, f0_mean - f0_std * np.random.uniform(1.0, 1.8))
        f0_max = f0_mean + f0_std * np.random.uniform(1.0, 1.8)
        voiced_ratio = np.random.uniform(0.40, 0.90)

        mfcc_means = [
            np.random.normal(-180, 25),
            np.random.normal(90, 15),
            np.random.normal(-10, 10),
            np.random.normal(8, 8),
            np.random.normal(-4, 6),
            np.random.normal(2, 5),
            np.random.normal(-2, 5),
            np.random.normal(1, 4),
            np.random.normal(-1, 4),
            np.random.normal(0, 3),
            np.random.normal(0, 3),
            np.random.normal(0, 2),
            np.random.normal(0, 2),
        ]
        mfcc_stds = [np.random.uniform(4.0, 16.0) for _ in range(13)] # Smooth variance
        zcr_mean = np.random.uniform(0.01, 0.08)
        ste_mean = np.random.uniform(0.02, 0.18)

        feat = [
            f0_mean, f0_std, f0_min, f0_max, jitter_hz, voiced_ratio,
            hnr_db, hnr_2_4khz_db,
            *mfcc_means, *mfcc_stds,
            mfcc_delta_energy, rolloff_mean, zcr_mean, ste_mean
        ]
        X.append(feat)
        y.append(1) # 1 = AI Synthetic Voice

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

def train_model():
    print("Generating acoustic DSP dataset (2,000 synthetic & natural audio samples)...")
    X, y = generate_dsp_dataset(n_samples_per_class=1000)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    print("Fitting Standard Scaler and ExtraTrees Classifier...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = ExtraTreesClassifier(n_estimators=150, max_depth=12, random_state=42)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Training Complete! Accuracy on Test Set: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=["Natural Human Voice", "AI Synthetic Voice"]))

    # Save model artifacts
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    os.makedirs(model_dir, exist_ok=True)

    clf_path = os.path.join(model_dir, "voice_classifier.joblib")
    scaler_path = os.path.join(model_dir, "scaler.joblib")

    joblib.dump(clf, clf_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved trained ML model to: {clf_path}")
    print(f"Saved feature scaler to: {scaler_path}")

if __name__ == "__main__":
    train_model()
