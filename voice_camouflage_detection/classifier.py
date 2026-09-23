"""
classifier.py — Threshold rules + SVM voice classifier
=======================================================
KTU S5 DSP Project: Voice Camouflage Detection

Three classifier strategies:
    1. ThresholdClassifier  — rule-based DSP baseline (always available)
    2. SVMClassifier        — scikit-learn SVC trained on feature vectors
    3. HybridClassifier     — combines all available classifiers

DSP Concept — KTU Unit 5 (Applications):
    Classification of audio into Natural / AI-Synthetic / Modified is
    an application of DSP feature analysis.  The threshold rules encode
    expert knowledge about how AI vocoders differ from human speech at the
    signal level (flat pitch, smooth spectra, unusual HNR).

    The SVM acts as a data-driven fallback that learns a decision boundary
    in the 38-dimensional feature space spanned by MFCC means/stds,
    pitch statistics, HNR, ZCR, STE, and spectral rolloff.
"""

import os
import json
import numpy as np
from typing import Dict, List, Optional

from config import (
    NATURAL_F0_STD_MIN, NATURAL_JITTER_MIN, NATURAL_DELTA_ENERGY_MIN,
    HNR_SYNTHETIC_HIGH, HNR_NOISE_LOW, HNR_BAND_MIN,
    ROLLOFF_NARROW_HZ,
    SEG_F0_STD_FLAT, SEG_JITTER_FLAT, SEG_DELTA_SMOOTH,
    SEG_SYN_RATIO_LOW, SEG_SYN_RATIO_HIGH,
    SEGMENT_LEN_S, SEGMENT_STEP_S,
    SVM_MODEL_PATH, SVM_SCALER_PATH,
    SAMPLE_RATE,
)
from utils import clamp


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Vector Builder  (shared between SVM and ExtraTrees)
# ═══════════════════════════════════════════════════════════════════════════════

def build_feature_vector(features: Dict) -> List[float]:
    """Build a 38-dimensional feature vector from the extracted feature dict.

    Dimensions:
        0–5   : Pitch statistics (mean, std, min, max, jitter, voiced_ratio)
        6–7   : HNR global and 2-4kHz band
        8–20  : MFCC means (13 coefficients)
        21–33 : MFCC stds  (13 coefficients)
        34    : MFCC delta energy
        35    : Spectral rolloff mean
        36    : ZCR mean
        37    : STE mean
    """
    ps = features["pitch"]["stats"]
    return [
        ps.get("f0_mean", 0.0),
        ps.get("f0_std", 0.0),
        ps.get("f0_min", 0.0),
        ps.get("f0_max", 0.0),
        ps.get("jitter_hz", 0.0),
        ps.get("voiced_ratio", 0.0),
        features["hnr"]["hnr_db"],
        features["hnr"]["hnr_2_4khz_db"],
        *features["mfcc"]["mfcc_means"],   # 13 values
        *features["mfcc"]["mfcc_stds"],    # 13 values
        features["mfcc"]["delta_energy"],
        features["rolloff"]["rolloff_mean"],
        features["zcr"]["zcr_mean"],
        features["ste"]["ste_mean"],
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ThresholdClassifier — Rule-based DSP baseline
# ═══════════════════════════════════════════════════════════════════════════════

class ThresholdClassifier:
    """Classify voice using acoustic baseline thresholds.

    No ML required — works on the raw feature values from features.py.
    This is the reference baseline that an engineer would derive from
    published literature on AI speech synthesis artifacts.
    """

    def analyze_segments(self, y: np.ndarray, sr: int,
                         import_features_fn) -> Dict:
        """Temporal diarization: detect AI speech within 1.5s segments."""
        seg_len = int(sr * SEGMENT_LEN_S)
        step_len = int(sr * SEGMENT_STEP_S)
        n = len(y)

        if n < seg_len:
            return {"syn_segments": 0, "human_segments": 0,
                    "total_segments": 0, "segment_timeline": []}

        syn_count = 0
        human_count = 0
        timeline = []

        for i in range(0, n - seg_len + 1, step_len):
            chunk = y[i : i + seg_len]
            energy = float(np.mean(chunk ** 2))
            if energy < 1e-4:
                continue

            t_start = round(i / sr, 1)
            t_end = round((i + seg_len) / sr, 1)

            seg_feats = import_features_fn(chunk, sr)
            f0_std = seg_feats["pitch"]["stats"].get("f0_std", 0.0)
            jitter = seg_feats["pitch"]["stats"].get("jitter_hz", 0.0)
            delta_e = seg_feats["mfcc"]["delta_energy"]
            hnr_band = seg_feats["hnr"]["hnr_2_4khz_db"]
            voiced_ratio = seg_feats["pitch"]["stats"].get("voiced_ratio", 0.0)

            is_syn = False
            reasons: List[str] = []

            if voiced_ratio > 0.15:
                if f0_std < SEG_F0_STD_FLAT and jitter < SEG_JITTER_FLAT:
                    is_syn = True
                    reasons.append("Flat F0 & suppressed micro-jitter")
                if delta_e < SEG_DELTA_SMOOTH:
                    is_syn = True
                    reasons.append("Smooth neural vocoder MFCC envelope")

            if hnr_band < 2.0:
                is_syn = True
                reasons.append("Vocoder band harmonic suppression")

            if is_syn:
                syn_count += 1
                timeline.append({"time": f"{t_start}s–{t_end}s",
                                  "type": "AI Synthetic Voice",
                                  "reasons": reasons})
            else:
                human_count += 1
                timeline.append({"time": f"{t_start}s–{t_end}s",
                                  "type": "Natural Human Voice",
                                  "reasons": ["Natural pitch fluctuation & micro-jitter"]})

        return {
            "syn_segments": syn_count,
            "human_segments": human_count,
            "total_segments": syn_count + human_count,
            "segment_timeline": timeline,
        }

    def predict(self, features: Dict,
                y: Optional[np.ndarray] = None,
                sr: int = SAMPLE_RATE) -> Dict:
        """Apply threshold rules to classify the voice sample.

        Args:
            features: Output of features.extract_all().
            y:        Raw clean signal (needed for segment analysis).
            sr:       Sample rate.

        Returns:
            dict with label, confidence, ai_percentage, human_percentage,
            anomalies, scores, segments.
        """
        anomalies: List[str] = []
        scores: Dict[str, float] = {}

        ps = features["pitch"]["stats"]
        f0_mean = ps.get("f0_mean", 0.0)
        f0_std = ps.get("f0_std", 0.0)
        jitter = ps.get("jitter_hz", 0.0)
        voiced_ratio = ps.get("voiced_ratio", 0.0)

        # 1. Pitch contour analysis
        if voiced_ratio > 0.10 and f0_mean > 50:
            if f0_std < NATURAL_F0_STD_MIN and jitter < NATURAL_JITTER_MIN:
                scores["f0_variance"] = 0.90
                anomalies.append(
                    f"Anomaly: Flat pitch (F0 σ={f0_std:.1f} Hz, jitter={jitter:.2f} Hz)"
                    f" — robotic synthetic voice marker"
                )
            elif f0_std > 85.0:
                scores["f0_variance"] = 0.40
                anomalies.append(
                    f"Warning: High F0 variation ({f0_std:.1f} Hz) — pitch-shift artifact"
                )
            if jitter > 18.0:
                scores["jitter"] = 0.50
                anomalies.append(f"Warning: Excessive jitter ({jitter:.2f} Hz)")

        # 2. HNR analysis
        hnr = features["hnr"]["hnr_db"]
        hnr_band = features["hnr"]["hnr_2_4khz_db"]

        if hnr < 0.0:
            scores["hnr"] = 0.70
            anomalies.append(f"Anomaly: Negative HNR ({hnr:.1f} dB) — phase cancellation / multi-speaker")
        elif hnr < HNR_NOISE_LOW:
            scores["hnr"] = 0.40
            anomalies.append(f"Warning: Low HNR ({hnr:.1f} dB)")
        elif hnr > HNR_SYNTHETIC_HIGH:
            scores["hnr"] = 0.80
            anomalies.append(f"Anomaly: Unnaturally high HNR ({hnr:.1f} dB) — synthetic signal")

        if hnr_band < HNR_BAND_MIN:
            scores["hnr_band"] = 0.60
            anomalies.append(f"Anomaly: 2–4 kHz band HNR suppression ({hnr_band:.1f} dB)")

        # 3. MFCC spectral smoothness
        delta_energy = features["mfcc"]["delta_energy"]
        if delta_energy < NATURAL_DELTA_ENERGY_MIN:
            scores["spectral_smoothness"] = 0.85
            anomalies.append(
                f"Anomaly: Smooth spectral envelope (MFCC Δ energy={delta_energy:.4f})"
                f" — neural vocoder artifact"
            )

        # 4. Spectral rolloff
        rolloff = features["rolloff"]["rolloff_mean"]
        if rolloff < ROLLOFF_NARROW_HZ:
            scores["rolloff"] = 0.40
            anomalies.append(f"Note: Narrow bandwidth (rolloff={rolloff:.0f} Hz)")

        # 5. Temporal segment diarization
        segments = {"syn_segments": 0, "human_segments": 0,
                    "total_segments": 0, "segment_timeline": []}
        if y is not None:
            import features as feat_mod
            def _quick_extract(chunk, sr_):
                import preprocessing as prep
                yc, frm = prep.run(chunk, sr_)
                return feat_mod.extract_all(yc, frm, sr_)
            segments = self.analyze_segments(y, sr, _quick_extract)

        syn_segs = segments["syn_segments"]
        tot_segs = max(1, segments["total_segments"])
        syn_ratio = syn_segs / tot_segs

        if syn_ratio > SEG_SYN_RATIO_LOW:
            anomalies.append(
                f"Anomaly: AI voice in {syn_segs}/{tot_segs} segments "
                f"({int(syn_ratio * 100)}% of recording)"
            )

        # Composite score
        syn_score = (
            scores.get("f0_variance", 0) * 0.25
            + scores.get("spectral_smoothness", 0) * 0.25
            + scores.get("hnr", 0) * 0.20
            + scores.get("hnr_band", 0) * 0.15
            + syn_ratio * 0.40
        )

        # Decision
        if syn_ratio > SEG_SYN_RATIO_HIGH or syn_score > 0.50:
            if segments["human_segments"] > 0:
                label = "AI Voice Detected (Mixed Audio)"
            else:
                label = "AI-Generated Synthetic Voice"
            confidence = clamp(70.0 + syn_score * 35.0, 85.0, 98.0)
        elif syn_ratio > SEG_SYN_RATIO_LOW or syn_score > 0.28:
            label = "AI Voice Detected in Audio Stream"
            confidence = clamp(55.0 + syn_score * 40.0, 75.0, 92.0)
        elif scores.get("jitter", 0) > 0.4 or scores.get("rolloff", 0) > 0.3:
            label = "Voice Modified / Pitch Shifted"
            confidence = clamp(60.0 + (scores.get("jitter", 0) + scores.get("rolloff", 0)) * 25.0,
                               70.0, 90.0)
        else:
            label = "Natural Human Voice"
            confidence = clamp(99.0 - syn_score * 40.0, 80.0, 96.0)

        if not anomalies:
            anomalies.append("No acoustic anomalies. Signal matches natural speech profile.")

        if segments["total_segments"] > 0:
            ai_pct = round((segments["syn_segments"] / segments["total_segments"]) * 100.0, 1)
        else:
            ai_pct = 0.0

        return {
            "label": label,
            "confidence": round(confidence, 1),
            "ai_percentage": ai_pct,
            "human_percentage": round(max(0.0, 100.0 - ai_pct), 1),
            "anomalies": anomalies,
            "scores": scores,
            "segments": segments,
            "ml_prediction": {"available": False},
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SVMClassifier — scikit-learn SVC
# ═══════════════════════════════════════════════════════════════════════════════

class SVMClassifier:
    """SVM-based voice classifier.

    Uses a radial-basis-function (RBF) SVC from scikit-learn trained on
    38-dimensional feature vectors extracted by build_feature_vector().

    Acts as a data-driven complement to the threshold rules:
    if feedback data is available, the SVM learns a soft boundary from it.
    Falls back to returning available=False if no model file exists.
    """

    def __init__(self, model_path: str = SVM_MODEL_PATH,
                 scaler_path: str = SVM_SCALER_PATH):
        self.model = None
        self.scaler = None
        self.is_loaded = False
        self._model_path = model_path
        self._scaler_path = scaler_path
        self._try_load()

    def _try_load(self):
        """Attempt to load a pre-trained model from disk."""
        try:
            import joblib
            base = os.path.dirname(os.path.abspath(__file__))
            mp = os.path.join(base, self._model_path)
            sp = os.path.join(base, self._scaler_path)
            if os.path.exists(mp) and os.path.exists(sp):
                self.model = joblib.load(mp)
                self.scaler = joblib.load(sp)
                self.is_loaded = True
        except Exception as e:
            print(f"[classifier] SVM load error: {e}")

    def train(self, feature_dicts: List[Dict], labels: List[int]) -> Dict:
        """Train (or retrain) the SVM on the provided samples.

        Args:
            feature_dicts: List of feature dicts from features.extract_all().
            labels:        List of int labels (0=Human, 1=AI).

        Returns:
            dict with success, n_samples, accuracy (on training data).
        """
        try:
            from sklearn.svm import SVC
            from sklearn.preprocessing import StandardScaler
            import joblib

            X = np.array([build_feature_vector(f) for f in feature_dicts], dtype=np.float32)
            y = np.array(labels, dtype=int)

            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            self.model = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True)
            self.model.fit(X_scaled, y)
            self.is_loaded = True

            # Persist
            base = os.path.dirname(os.path.abspath(__file__))
            os.makedirs(os.path.join(base, "model"), exist_ok=True)
            joblib.dump(self.model, os.path.join(base, self._model_path))
            joblib.dump(self.scaler, os.path.join(base, self._scaler_path))

            train_acc = self.model.score(X_scaled, y)
            return {"success": True, "n_samples": len(y),
                    "accuracy": round(train_acc * 100.0, 1)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def predict(self, features: Dict) -> Dict:
        """Predict class probabilities for a single feature dict."""
        if not self.is_loaded:
            return {"available": False, "ai_probability": 0.0,
                    "human_probability": 0.0}
        try:
            vec = np.array([build_feature_vector(features)], dtype=np.float32)
            vec_scaled = self.scaler.transform(vec)
            probs = self.model.predict_proba(vec_scaled)[0]
            p_human, p_ai = float(probs[0]), float(probs[1])
            return {
                "available": True,
                "ai_probability": round(p_ai * 100.0, 1),
                "human_probability": round(p_human * 100.0, 1),
                "predicted_class": "AI Synthetic Voice" if p_ai > 0.5 else "Natural Human Voice",
                "model_name": "SVM (RBF kernel)",
            }
        except Exception as e:
            print(f"[classifier] SVM predict error: {e}")
            return {"available": False, "ai_probability": 0.0,
                    "human_probability": 0.0}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HybridClassifier — DSP Rules + SVM + Optional ExtraTrees
# ═══════════════════════════════════════════════════════════════════════════════

class HybridClassifier:
    """Combine threshold rules, SVM, and ExtraTrees (if available).

    Decision fusion weights:
        DSP rules    — 40%  (always present, robust to distribution shift)
        SVM          — 30%  (if trained on feedback data)
        ExtraTrees   — 30%  (if pre-trained model file exists)
    """

    def __init__(self):
        self.threshold = ThresholdClassifier()
        self.svm = SVMClassifier()
        self._et_model = None
        self._et_scaler = None
        self._try_load_extratrees()

    def _try_load_extratrees(self):
        from config import EXTRATREES_MODEL_PATH, EXTRATREES_SCALER_PATH
        try:
            import joblib
            base = os.path.dirname(os.path.abspath(__file__))
            mp = os.path.join(base, EXTRATREES_MODEL_PATH)
            sp = os.path.join(base, EXTRATREES_SCALER_PATH)
            if os.path.exists(mp) and os.path.exists(sp):
                self._et_model = joblib.load(mp)
                self._et_scaler = joblib.load(sp)
        except Exception:
            pass

    def _extratrees_predict(self, features: Dict) -> Dict:
        if self._et_model is None:
            return {"available": False}
        try:
            vec = np.array([build_feature_vector(features)], dtype=np.float32)
            vec_s = self._et_scaler.transform(vec)
            probs = self._et_model.predict_proba(vec_s)[0]
            return {
                "available": True,
                "ai_probability": round(float(probs[1]) * 100.0, 1),
                "human_probability": round(float(probs[0]) * 100.0, 1),
                "predicted_class": "AI Synthetic Voice" if probs[1] > 0.5 else "Natural Human Voice",
                "model_name": "ExtraTrees Acoustic Classifier v1.0",
            }
        except Exception:
            return {"available": False}

    def predict(self, features: Dict,
                y: Optional[np.ndarray] = None,
                sr: int = SAMPLE_RATE) -> Dict:
        """Fuse DSP + SVM + ExtraTrees predictions into final classification."""
        dsp_result = self.threshold.predict(features, y, sr)
        svm_result = self.svm.predict(features)
        et_result = self._extratrees_predict(features)

        # Collect ML AI probabilities (normalised to [0,1])
        ml_ai_prob = 0.0
        ml_weight = 0.0
        if svm_result.get("available"):
            ml_ai_prob += (svm_result["ai_probability"] / 100.0) * 0.30
            ml_weight += 0.30
        if et_result.get("available"):
            ml_ai_prob += (et_result["ai_probability"] / 100.0) * 0.30
            ml_weight += 0.30

        # DSP score (0→1) from anomaly analysis
        scores = dsp_result["scores"]
        syn_ratio = (dsp_result["segments"]["syn_segments"] /
                     max(1, dsp_result["segments"]["total_segments"]))
        dsp_score = (
            scores.get("f0_variance", 0) * 0.25
            + scores.get("spectral_smoothness", 0) * 0.25
            + scores.get("hnr", 0) * 0.20
            + scores.get("hnr_band", 0) * 0.15
            + syn_ratio * 0.40
        )

        # Fuse
        total_ai_prob = dsp_score * 0.40 + ml_ai_prob

        # Update label if ML overrides
        if total_ai_prob > 0.55:
            if dsp_result["segments"]["human_segments"] > 0:
                label = "AI Voice Detected (Mixed Audio)"
            else:
                label = "AI-Generated Synthetic Voice"
            confidence = clamp(70.0 + total_ai_prob * 30.0, 85.0, 98.0)
        elif total_ai_prob > 0.35:
            label = "AI Voice Detected in Audio Stream"
            confidence = clamp(55.0 + total_ai_prob * 40.0, 75.0, 92.0)
        elif dsp_result["label"].startswith("Voice Modified"):
            label = dsp_result["label"]
            confidence = dsp_result["confidence"]
        else:
            label = dsp_result["label"]
            confidence = dsp_result["confidence"]

        # Attach ML prediction banners
        ml_prediction = svm_result if svm_result.get("available") else et_result

        return {
            **dsp_result,
            "label": label,
            "confidence": round(confidence, 1),
            "ml_prediction": ml_prediction,
            "ai_percentage": round(max(0.0, min(100.0, total_ai_prob * 100.0)), 1),
            "human_percentage": round(max(0.0, 100.0 - total_ai_prob * 100.0), 1),
        }

    def add_feedback(self, features: Dict, label: str) -> Dict:
        """Store a labelled sample and optionally retrain the SVM."""
        from config import FEEDBACK_DATA_PATH
        base = os.path.dirname(os.path.abspath(__file__))
        fpath = os.path.join(base, FEEDBACK_DATA_PATH)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)

        label_code = 1 if label.lower() in ["ai", "synthetic", "mixed"] else 0
        vec = build_feature_vector(features)

        data = []
        if os.path.exists(fpath):
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
            except Exception:
                data = []

        data.append({"vector": vec, "label": label_code, "ground_truth": label})
        with open(fpath, "w") as f:
            json.dump(data, f, indent=2)

        # Retrain SVM if enough samples
        if len(data) >= 4:
            all_X = [d["vector"] for d in data]
            all_y = [d["label"] for d in data]
            # Reconstruct dummy feature dicts from stored vectors (bypass re-extraction)
            # We pass raw arrays directly since SVMClassifier.train expects feature_dicts
            # Use a direct sklearn path here:
            try:
                from sklearn.svm import SVC
                from sklearn.preprocessing import StandardScaler
                import joblib

                X = np.array(all_X, dtype=np.float32)
                y_arr = np.array(all_y, dtype=int)
                self.svm.scaler = StandardScaler()
                X_s = self.svm.scaler.fit_transform(X)
                self.svm.model = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True)
                self.svm.model.fit(X_s, y_arr)
                self.svm.is_loaded = True

                mp = os.path.join(base, SVM_MODEL_PATH)
                sp = os.path.join(base, SVM_SCALER_PATH)
                os.makedirs(os.path.dirname(mp), exist_ok=True)
                joblib.dump(self.svm.model, mp)
                joblib.dump(self.svm.scaler, sp)

                acc = self.svm.model.score(X_s, y_arr)
                return {"success": True, "n_samples": len(data),
                        "model_status": "SVM Retrained", "accuracy": round(acc * 100.0, 1)}
            except Exception as e:
                return {"success": False, "error": str(e), "n_samples": len(data)}

        return {"success": True, "n_samples": len(data),
                "model_status": "Sample stored (need ≥4 to retrain)"}
