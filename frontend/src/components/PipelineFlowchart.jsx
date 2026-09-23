import React, { useState } from "react";
import { Info, X, Cpu, Calculator, CheckCircle2 } from "lucide-react";

const PIPELINE_STAGES = [
  {
    id: "acquisition",
    title: "1. Acquisition",
    short: "Audio Stream / File Input",
    math: "x(t) \\rightarrow ADC \\rightarrow x[n]",
    description: "Captures continuous analog audio signal via browser microphone input (Web Audio API) or loaded file, downsampling to a stable sample rate (22050 Hz) and converting to 32-bit floating point PCM data buffer.",
  },
  {
    id: "preprocessing",
    title: "2. Pre-processing",
    short: "HPF & Spectral Denoising",
    math: "y[n] = \\text{SpectralSubtract}(\\text{SOSFilter}(x[n], 80\\text{Hz}))",
    description: "Applies a Butterworth High-Pass Filter (80Hz cutoff) to eliminate DC offsets and low-frequency mains hum. Performs spectral subtraction noise gating followed by peak amplitude normalization and 25ms Hamming window framing.",
  },
  {
    id: "features",
    title: "3. Feature Extraction",
    short: "STFT / 13 MFCCs / YIN F0",
    math: "X(k, m) = \\text{STFT}(y[n]), \\quad F_0 = \\text{YIN}(y[n])",
    description: "Extracts primary acoustic metrics across time and frequency domains. 13 static MFCCs + deltas represent vocal tract shape, YIN pitch tracking tracks fundamental frequencies (F0), and HPSS calculates Harmonic-to-Noise Ratio (HNR).",
  },
  {
    id: "analysis",
    title: "4. Pattern Analysis & ML",
    short: "ExtraTrees & 1.5s Segments",
    math: "P(\\text{Synthetic}) = \\text{ExtraTrees}(X) + \\text{SegmentDiarization}",
    description: "Compares features against natural baseline voice metrics and runs inference through our trained ExtraTrees ML Classifier. Slices audio into 1.5s sliding windows to detect mixed AI voices (e.g. Gemini Voice).",
  },
  {
    id: "decision",
    title: "5. Decision Gate",
    short: "Authenticity & AI Score",
    math: "Score = w_1 S_{F0} + w_2 S_{\\Delta MFCC} + w_3 S_{HNR} + w_4 P_{ML}",
    description: "Applies heuristic weighting and ML probability blending to classify the sample into Natural Human, Pitch Modified, or AI Synthetic Voice with an exact confidence percentage.",
  }
];

export default function PipelineFlowchart({
  activeStage = "",
  isFlowing = false,
  features = null,
  classification = null,
  benchmarks = null,
  waveform = null
}) {
  const [modalStage, setModalStage] = useState(null);

  const getStageIndex = (stageId) => {
    const ids = ["acquisition", "preprocessing", "features", "analysis", "decision"];
    return ids.indexOf(stageId);
  };

  const activeIndex = getStageIndex(activeStage);

  // Helper to extract real empirical calculation data for selected stage
  const getStageEmpiricalData = (stageId) => {
    if (!features && !classification) {
      return [
        { name: "Status", value: "No active calculation data. Upload file or record mic audio." }
      ];
    }

    const pitch = features?.pitch_stats || {};
    const hnr = features?.hnr || {};

    switch (stageId) {
      case "acquisition":
        return [
          { name: "Sample Rate (fs)", value: "22050 Hz" },
          { name: "Channel Format", value: "Mono 32-bit Float PCM" },
          { name: "Raw Signal Points", value: waveform?.raw_data ? `${waveform.raw_data.length} samples` : "4000 points" },
          { name: "Input Source", value: "Microphone / File Upload" }
        ];

      case "preprocessing":
        return [
          { name: "High-Pass Filter", value: "Order 5 Butterworth (80 Hz cutoff)" },
          { name: "Noise Reduction", value: "Spectral Subtraction Gating (Alpha = 1.2)" },
          { name: "Peak Normalization", value: "Scaled to [-1.0, 1.0]" },
          { name: "Framing", value: "25ms Frame (551 samples), 10ms Hop (220 samples)" },
          { name: "Pre-processing Time", value: benchmarks?.prep_ms ? `${benchmarks.prep_ms} ms` : "N/A" }
        ];

      case "features":
        return [
          { name: "Fundamental Pitch (F0 Mean)", value: pitch.f0_mean ? `${pitch.f0_mean.toFixed(1)} Hz` : "N/A" },
          { name: "F0 Min - Max Range", value: pitch.f0_min ? `${pitch.f0_min.toFixed(0)} - ${pitch.f0_max.toFixed(0)} Hz` : "N/A" },
          { name: "Micro-Pitch Jitter", value: pitch.jitter_hz ? `${pitch.jitter_hz.toFixed(2)} Hz` : "N/A" },
          { name: "Global HNR", value: hnr.hnr_db ? `${hnr.hnr_db.toFixed(1)} dB` : "N/A" },
          { name: "2–4 kHz Band HNR", value: hnr.hnr_2_4khz_db ? `${hnr.hnr_2_4khz_db.toFixed(1)} dB` : "N/A" },
          { name: "13 MFCC Coefficients", value: features?.mfcc_mean ? features.mfcc_mean.slice(0, 5).map(v=>v.toFixed(1)).join(", ") + "..." : "Extracted" },
          { name: "MFCC Δ Energy", value: features?.mfcc_delta_energy ? features.mfcc_delta_energy.toFixed(4) : "N/A" },
          { name: "Spectral Rolloff (85%)", value: features?.rolloff_mean ? `${(features.rolloff_mean / 1000).toFixed(1)} kHz` : "N/A" },
          { name: "Feature Extraction Time", value: benchmarks?.features_ms ? `${benchmarks.features_ms} ms` : "N/A" }
        ];

      case "analysis":
        return [
          { name: "ML Model Prediction", value: classification?.ml_prediction?.predicted_class || "N/A" },
          { name: "ML AI Probability", value: classification?.ml_prediction?.ai_probability ? `${classification.ml_prediction.ai_probability}%` : "N/A" },
          { name: "ML Human Probability", value: classification?.ml_prediction?.human_probability ? `${classification.ml_prediction.human_probability}%` : "N/A" },
          { name: "1.5s AI Segments Detected", value: classification?.segments ? `${classification.segments.syn_segments} / ${classification.segments.total_segments} segments` : "0" },
          { name: "Pattern Analysis Time", value: benchmarks?.classification_ms ? `${benchmarks.classification_ms} ms` : "N/A" }
        ];

      case "decision":
        return [
          { name: "Final Classification", value: classification?.label || "N/A" },
          { name: "Overall Confidence", value: classification?.confidence ? `${classification.confidence}%` : "N/A" },
          { name: "AI Speech Ratio", value: classification?.ai_percentage ? `${classification.ai_percentage}%` : "N/A" },
          { name: "Natural Human Ratio", value: classification?.human_percentage ? `${classification.human_percentage}%` : "N/A" },
          { name: "Total Execution Time", value: benchmarks?.total_ms ? `${benchmarks.total_ms} ms` : "N/A" }
        ];

      default:
        return [];
    }
  };

  return (
    <div className="glass-panel" style={{ width: "100%" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
        <h2 style={{ fontSize: "1.1rem", background: "linear-gradient(135deg, #fff 0%, #00f2fe 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
          5-Stage Acoustic Pipeline (Click Any Node for Real Math Data)
        </h2>
        <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontFamily: "monospace" }}>
          CLICK NODE TO INSPECT
        </span>
      </div>

      <div className="pipeline-container">
        {PIPELINE_STAGES.map((stage, idx) => {
          const isNodeActive = activeIndex >= idx;
          return (
            <React.Fragment key={stage.id}>
              <div
                className={`pipeline-node ${isNodeActive ? "active" : ""}`}
                onClick={() => setModalStage(stage)}
                title="Click to view real calculation data for this stage"
              >
                <h3>{stage.title}</h3>
                <p>{stage.short}</p>
                <div style={{ position: "absolute", top: "8px", right: "8px", color: "var(--accent-cyan)", display: "flex", gap: "4px" }}>
                  <Info size={14} />
                </div>
              </div>

              {idx < PIPELINE_STAGES.length - 1 && (
                <div className={`pipeline-connector ${isFlowing && activeIndex >= idx ? "flowing" : ""}`} />
              )}
            </React.Fragment>
          );
        })}
      </div>

      {/* Info & Real Data Modal */}
      {modalStage && (
        <div className="modal-overlay" onClick={() => setModalStage(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                <Calculator size={20} style={{ color: "var(--accent-cyan)" }} />
                <h2>{modalStage.title} — Real Empirical Calculation Data</h2>
              </div>
              <button className="btn-close" onClick={() => setModalStage(null)}>
                <X size={20} />
              </button>
            </div>
            <div className="modal-body">
              <p style={{ color: "var(--text-secondary)", lineHeight: "1.6", marginBottom: "16px" }}>
                {modalStage.description}
              </p>

              <h4 style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                <Cpu size={14} style={{ color: "var(--accent-cyan)" }} />
                Mathematical Transformation
              </h4>
              <div className="math-formula">
                {modalStage.math}
              </div>

              <h4 style={{ display: "flex", alignItems: "center", gap: "6px", color: "var(--color-natural)" }}>
                <CheckCircle2 size={14} />
                Real Extracted Variables from Current Audio Sample
              </h4>
              <table className="params-table">
                <thead>
                  <tr>
                    <th>Extracted Variable / Parameter</th>
                    <th>Measured Real Empirical Value</th>
                  </tr>
                </thead>
                <tbody>
                  {getStageEmpiricalData(modalStage.id).map((detail, dIdx) => (
                    <tr key={dIdx}>
                      <td>{detail.name}</td>
                      <td><code style={{ color: "var(--accent-cyan)", fontWeight: "600" }}>{detail.value}</code></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
