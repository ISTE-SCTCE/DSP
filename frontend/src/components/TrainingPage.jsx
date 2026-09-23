import React, { useState, useEffect } from "react";
import { Brain, Upload, CheckCircle2, AlertCircle, RefreshCw, Cpu, Layers, BarChart3, Sparkles } from "lucide-react";

export default function TrainingPage() {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState("");
  const [groundTruth, setGroundTruth] = useState("ai"); // human | ai | mixed
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [statusMsg, setStatusMsg] = useState(null);
  const [error, setError] = useState(null);

  const [modelStats, setModelStats] = useState(null);
  const [loadingStats, setLoadingStats] = useState(true);

  // Fetch model stats on mount
  const fetchModelStats = async () => {
    setLoadingStats(true);
    try {
      const res = await fetch("http://localhost:8000/api/model-stats");
      if (res.ok) {
        const data = await res.json();
        setModelStats(data);
      }
    } catch (err) {
      console.error("Failed to load model stats", err);
    } finally {
      setLoadingStats(false);
    }
  };

  useEffect(() => {
    fetchModelStats();
  }, []);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      setFile(selected);
      setFileName(selected.name);
      setError(null);
    }
  };

  const handleTrainSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError("Please select an audio file to provide training feedback.");
      return;
    }

    setIsSubmitting(true);
    setError(null);
    setStatusMsg("Submitting a feedback sample for the reviewed training workflow...");

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("ground_truth", groundTruth);

      const response = await fetch("http://localhost:8000/api/train", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errDetail = await response.json();
        throw new Error(errDetail.detail || "Server error during model retraining");
      }

      const result = await response.json();
      setStatusMsg(result.message || "Feedback sample recorded for review.");
      setModelStats(result.model_stats);
      setFile(null);
      setFileName("");
    } catch (err) {
      console.error(err);
      setError(err.message || "Failed to retrain model.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div style={{ maxWidth: "1200px", margin: "0 auto", padding: "24px", display: "flex", flexDirection: "column", gap: "24px" }}>
      {/* Banner */}
      <div className="glass-panel" style={{ background: "linear-gradient(135deg, rgba(79, 172, 254, 0.1) 0%, rgba(0, 242, 254, 0.05) 100%)", border: "1px solid var(--accent-cyan)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          <div style={{ padding: "14px", borderRadius: "12px", background: "rgba(0, 242, 254, 0.15)", border: "1px solid var(--accent-cyan)" }}>
            <Brain size={32} style={{ color: "var(--accent-cyan)" }} />
          </div>
          <div>
            <h2 style={{ fontSize: "1.3rem", fontWeight: "700" }}>ML Model Training & Active Feedback Correction</h2>
            <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginTop: "4px" }}>
              Submit labelled samples for review. A model is published only after real-audio validation on a held-out test set.
            </p>
          </div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))", gap: "24px" }}>
        {/* Left Column: Sample Marking & Training Form */}
        <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
          <h3 style={{ fontSize: "1.05rem", fontWeight: "600", display: "flex", alignItems: "center", gap: "8px" }}>
            <Sparkles size={18} style={{ color: "var(--color-natural)" }} />
            Mark & Submit Training Sample
          </h3>

          <form onSubmit={handleTrainSubmit} style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
            {/* File Upload Box */}
            <div style={{ border: "2px dashed var(--border-color)", borderRadius: "12px", padding: "24px", textAlign: "center", cursor: "pointer", background: "rgba(255, 255, 255, 0.01)" }}>
              <input type="file" accept=".wav,.mp3,.flac,.m4a" onChange={handleFileChange} style={{ display: "none" }} id="train-file-input" />
              <label htmlFor="train-file-input" style={{ cursor: "pointer", display: "flex", flexDirection: "column", alignItems: "center", gap: "10px" }}>
                <Upload size={32} style={{ color: "var(--accent-cyan)" }} />
                <span style={{ fontSize: "0.9rem", fontWeight: "600" }}>
                  {fileName ? fileName : "Click to select sample audio file"}
                </span>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
                  Supports .wav, .mp3, .flac, .m4a audio recordings
                </span>
              </label>
            </div>

            {/* Ground Truth Label Selector */}
            <div>
              <label style={{ fontSize: "0.85rem", fontWeight: "600", color: "var(--text-secondary)", marginBottom: "10px", display: "block" }}>
                Confirm Ground-Truth Audio Label:
              </label>
              <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                <label style={{ display: "flex", alignItems: "center", gap: "10px", padding: "10px 14px", borderRadius: "8px", background: groundTruth === "ai" ? "rgba(255, 23, 68, 0.1)" : "rgba(255, 255, 255, 0.02)", border: groundTruth === "ai" ? "1px solid var(--color-synthetic)" : "1px solid var(--border-color)", cursor: "pointer" }}>
                  <input type="radio" name="ground_truth" value="ai" checked={groundTruth === "ai"} onChange={() => setGroundTruth("ai")} />
                  <div>
                    <strong style={{ color: "var(--color-synthetic)", fontSize: "0.85rem" }}>🤖 AI-Generated Synthetic Voice</strong>
                    <p style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Gemini Voice, ElevenLabs, WaveNet, TTS synthesis</p>
                  </div>
                </label>

                <label style={{ display: "flex", alignItems: "center", gap: "10px", padding: "10px 14px", borderRadius: "8px", background: groundTruth === "human" ? "rgba(0, 230, 118, 0.1)" : "rgba(255, 255, 255, 0.02)", border: groundTruth === "human" ? "1px solid var(--color-natural)" : "1px solid var(--border-color)", cursor: "pointer" }}>
                  <input type="radio" name="ground_truth" value="human" checked={groundTruth === "human"} onChange={() => setGroundTruth("human")} />
                  <div>
                    <strong style={{ color: "var(--color-natural)", fontSize: "0.85rem" }}>🗣️ Natural Human Voice</strong>
                    <p style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Real un-synthesized human speech recording</p>
                  </div>
                </label>

                <label style={{ display: "flex", alignItems: "center", gap: "10px", padding: "10px 14px", borderRadius: "8px", background: groundTruth === "mixed" ? "rgba(213, 0, 249, 0.1)" : "rgba(255, 255, 255, 0.02)", border: groundTruth === "mixed" ? "1px solid var(--color-modified)" : "1px solid var(--border-color)", cursor: "pointer" }}>
                  <input type="radio" name="ground_truth" value="mixed" checked={groundTruth === "mixed"} onChange={() => setGroundTruth("mixed")} />
                  <div>
                    <strong style={{ color: "var(--color-modified)", fontSize: "0.85rem" }}>🔀 Mixed Audio (AI + Human Speech)</strong>
                    <p style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Audio recording containing both human & AI voices</p>
                  </div>
                </label>
              </div>
            </div>

            {/* Retrain Button */}
            <button
              type="submit"
              disabled={isSubmitting || !file}
              className="btn-mode active"
              style={{ padding: "12px", justifyContent: "center", fontSize: "0.95rem", opacity: (!file || isSubmitting) ? 0.6 : 1 }}
            >
              {isSubmitting ? <RefreshCw size={18} className="spin-loader" style={{ animation: "spin 1s linear infinite" }} /> : <Cpu size={18} />}
              {isSubmitting ? "Submitting feedback..." : "Submit feedback for review"}
            </button>
          </form>

          {statusMsg && (
            <div style={{ display: "flex", alignItems: "center", gap: "8px", background: "rgba(0, 230, 118, 0.1)", border: "1px solid rgba(0, 230, 118, 0.3)", padding: "12px", borderRadius: "8px", color: "var(--color-natural)", fontSize: "0.85rem" }}>
              <CheckCircle2 size={18} />
              <span>{statusMsg}</span>
            </div>
          )}

          {error && (
            <div style={{ display: "flex", alignItems: "center", gap: "8px", background: "rgba(239, 68, 68, 0.1)", border: "1px solid rgba(239, 68, 68, 0.3)", padding: "12px", borderRadius: "8px", color: "#fda4af", fontSize: "0.85rem" }}>
              <AlertCircle size={18} />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Right Column: Live Model Performance & Feature Importances */}
        <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <h3 style={{ fontSize: "1.05rem", fontWeight: "600", display: "flex", alignItems: "center", gap: "8px" }}>
              <BarChart3 size={18} style={{ color: "var(--accent-cyan)" }} />
              Live ML Model Statistics & Weights
            </h3>
            <button onClick={fetchModelStats} className="btn-mode" style={{ padding: "4px 8px", fontSize: "0.75rem" }}>
              <RefreshCw size={12} />
              Refresh
            </button>
          </div>

          {loadingStats ? (
            <p style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>Loading ML model metrics...</p>
          ) : modelStats && modelStats.loaded ? (
            <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "12px" }}>
                <div className="feature-badge">
                  <div className="label">Model Architecture</div>
                  <div className="value" style={{ color: "var(--accent-cyan)" }}>{modelStats.model_name}</div>
                </div>
                <div className="feature-badge">
                  <div className="label">Validation Accuracy</div>
                  <div className="value" style={{ color: "var(--text-muted)" }}>Not evaluated</div>
                </div>
                <div className="feature-badge">
                  <div className="label">Total Estimators</div>
                  <div className="value" style={{ color: "var(--accent-blue)" }}>{modelStats.total_estimators} Trees</div>
                </div>
                <div className="feature-badge">
                  <div className="label">Input DSP Dimensions</div>
                  <div className="value" style={{ color: "var(--color-modified)" }}>{modelStats.n_features} Features</div>
                </div>
              </div>

              {/* Top Feature Importances */}
              <div>
                <h4 style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: "12px", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                  Top Acoustic Feature Importances:
                </h4>
                <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                  {modelStats.top_features?.map((item, idx) => (
                    <div key={idx} style={{ background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-color)", padding: "10px 14px", borderRadius: "8px" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", fontWeight: "600", marginBottom: "6px" }}>
                        <span>{item.feature}</span>
                        <span style={{ color: "var(--accent-cyan)" }}>{item.importance}% Weight</span>
                      </div>
                      <div style={{ width: "100%", height: "4px", background: "rgba(255,255,255,0.06)", borderRadius: "2px", overflow: "hidden" }}>
                        <div style={{ width: `${item.importance}%`, height: "100%", background: "linear-gradient(90deg, var(--accent-blue), var(--accent-cyan))" }} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <p style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>ML model loading...</p>
          )}
        </div>
      </div>
    </div>
  );
}
