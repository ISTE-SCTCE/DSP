import React, { useEffect, useState } from "react";
import { Clock, CheckCircle2, Loader2, Cpu, Timer } from "lucide-react";

export default function ProcessingTracker({ isAnalyzing, activeStage, benchmarks = null }) {
  const [elapsedMs, setElapsedMs] = useState(0);

  useEffect(() => {
    let interval = null;
    if (isAnalyzing) {
      setElapsedMs(0);
      interval = setInterval(() => {
        setElapsedMs((prev) => prev + 50);
      }, 50);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isAnalyzing]);

  const stagesList = [
    {
      id: "preprocessing",
      name: "Stage 1: Pre-processing & Spectral Noise Reduction",
      detail: "Filtering 80Hz hum & applying spectral subtraction noise gate...",
      benchmarkKey: "prep_ms"
    },
    {
      id: "features",
      name: "Stage 2: Feature Extraction (STFT, Pitch & 13 MFCCs)",
      detail: "Computing 13 Mel-Frequency Cepstral Coefficients, YIN F0 Pitch & HNR...",
      benchmarkKey: "features_ms"
    },
    {
      id: "analysis",
      name: "Stage 3: Acoustic Pattern Analysis & Decision Gate",
      detail: "Evaluating micro-jitter, spectral envelope smoothness & anomaly scores...",
      benchmarkKey: "classification_ms"
    }
  ];

  const getStageStatus = (stageId) => {
    if (!isAnalyzing && benchmarks) return "completed";
    const order = ["idle", "acquisition", "preprocessing", "features", "analysis", "decision"];
    const currentIdx = order.indexOf(activeStage);
    const stageIdx = order.indexOf(stageId);

    if (currentIdx > stageIdx) return "completed";
    if (currentIdx === stageIdx) return "active";
    return "pending";
  };

  return (
    <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h3 style={{ fontSize: "0.95rem", color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.05em", display: "flex", alignItems: "center", gap: "8px" }}>
          <Cpu size={18} style={{ color: "var(--accent-cyan)" }} />
          Real-Time Processing & Execution Benchmarks
        </h3>
        <span className="brand-badge" style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <Timer size={12} />
          {isAnalyzing ? `${(elapsedMs / 1000).toFixed(2)}s RUNNING` : benchmarks ? `TOTAL: ${benchmarks.total_ms}ms` : "STANDBY"}
        </span>
      </div>

      {isAnalyzing && (
        <div style={{ background: "rgba(0, 242, 254, 0.05)", border: "1px solid var(--border-focus)", padding: "12px 16px", borderRadius: "10px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem", fontWeight: "600", marginBottom: "8px" }}>
            <span style={{ display: "flex", alignItems: "center", gap: "8px", color: "var(--accent-cyan)" }}>
              <Loader2 size={16} style={{ animation: "spin 1s linear infinite" }} />
              Executing DSP Pipeline... (Current: {activeStage.toUpperCase()})
            </span>
            <span style={{ color: "var(--text-secondary)" }}>Estimated output in ~0.2s</span>
          </div>
          <div style={{ width: "100%", height: "6px", background: "rgba(255, 255, 255, 0.1)", borderRadius: "3px", overflow: "hidden" }}>
            <div
              style={{
                height: "100%",
                width: activeStage === "preprocessing" ? "35%" : activeStage === "features" ? "70%" : activeStage === "analysis" ? "90%" : "100%",
                background: "linear-gradient(90deg, var(--accent-blue), var(--accent-cyan))",
                transition: "width 0.3s ease"
              }}
            />
          </div>
        </div>
      )}

      {/* Step by Step Breakdown */}
      <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
        {stagesList.map((step) => {
          const status = getStageStatus(step.id);
          const stepTiming = benchmarks ? benchmarks[step.benchmarkKey] : null;

          return (
            <div
              key={step.id}
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "10px 14px",
                borderRadius: "8px",
                background: status === "active" ? "rgba(79, 172, 254, 0.1)" : "rgba(255, 255, 255, 0.02)",
                border: status === "active" ? "1px solid var(--accent-cyan)" : "1px solid var(--border-color)",
                transition: "all 0.2s ease"
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                {status === "completed" ? (
                  <CheckCircle2 size={18} style={{ color: "var(--color-natural)" }} />
                ) : status === "active" ? (
                  <Loader2 size={18} style={{ color: "var(--accent-cyan)", animation: "spin 1s linear infinite" }} />
                ) : (
                  <Clock size={18} style={{ color: "var(--text-muted)" }} />
                )}
                <div>
                  <p style={{ fontSize: "0.85rem", fontWeight: "600", color: status === "active" ? "var(--accent-cyan)" : status === "completed" ? "var(--text-primary)" : "var(--text-secondary)" }}>
                    {step.name}
                  </p>
                  <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "2px" }}>
                    {step.detail}
                  </p>
                </div>
              </div>

              <div style={{ textAlign: "right", fontFamily: "'Fira Code', monospace", fontSize: "0.8rem" }}>
                {stepTiming !== null && stepTiming !== undefined ? (
                  <span style={{ color: "var(--color-natural)", fontWeight: "600" }}>{stepTiming} ms</span>
                ) : status === "active" ? (
                  <span style={{ color: "var(--accent-cyan)" }}>In Progress...</span>
                ) : (
                  <span style={{ color: "var(--text-muted)" }}>Waiting</span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
