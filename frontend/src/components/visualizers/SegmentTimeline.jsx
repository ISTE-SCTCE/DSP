import React from "react";
import { Clock, Layers, Sparkles } from "lucide-react";

export default function SegmentTimeline({ segments = null }) {
  if (!segments || !segments.segment_timeline || segments.segment_timeline.length === 0) {
    return null;
  }

  const timeline = segments.segment_timeline;
  const synCount = segments.syn_segments;
  const humanCount = segments.human_segments;
  const total = segments.total_segments;

  return (
    <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h3 style={{ fontSize: "0.95rem", color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.05em", display: "flex", alignItems: "center", gap: "8px" }}>
          <Layers size={16} style={{ color: "var(--color-modified)" }} />
          Segmented Temporal Diarization (1.5s Windows)
        </h3>
        <span style={{ fontSize: "0.75rem", fontFamily: "monospace", color: synCount > 0 ? "var(--color-synthetic)" : "var(--color-natural)" }}>
          {synCount > 0 ? `⚠️ ${synCount}/${total} AI Voice Segments` : `✓ All ${total} Segments Human`}
        </span>
      </div>

      {/* Visual Timeline Bar */}
      <div style={{ display: "flex", width: "100%", height: "28px", borderRadius: "8px", overflow: "hidden", background: "#090a0f", border: "1px solid var(--border-color)", padding: "3px", gap: "3px" }}>
        {timeline.map((item, idx) => {
          const isAI = item.type === "Acoustic anomaly segment";
          return (
            <div
              key={idx}
              title={`${item.time} — ${item.type}: ${item.reasons.join(", ")}`}
              style={{
                flex: 1,
                borderRadius: "4px",
                background: isAI
                  ? "linear-gradient(135deg, rgba(255,23,68,0.8), rgba(213,0,249,0.8))"
                  : "linear-gradient(135deg, rgba(0,230,118,0.6), rgba(0,242,254,0.6))",
                boxShadow: isAI ? "0 0 8px rgba(255,23,68,0.4)" : "0 0 6px rgba(0,230,118,0.2)",
                cursor: "pointer",
                transition: "all 0.2s ease"
              }}
            />
          );
        })}
      </div>

      {/* Detailed Segment Cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: "10px", maxHeight: "160px", overflowY: "auto", paddingRight: "4px" }}>
        {timeline.map((item, idx) => {
          const isAI = item.type === "Acoustic anomaly segment";
          return (
            <div
              key={idx}
              style={{
                background: isAI ? "rgba(255, 23, 68, 0.06)" : "rgba(0, 230, 118, 0.04)",
                border: isAI ? "1px solid rgba(255, 23, 68, 0.3)" : "1px solid rgba(0, 230, 118, 0.2)",
                borderRadius: "8px",
                padding: "8px 12px"
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.75rem", fontWeight: "600" }}>
                <span style={{ display: "flex", alignItems: "center", gap: "4px", color: "var(--text-secondary)" }}>
                  <Clock size={12} />
                  {item.time}
                </span>
                <span style={{ color: isAI ? "var(--color-synthetic)" : "var(--color-natural)" }}>
                  {isAI ? "Acoustic anomaly" : "No configured anomaly"}
                </span>
              </div>
              <p style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginTop: "4px", lineHeight: "1.3" }}>
                {item.reasons.join(" • ")}
              </p>
              {item.evidence && (
                <p style={{ fontSize: "0.66rem", color: "var(--text-secondary)", marginTop: "6px", fontFamily: "monospace", lineHeight: "1.35" }}>
                  F0 σ {item.evidence.f0_std_hz} Hz · HNR {item.evidence.hnr_db} dB · ΔMFCC {item.evidence.mfcc_delta_energy}
                </p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
