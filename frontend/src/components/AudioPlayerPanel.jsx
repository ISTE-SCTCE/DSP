import React from "react";
import { Download, Volume2, Sparkles, FileAudio } from "lucide-react";

export default function AudioPlayerPanel({ audioFiles = null, filename = "recording.wav" }) {
  if (!audioFiles || (!audioFiles.original_wav && !audioFiles.noise_reduced_wav)) {
    return (
      <div className="glass-panel" style={{ textAlign: "center", padding: "20px" }}>
        <p style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>
          Audio playback & noise-reduced file will appear here after analysis.
        </p>
      </div>
    );
  }

  const cleanFileName = filename ? `noise_reduced_${filename.split('.')[0]}.wav` : "noise_reduced_voice.wav";

  return (
    <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h3 style={{ fontSize: "0.95rem", color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.05em", display: "flex", alignItems: "center", gap: "8px" }}>
          <Sparkles size={16} style={{ color: "var(--color-natural)" }} />
          Audio Playback & Noise Reduction Inspection
        </h3>
        <span className="brand-badge" style={{ borderColor: "var(--color-natural)", color: "var(--color-natural)" }}>
          DSP DENOISED
        </span>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "16px" }}>
        {/* Original Audio Player */}
        <div style={{ background: "rgba(255, 255, 255, 0.02)", border: "1px solid var(--border-color)", borderRadius: "12px", padding: "16px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "10px" }}>
            <FileAudio size={18} style={{ color: "var(--accent-blue)" }} />
            <span style={{ fontSize: "0.85rem", fontWeight: "600" }}>Original Input Audio</span>
          </div>
          {audioFiles.original_wav ? (
            <audio controls src={audioFiles.original_wav} style={{ width: "100%", height: "40px" }} />
          ) : (
            <p style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Original audio unavailable</p>
          )}
        </div>

        {/* Noise-Reduced Audio Player */}
        <div style={{ background: "rgba(0, 230, 118, 0.04)", border: "1px solid rgba(0, 230, 118, 0.3)", borderRadius: "12px", padding: "16px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <Volume2 size={18} style={{ color: "var(--color-natural)" }} />
              <span style={{ fontSize: "0.85rem", fontWeight: "600", color: "var(--color-natural)" }}>
                Noise-Reduced (Cleaned) Audio
              </span>
            </div>
            {audioFiles.noise_reduced_wav && (
              <a
                href={audioFiles.noise_reduced_wav}
                download={cleanFileName}
                className="btn-mode active"
                style={{ fontSize: "0.75rem", padding: "4px 10px", textDecoration: "none" }}
              >
                <Download size={14} />
                Download WAV
              </a>
            )}
          </div>

          {audioFiles.noise_reduced_wav ? (
            <audio controls src={audioFiles.noise_reduced_wav} style={{ width: "100%", height: "40px" }} />
          ) : (
            <p style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Noise reduced audio generating...</p>
          )}
          <p style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginTop: "8px" }}>
            Cleaned via 80Hz Butterworth High-Pass Filter + Spectral Subtraction Noise Gate
          </p>
        </div>
      </div>
    </div>
  );
}
