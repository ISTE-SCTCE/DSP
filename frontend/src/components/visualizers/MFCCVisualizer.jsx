import React, { useRef, useEffect } from "react";
import { Activity } from "lucide-react";

export default function MFCCVisualizer({ mfccData = null, mfccMean = [] }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;

    // Clear
    ctx.fillStyle = "#090a0f";
    ctx.fillRect(0, 0, width, height);

    const matrix = mfccData?.matrix || []; // 2D array [13][time_steps]
    const means = mfccData?.means || mfccMean || [];

    if (matrix.length > 0) {
      // Draw 2D MFCC Heatmap (13 coefficients vs time)
      const numCoeffs = matrix.length; // 13
      const numTimes = matrix[0].length;

      const dx = width / numTimes;
      const dy = height / numCoeffs;

      for (let c = 0; c < numCoeffs; c++) {
        const row = matrix[c];
        const rowMin = Math.min(...row);
        const rowMax = Math.max(...row);
        const range = rowMax - rowMin || 1;

        for (let t = 0; t < numTimes; t++) {
          const val = row[t];
          const norm = (val - rowMin) / range; // [0, 1]

          // Color mapping: purple/blue to bright cyan/yellow
          const hue = 260 - norm * 200; // 260 (purple) to 60 (yellow)
          const sat = 90;
          const lum = 15 + norm * 55;

          ctx.fillStyle = `hsl(${hue}, ${sat}%, ${lum}%)`;
          ctx.fillRect(t * dx, c * dy, dx + 1, dy + 1);
        }
      }

      // Overlay label for 13 MFCC bands
      ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
      ctx.font = "9px monospace";
      for (let c = 0; c < 13; c += 2) {
        const y = c * dy + dy / 2 + 3;
        ctx.fillText(`MFCC ${c}`, 6, y);
      }

    } else if (means.length > 0) {
      // Draw 13-Bar Spectrum
      const numBars = means.length;
      const barWidth = (width - 40) / numBars;
      const maxVal = Math.max(...means.map(Math.abs)) || 1;

      // Draw baseline
      const zeroY = height / 2;
      ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
      ctx.beginPath();
      ctx.moveTo(0, zeroY);
      ctx.lineTo(width, zeroY);
      ctx.stroke();

      means.forEach((val, idx) => {
        const x = 20 + idx * barWidth;
        const norm = val / maxVal;
        const barH = norm * (height / 2 - 20);

        const gradient = ctx.createLinearGradient(0, zeroY, 0, zeroY - barH);
        gradient.addColorStop(0, "rgba(0, 242, 254, 0.8)");
        gradient.addColorStop(1, "rgba(79, 172, 254, 0.4)");

        ctx.fillStyle = gradient;
        ctx.fillRect(x + 4, barH < 0 ? zeroY : zeroY - barH, barWidth - 8, Math.abs(barH));

        // Label
        ctx.fillStyle = "rgba(255, 255, 255, 0.6)";
        ctx.font = "9px monospace";
        ctx.fillText(`c${idx}`, x + barWidth / 2 - 6, height - 6);
      });
    } else {
      ctx.fillStyle = "rgba(255, 255, 255, 0.2)";
      ctx.font = "11px monospace";
      ctx.fillText("Waiting for MFCC spectral data...", width / 2 - 90, height / 2);
    }
  }, [mfccData, mfccMean]);

  return (
    <div className="glass-panel" style={{ flex: 1, minWidth: "280px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
        <h3 style={{ fontSize: "0.95rem", color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.05em", display: "flex", alignItems: "center", gap: "8px" }}>
          <Activity size={16} style={{ color: "var(--accent-cyan)" }} />
          13-Band MFCC Spectrum & Heatmap
        </h3>
        <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontFamily: "monospace" }}>
          CEPSTRAL ENVELOPE
        </span>
      </div>
      <canvas
        ref={canvasRef}
        width={600}
        height={180}
        style={{
          width: "100%",
          height: "180px",
          borderRadius: "8px",
          background: "#090a0f",
          border: "1px solid var(--border-color)",
          display: "block"
        }}
      />
    </div>
  );
}
