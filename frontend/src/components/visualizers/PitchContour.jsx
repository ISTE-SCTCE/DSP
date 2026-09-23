import React, { useRef, useEffect } from "react";
import { Gauge } from "lucide-react";

export default function PitchContour({ pitchData = null, isLive = false }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.fillStyle = "#090a0f";
    ctx.fillRect(0, 0, width, height);

    // Draw grid lines
    ctx.strokeStyle = "rgba(255, 255, 255, 0.03)";
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    // Horizontal pitch lines in Hz (50Hz - 500Hz)
    const pitchGrid = [50, 100, 150, 200, 250, 300, 400, 500];
    ctx.fillStyle = "rgba(255, 255, 255, 0.3)";
    ctx.font = "9px monospace";
    pitchGrid.forEach((hz) => {
      const y = height - ((hz - 50) / 450) * height;
      ctx.beginPath();
      ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
      ctx.fillText(`${hz}Hz`, 6, y - 3);
    });

    const f0 = pitchData?.f0 || [];
    if (!f0 || f0.length === 0) {
      ctx.fillStyle = "rgba(255, 255, 255, 0.2)";
      ctx.font = "11px monospace";
      ctx.fillText("Waiting for fundamental frequency (F0)...", width / 2 - 120, height / 2);
      return;
    }

    // Draw F0 Contour line
    ctx.lineWidth = 3;
    ctx.strokeStyle = "#a18cd1"; // Violet contour
    ctx.shadowBlur = 8;
    ctx.shadowColor = "rgba(161, 140, 209, 0.5)";

    const sliceWidth = width / f0.length;
    let drawingSegment = false;

    ctx.beginPath();
    for (let i = 0; i < f0.length; i++) {
      const hz = f0[i];
      const x = i * sliceWidth;

      if (hz > 0) {
        // Map [50, 500] Hz to height
        const y = height - Math.min(1.0, Math.max(0.0, (hz - 50) / 450)) * height;
        if (!drawingSegment) {
          ctx.moveTo(x, y);
          drawingSegment = true;
        } else {
          ctx.lineTo(x, y);
        }
      } else {
        if (drawingSegment) {
          ctx.stroke();
          ctx.beginPath();
          drawingSegment = false;
        }
      }
    }
    if (drawingSegment) {
      ctx.stroke();
    }

    ctx.shadowBlur = 0;

    // Stats summary overlay
    const stats = pitchData?.stats;
    if (stats && stats.f0_mean > 0) {
      ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
      ctx.fillRect(width - 160, 8, 152, 60);
      ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
      ctx.strokeRect(width - 160, 8, 152, 60);

      ctx.fillStyle = "#00f2fe";
      ctx.font = "10px Outfit, sans-serif";
      ctx.fillText(`Mean F0: ${stats.f0_mean.toFixed(1)} Hz`, width - 152, 24);
      ctx.fillStyle = "#a7f3d0";
      ctx.fillText(`Range: ${stats.f0_min ? stats.f0_min.toFixed(0) : 0}-${stats.f0_max ? stats.f0_max.toFixed(0) : 0} Hz`, width - 152, 38);
      ctx.fillStyle = "#e2e8f0";
      ctx.fillText(`Jitter: ${stats.jitter_hz ? stats.jitter_hz.toFixed(2) : 0} Hz`, width - 152, 52);
    }
  }, [pitchData, isLive]);

  return (
    <div className="glass-panel" style={{ flex: 1, minWidth: "280px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
        <h3 style={{ fontSize: "0.95rem", color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.05em", display: "flex", alignItems: "center", gap: "8px" }}>
          <Gauge size={16} style={{ color: "#a18cd1" }} />
          Real Frequency Pitch (F0) Contour
        </h3>
        <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontFamily: "monospace" }}>
          YIN ALGORITHM
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
