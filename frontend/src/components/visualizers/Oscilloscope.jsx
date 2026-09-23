import React, { useRef, useEffect } from "react";

export default function Oscilloscope({ audioData = [], isLive = false }) {
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

    // Draw background grid lines
    ctx.strokeStyle = "rgba(255, 255, 255, 0.03)";
    ctx.lineWidth = 1;
    const gridSpacing = 30;
    for (let x = 0; x < width; x += gridSpacing) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += gridSpacing) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Center line
    ctx.strokeStyle = "rgba(0, 242, 254, 0.1)";
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    if (!audioData || audioData.length === 0) {
      // Draw flat line
      ctx.strokeStyle = "#475569";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, height / 2);
      ctx.lineTo(width, height / 2);
      ctx.stroke();
      return;
    }

    // Draw audio wave
    ctx.strokeStyle = isLive ? "#00f2fe" : "#00e676";
    ctx.lineWidth = 2;
    ctx.shadowBlur = 8;
    ctx.shadowColor = isLive ? "rgba(0, 242, 254, 0.5)" : "rgba(0, 230, 118, 0.5)";
    ctx.beginPath();

    const sliceWidth = width / audioData.length;
    let x = 0;

    for (let i = 0; i < audioData.length; i++) {
      // audioData ranges [-1, 1]
      const v = audioData[i];
      const y = (v * (height / 2) * 0.8) + (height / 2);

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
      x += sliceWidth;
    }

    ctx.lineTo(width, height / 2);
    ctx.stroke();

    // Reset shadow for next renders
    ctx.shadowBlur = 0;
  }, [audioData, isLive]);

  return (
    <div className="glass-panel" style={{ flex: 1, minWidth: "280px" }}>
      <h3 style={{ fontSize: "0.95rem", color: "var(--text-secondary)", marginBottom: "12px", textTransform: "uppercase", letterSpacing: "0.05em" }}>
        {isLive ? "Live Oscilloscope (Waveform)" : "Signal Waveform"}
      </h3>
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
