import React, { useRef, useEffect } from "react";
import { Sliders } from "lucide-react";

export default function Spectrogram({ spectrogramData = null, liveAnalyser = null }) {
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const tempCanvasRef = useRef(null);

  // Effect for Static File Analysis Spectrogram
  useEffect(() => {
    if (liveAnalyser || !spectrogramData) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;

    const data = spectrogramData.data || []; // 2D array [freq][time]
    if (data.length === 0) return;

    const numFreqs = data.length;
    const numTimes = data[0].length;

    const dx = width / numTimes;
    const dy = height / numFreqs;

    // Draw the full high-precision STFT heatmap
    for (let f = 0; f < numFreqs; f++) {
      for (let t = 0; t < numTimes; t++) {
        const val = data[numFreqs - 1 - f][t]; // Invert Y axis (low freqs at bottom)
        const normVal = Math.min(255, Math.max(0, (val + 80) * (255 / 80)));

        // Viridis/Inferno Palette: Deep purple → Blue → Cyan → Green → Yellow → Red
        let hue, sat, lum;
        if (normVal < 30) {
          hue = 260; sat = 80; lum = (normVal / 30) * 15;
        } else if (normVal < 100) {
          hue = 240 - ((normVal - 30) / 70) * 80; sat = 90; lum = 15 + ((normVal - 30) / 70) * 30;
        } else if (normVal < 180) {
          hue = 160 - ((normVal - 100) / 80) * 100; sat = 95; lum = 45 + ((normVal - 100) / 80) * 15;
        } else {
          hue = 60 - ((normVal - 180) / 75) * 60; sat = 100; lum = 60 + ((normVal - 180) / 75) * 25;
        }

        ctx.fillStyle = `hsl(${hue}, ${sat}%, ${lum}%)`;
        ctx.fillRect(t * dx, f * dy, dx + 1, dy + 1);
      }
    }

    // Y-Axis Frequency Grid Overlay (11kHz down to 0Hz)
    const freqsList = [11000, 8000, 5000, 2500, 1000, 0];
    ctx.font = "9px monospace";
    ctx.fillStyle = "rgba(255, 255, 255, 0.7)";
    ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";

    freqsList.forEach((hz) => {
      const y = height - (hz / 11000) * height;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();

      ctx.fillStyle = "rgba(9, 10, 15, 0.8)";
      ctx.fillRect(4, Math.max(2, y - 10), 45, 12);
      ctx.fillStyle = "#00f2fe";
      ctx.fillText(`${(hz / 1000).toFixed(1)}kHz`, 6, Math.max(10, y));
    });

  }, [spectrogramData, liveAnalyser]);

  // Effect for Live Microphone Rolling Spectrogram
  useEffect(() => {
    if (!liveAnalyser) {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;

    if (!tempCanvasRef.current) {
      tempCanvasRef.current = document.createElement("canvas");
    }
    const tempCanvas = tempCanvasRef.current;
    tempCanvas.width = width;
    tempCanvas.height = height;
    const tempCtx = tempCanvas.getContext("2d");

    const bufferLength = liveAnalyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const drawLive = () => {
      liveAnalyser.getByteFrequencyData(dataArray);

      tempCtx.drawImage(canvas, 0, 0);
      ctx.drawImage(tempCanvas, -2, 0);

      const colWidth = 2;
      const barHeight = height / bufferLength;

      for (let i = 0; i < bufferLength; i++) {
        const value = dataArray[i];
        const hue = (1.0 - value / 255) * 240;
        const lum = value > 5 ? (value / 255) * 65 : 0;

        ctx.fillStyle = `hsl(${hue}, 100%, ${lum}%)`;
        ctx.fillRect(width - colWidth, height - (i * barHeight), colWidth, barHeight + 1);
      }

      animationRef.current = requestAnimationFrame(drawLive);
    };

    drawLive();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [liveAnalyser]);

  return (
    <div className="glass-panel" style={{ flex: 1, minWidth: "280px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
        <h3 style={{ fontSize: "0.95rem", color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.05em", display: "flex", alignItems: "center", gap: "8px" }}>
          <Sliders size={16} style={{ color: "var(--accent-cyan)" }} />
          STFT 2D Spectrogram (0–11 kHz vs Time)
        </h3>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontFamily: "monospace" }}>
            POWER: -80 dB to 0 dB
          </span>
          <span style={{ fontSize: "0.7rem", color: "var(--accent-cyan)", fontFamily: "monospace", padding: "2px 6px", background: "rgba(0,242,254,0.1)", borderRadius: "4px" }}>
            {liveAnalyser ? "LIVE STREAM" : "STFT MATRIX"}
          </span>
        </div>
      </div>
      <canvas
        ref={canvasRef}
        width={600}
        height={220}
        style={{
          width: "100%",
          height: "220px",
          borderRadius: "8px",
          background: "#050608",
          border: "1px solid var(--border-color)",
          display: "block"
        }}
      />
    </div>
  );
}
