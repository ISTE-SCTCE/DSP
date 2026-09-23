import React from "react";

export default function ConfidenceRing({ value = 0, color = "var(--accent-cyan)" }) {
  const size = 160;
  const strokeWidth = 8;
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const offset = circumference - (value / 100) * circumference;

  return (
    <div className="gauge-container">
      <svg width={size} height={size} className="gauge-svg">
        <circle
          className="gauge-bg"
          cx={size / 2}
          cy={size / 2}
          r={radius}
        />
        <circle
          className="gauge-fill"
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={color}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
        />
      </svg>
      <div className="gauge-value">
        <span className="number" style={{ color }}>{value}%</span>
        <span className="label">Confidence</span>
      </div>
    </div>
  );
}
