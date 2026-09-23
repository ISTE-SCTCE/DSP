import React, { useEffect, useRef } from "react";

export default function DiagnosticsTerminal({ anomalies = [] }) {
  const terminalEndRef = useRef(null);

  useEffect(() => {
    if (terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [anomalies]);

  const getLineClass = (line) => {
    const text = line.toLowerCase();
    if (text.includes("anomaly")) return "terminal-line danger";
    if (text.includes("warning")) return "terminal-line warning";
    if (text.includes("natural") || text.includes("no significant")) return "terminal-line success";
    return "terminal-line system";
  };

  return (
    <div className="terminal-window">
      <div className="terminal-header">
        <div className="terminal-dots">
          <span className="dot red"></span>
          <span className="dot yellow"></span>
          <span className="dot green"></span>
        </div>
        <span>ANOMALY_DIAGNOSTICS_v1.0</span>
      </div>
      <div className="terminal-body">
        <div className="terminal-line system">
          [{new Date().toLocaleTimeString()}] System loaded. Ready for audio diagnostic capture...
        </div>
        {anomalies.map((anomaly, idx) => (
          <div key={idx} className={getLineClass(anomaly)}>
            [{new Date().toLocaleTimeString()}] {anomaly}
          </div>
        ))}
        <div ref={terminalEndRef} />
      </div>
    </div>
  );
}
