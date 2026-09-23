import React, { useState, useRef } from "react";
import { Upload, AudioLines, FileAudio, AlertCircle } from "lucide-react";

export default function AudioUploader({
  onAnalysisSuccess,
  onStateChange,
  isAnalyzing,
  setIsAnalyzing
}) {
  const [dragActive, setDragActive] = useState(false);
  const [fileName, setFileName] = useState("");
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const onButtonClick = () => {
    fileInputRef.current.click();
  };

  const processFile = async (file) => {
    // Check extension
    const ext = file.name.substring(file.name.lastIndexOf(".")).toLowerCase();
    const allowed = [".wav", ".mp3", ".flac", ".m4a", ".mp4"];
    
    if (!allowed.includes(ext)) {
      setError(`Unsupported file format. Please upload: ${allowed.join(", ")}`);
      return;
    }

    setError(null);
    setFileName(file.name);
    setIsAnalyzing(true);

    // Dynamic stage sequence simulations for UI pipeline visualizer
    const stages = ["acquisition", "preprocessing", "features", "analysis", "decision"];
    
    const animateStages = () => {
      let index = 0;
      const interval = setInterval(() => {
        if (index < stages.length) {
          onStateChange(stages[index]);
          index++;
        } else {
          clearInterval(interval);
        }
      }, 350); // animate nodes sequentially
    };

    animateStages();

    // Call REST endpoint
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/api/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errDetail = await response.json();
        throw new Error(errDetail.detail || "Server error analyzing audio file");
      }

      const result = await response.json();
      
      // Delay slightly if file processes faster than the flowchart animation completes
      setTimeout(() => {
        onAnalysisSuccess(result);
        setIsAnalyzing(false);
      }, 1500);

    } catch (err) {
      console.error(err);
      setError(err.message || "Failed to analyze audio file.");
      setIsAnalyzing(false);
      onStateChange("idle");
    }
  };

  return (
    <div style={{ padding: "12px 0" }}>
      <input
        ref={fileInputRef}
        type="file"
        style={{ display: "none" }}
        accept=".wav,.mp3,.flac,.m4a,.mp4"
        onChange={handleChange}
      />
      
      <div
        className={`uploader-box ${dragActive ? "drag-active" : ""}`}
        onDragEnter={handleDrag}
        onDragOver={handleDrag}
        onDragLeave={handleDrag}
        onDrop={handleDrop}
        onClick={onButtonClick}
      >
        {isAnalyzing ? (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "12px" }}>
            <AudioLines className="pulse-mic" size={40} style={{ color: "var(--accent-cyan)", animation: "spin 3s linear infinite" }} />
            <p style={{ fontWeight: "600" }}>Extracting Acoustic Envelopes...</p>
            <p style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>
              Running DSP transforms on: <code>{fileName}</code>
            </p>
          </div>
        ) : fileName ? (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "12px" }}>
            <FileAudio size={40} style={{ color: "var(--color-natural)" }} />
            <p style={{ fontWeight: "600" }}>{fileName}</p>
            <p style={{ fontSize: "0.75rem", color: "var(--text-secondary)" }}>
              Click or drag another file to replace
            </p>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "12px" }}>
            <Upload size={40} style={{ color: "var(--text-secondary)" }} />
            <p style={{ fontWeight: "600" }}>Upload Audio File</p>
            <p style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
              Drag & drop or browse .wav, .mp3, .flac, or .m4a (Max 2 mins)
            </p>
          </div>
        )}
      </div>

      {error && (
        <div style={{ display: "flex", gap: "8px", alignItems: "center", background: "rgba(239, 68, 68, 0.1)", border: "1px solid rgba(239, 68, 68, 0.2)", padding: "10px 14px", borderRadius: "8px", color: "#fda4af", fontSize: "0.8rem", marginTop: "12px" }}>
          <AlertCircle size={16} style={{ flexShrink: 0 }} />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
