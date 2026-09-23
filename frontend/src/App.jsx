import React, { useState } from "react";
import { ShieldAlert, Fingerprint, AudioLines, Brain, Layers, BookOpen, CheckCircle2, AlertTriangle } from "lucide-react";
import AudioUploader from "./components/AudioUploader";
import AudioContextManager from "./components/AudioContextManager";
import PipelineFlowchart from "./components/PipelineFlowchart";
import DiagnosticsTerminal from "./components/DiagnosticsTerminal";
import ConfidenceRing from "./components/ConfidenceRing";
import AudioPlayerPanel from "./components/AudioPlayerPanel";
import ProcessingTracker from "./components/ProcessingTracker";
import TrainingPage from "./components/TrainingPage";

// Visualizer components
import Oscilloscope from "./components/visualizers/Oscilloscope";
import Spectrogram from "./components/visualizers/Spectrogram";
import PitchContour from "./components/visualizers/PitchContour";
import MFCCVisualizer from "./components/visualizers/MFCCVisualizer";
import SegmentTimeline from "./components/visualizers/SegmentTimeline";

export default function App() {
  const [activeTab, setActiveTab] = useState("detect"); // detect | train
  const [activeMode, setActiveMode] = useState("upload"); // upload | live

  // Pipeline state
  const [pipelineStage, setPipelineStage] = useState("idle"); // idle | acquisition | preprocessing | features | analysis | decision
  const [isFlowing, setIsFlowing] = useState(false);

  // Analysis Output States
  const [classification, setClassification] = useState({
    label: "No Input Source",
    confidence: 0,
    ai_percentage: 0,
    human_percentage: 0,
    anomalies: ["System idle. Provide file or live audio stream."]
  });

  const [features, setFeatures] = useState({
    mfcc_mean: [],
    pitch_stats: { f0_mean: 0, f0_std: 0, jitter_hz: 0, f0_min: 0, f0_max: 0 },
    hnr: { hnr_db: 0, hnr_2_4khz_db: 0 },
    ste: { mean: 0, std: 0 },
    zcr: { mean: 0, std: 0 },
    rolloff_mean: 0
  });

  // Visualization & Audio File States
  const [waveform, setWaveform] = useState([]);
  const [spectrogram, setSpectrogram] = useState(null);
  const [pitchContour, setPitchContour] = useState(null);
  const [mfccData, setMfccData] = useState(null);
  const [audioFiles, setAudioFiles] = useState(null);
  const [benchmarks, setBenchmarks] = useState(null);
  const [methodology, setMethodology] = useState([]);
  const [dataQuality, setDataQuality] = useState(null);
  const [liveAnalyser, setLiveAnalyser] = useState(null);
  const [activeFilename, setActiveFilename] = useState("");

  // Loading States
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isRecording, setIsRecording] = useState(false);

  // Triggered when static file upload OR microphone recording completes
  const handleUploadSuccess = (result) => {
    setClassification(result.classification);
    setFeatures(result.features);
    setWaveform(result.waveform.data);
    setSpectrogram(result.spectrogram);
    setPitchContour(result.pitch_contour);
    setMfccData(result.mfcc_data);
    setAudioFiles(result.audio_files);
    setBenchmarks(result.benchmarks);
    setMethodology(result.methodology || []);
    setDataQuality(result.data_quality || null);
    setActiveFilename(result.filename || "recording.wav");
    setPipelineStage("decision");
    setIsFlowing(false);
  };

  // Reset function when switching modes
  const handleModeSwitch = (mode) => {
    setActiveMode(mode);
    setPipelineStage("idle");
    setIsFlowing(false);
    setClassification({
      label: "Waiting for Input",
      confidence: 0,
      ai_percentage: 0,
      human_percentage: 0,
      anomalies: ["Select mode and trigger input to capture anomalies."]
    });
    setFeatures({
      mfcc_mean: [],
      pitch_stats: { f0_mean: 0, f0_std: 0, jitter_hz: 0, f0_min: 0, f0_max: 0 },
      hnr: { hnr_db: 0, hnr_2_4khz_db: 0 },
      ste: { mean: 0, std: 0 },
      zcr: { mean: 0, std: 0 },
      rolloff_mean: 0
    });
    setWaveform([]);
    setSpectrogram(null);
    setPitchContour(null);
    setMfccData(null);
    setAudioFiles(null);
    setBenchmarks(null);
    setMethodology([]);
    setDataQuality(null);
    setLiveAnalyser(null);
  };

  // Class helper for status badges
  const getStatusClass = (label) => {
    if (label.includes("Natural")) return "natural";
    if (label.includes("Modified")) return "modified";
    if (label.includes("Synthetic") || label.includes("AI") || label.includes("AI-Generated")) return "synthetic";
    return "idle";
  };

  const getStatusColor = (label) => {
    if (label.includes("Natural")) return "var(--color-natural)";
    if (label.includes("Modified")) return "var(--color-modified)";
    if (label.includes("Synthetic") || label.includes("AI") || label.includes("AI-Generated")) return "var(--color-synthetic)";
    return "var(--text-secondary)";
  };

  const getStatusCardClass = (label) => {
    if (label.includes("Natural")) return "active-glow-natural";
    if (label.includes("Modified")) return "active-glow-modified";
    if (label.includes("Synthetic") || label.includes("AI") || label.includes("AI-Generated")) return "active-glow-synthetic";
    return "";
  };

  return (
    <div>
      {/* Header */}
      <header className="app-header">
        <div className="brand">
          <ShieldAlert size={24} style={{ color: "var(--accent-cyan)" }} />
          <h1>VOICE CAMOUFLAGE DETECTOR</h1>
          <span className="brand-badge">DSP + ML ENGINE</span>
        </div>

        {/* Global Page Tabs: Detection Mode vs Training Page */}
        <div className="controls-header">
          <button
            onClick={() => setActiveTab("detect")}
            className={`btn-mode ${activeTab === "detect" ? "active" : ""}`}
          >
            <Layers size={16} />
            Detection & Analysis
          </button>
          <button
            onClick={() => setActiveTab("train")}
            className={`btn-mode ${activeTab === "train" ? "active" : ""}`}
          >
            <Brain size={16} />
            Model Training & Feedback
          </button>
        </div>
      </header>

      {/* Main App Page Rendering */}
      {activeTab === "train" ? (
        <TrainingPage />
      ) : (
        <React.Fragment>
          {/* Audio Input Selector Sub-Header */}
          <div style={{ maxWidth: "1600px", margin: "16px auto 0 auto", padding: "0 24px", display: "flex", gap: "12px", justifyContent: "flex-start" }}>
            <button
              onClick={() => handleModeSwitch("upload")}
              className={`btn-mode ${activeMode === "upload" ? "active" : ""}`}
              disabled={isRecording || isAnalyzing}
            >
              <AudioLines size={16} />
              File Upload Mode
            </button>
            <button
              onClick={() => handleModeSwitch("live")}
              className={`btn-mode ${activeMode === "live" ? "active" : ""}`}
              disabled={isAnalyzing}
            >
              <Fingerprint size={16} />
              Microphone Mode
            </button>
          </div>

          {/* Main Grid */}
          <div className="dashboard-grid">
            {/* Left Side: Controller / Results */}
            <div className="main-content">
              <div className="glass-panel">
                <h2 style={{ fontSize: "1.1rem", marginBottom: "16px", fontWeight: "600" }}>Audio Acquisition</h2>
                {activeMode === "upload" ? (
                  <AudioUploader
                    onAnalysisSuccess={handleUploadSuccess}
                    onStateChange={setPipelineStage}
                    isAnalyzing={isAnalyzing}
                    setIsAnalyzing={setIsAnalyzing}
                  />
                ) : (
                  <AudioContextManager
                    onAnalysisSuccess={handleUploadSuccess}
                    onStateChange={setPipelineStage}
                    isRecording={isRecording}
                    setIsRecording={setIsRecording}
                    setLiveAnalyser={setLiveAnalyser}
                    setIsAnalyzing={setIsAnalyzing}
                  />
                )}
              </div>

              {/* Decision Results */}
              <div className={`glass-panel ${getStatusCardClass(classification.label)}`}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
                  <h2 style={{ fontSize: "1.1rem", fontWeight: "600" }}>Classification Result</h2>
                  <span className={`status-badge ${getStatusClass(classification.label)}`}>
                    {classification.label}
                  </span>
                </div>

                <div style={{ display: "flex", justifyContent: "center" }}>
                  <ConfidenceRing
                    value={classification.confidence}
                    color={getStatusColor(classification.label)}
                  />
                </div>

                {/* Mixed Input AI Percentage Breakdown Bar */}
                {(classification.ai_percentage !== undefined && classification.ai_percentage > 0) && (
                  <div style={{ background: "rgba(0, 0, 0, 0.25)", border: "1px solid var(--border-color)", borderRadius: "10px", padding: "12px", marginTop: "16px" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem", fontWeight: "600", marginBottom: "6px" }}>
                      <span style={{ color: "var(--color-synthetic)" }}>🤖 AI Voice: {classification.ai_percentage}%</span>
                      <span style={{ color: "var(--color-natural)" }}>🗣️ Human Speech: {classification.human_percentage}%</span>
                    </div>
                    <div style={{ width: "100%", height: "8px", background: "rgba(255, 255, 255, 0.08)", borderRadius: "4px", overflow: "hidden", display: "flex" }}>
                      <div style={{ width: `${classification.ai_percentage}%`, height: "100%", background: "linear-gradient(90deg, #ff1744, #d500f9)" }} />
                      <div style={{ width: `${classification.human_percentage}%`, height: "100%", background: "linear-gradient(90deg, #00e676, #00f2fe)" }} />
                    </div>
                  </div>
                )}

                {/* ML Prediction Probability Banner */}
                {classification.ml_prediction?.available && (
                  <div style={{ background: "rgba(0, 242, 254, 0.05)", border: "1px solid rgba(0, 242, 254, 0.2)", borderRadius: "10px", padding: "10px 14px", marginTop: "16px", fontSize: "0.8rem" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
                      <span style={{ fontWeight: "600", color: "var(--accent-cyan)" }}>
                        🤖 ML Model Prediction: {classification.ml_prediction.predicted_class}
                      </span>
                      <span style={{ fontSize: "0.7rem", color: "var(--text-muted)", fontFamily: "monospace" }}>
                        ExtraTrees Classifier
                      </span>
                    </div>
                    <div style={{ display: "flex", gap: "12px", fontSize: "0.75rem", color: "var(--text-secondary)" }}>
                      <span>AI Prob: <strong style={{ color: "var(--color-synthetic)" }}>{classification.ml_prediction.ai_probability}%</strong></span>
                      <span>Human Prob: <strong style={{ color: "var(--color-natural)" }}>{classification.ml_prediction.human_probability}%</strong></span>
                    </div>
                  </div>
                )}

                {/* DSP Features Summary Badges */}
                <div className="feature-badge-grid">
                  <div className="feature-badge">
                    <div className="label">Mean Pitch (F0)</div>
                    <div className="value" style={{ color: "var(--accent-cyan)" }}>
                      {features.pitch_stats.f0_mean > 0 ? `${features.pitch_stats.f0_mean.toFixed(1)} Hz` : "N/A"}
                    </div>
                  </div>
                  <div className="feature-badge">
                    <div className="label">Pitch Jitter</div>
                    <div className="value" style={{ color: "var(--accent-blue)" }}>
                      {features.pitch_stats.f0_mean > 0 ? `${features.pitch_stats.jitter_hz.toFixed(2)} Hz` : "N/A"}
                    </div>
                  </div>
                  <div className="feature-badge">
                    <div className="label">Harmonicity (HNR)</div>
                    <div className="value" style={{ color: "var(--color-natural)" }}>
                      {features.hnr.hnr_db !== 0 ? `${features.hnr.hnr_db.toFixed(1)} dB` : "N/A"}
                    </div>
                  </div>
                  <div className="feature-badge">
                    <div className="label">Spectral Rolloff</div>
                    <div className="value" style={{ color: "var(--color-modified)" }}>
                      {features.rolloff_mean > 0 ? `${(features.rolloff_mean / 1000).toFixed(1)} kHz` : "N/A"}
                    </div>
                  </div>
                </div>
              </div>

              {/* Audio Inspection & Noise Reduced Audio Player */}
              <AudioPlayerPanel audioFiles={audioFiles} filename={activeFilename} />
            </div>

            {/* Right Side: Tracker, Flowchart & Visualizers */}
            <div className="main-content">
              {/* Real-time process tracker & execution benchmarks */}
              <ProcessingTracker
                isAnalyzing={isAnalyzing}
                activeStage={pipelineStage}
                benchmarks={benchmarks}
              />

              {/* Animated 5-Stage Node Diagram */}
              <PipelineFlowchart
                activeStage={pipelineStage}
                isFlowing={isFlowing || isRecording || isAnalyzing}
                features={features}
                classification={classification}
                benchmarks={benchmarks}
                waveform={waveform}
              />

              {/* Signal Visualizations */}
              <div className="charts-row">
                <Oscilloscope audioData={waveform} isLive={activeMode === "live" && isRecording} />
                <PitchContour pitchData={pitchContour} isLive={activeMode === "live"} />
              </div>

              <div className="charts-row">
                <Spectrogram spectrogramData={spectrogram} liveAnalyser={isRecording ? liveAnalyser : null} />
                <MFCCVisualizer mfccData={mfccData} mfccMean={features.mfcc_mean} />
              </div>

              {/* Segmented Temporal Diarization Timeline */}
              <SegmentTimeline segments={classification.segments} />

              {/* This panel is driven entirely by the backend methodology registry. */}
              {(methodology.length > 0 || dataQuality) && (
                <section className="glass-panel methodology-panel" aria-label="DSP equations and validation">
                  <div className="methodology-heading">
                    <div>
                      <p className="eyebrow"><BookOpen size={14} /> DSP LAB EVIDENCE</p>
                      <h2>Equations used in this result</h2>
                    </div>
                    {dataQuality && (
                      <span className={`quality-pill ${dataQuality.valid ? "valid" : "invalid"}`}>
                        {dataQuality.valid ? <CheckCircle2 size={14} /> : <AlertTriangle size={14} />}
                        {dataQuality.valid ? "Input checks passed" : "Input needs attention"}
                      </span>
                    )}
                  </div>
                  {dataQuality && (
                    <p className="quality-copy">Peak: {dataQuality.peak_amplitude} · Voiced frames: {(dataQuality.voiced_ratio * 100).toFixed(1)}%{dataQuality.warning ? ` · ${dataQuality.warning}` : ""}</p>
                  )}
                  <div className="equation-grid">
                    {methodology.map((method) => (
                      <article className="equation-card" key={method.id}>
                        <p>{method.topic}</p>
                        <code>{method.equation}</code>
                        <span>{method.purpose}</span>
                        {Object.keys(method.parameters || {}).length > 0 && (
                          <small>{Object.entries(method.parameters).map(([key, value]) => `${key}: ${value}`).join(" · ")}</small>
                        )}
                      </article>
                    ))}
                  </div>
                  <p className="method-note">Verdict basis: {classification.decision_basis || "Awaiting analysis."}</p>
                </section>
              )}

              <DiagnosticsTerminal anomalies={classification.anomalies} />
            </div>
          </div>
        </React.Fragment>
      )}
    </div>
  );
}
