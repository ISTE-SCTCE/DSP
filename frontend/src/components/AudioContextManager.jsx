import React, { useState, useRef, useEffect } from "react";
import { Mic, Square, AlertCircle, Loader2 } from "lucide-react";

export default function AudioContextManager({
  onAnalysisSuccess,
  onStateChange,
  isRecording,
  setIsRecording,
  setLiveAnalyser,
  setIsAnalyzing
}) {
  const [error, setError] = useState(null);
  const [recordingTime, setRecordingTime] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);

  const audioContextRef = useRef(null);
  const streamRef = useRef(null);
  const processorRef = useRef(null);
  const recordedSamplesRef = useRef([]);
  const timerRef = useRef(null);

  const startRecording = async () => {
    setError(null);
    recordedSamplesRef.current = [];
    setRecordingTime(0);
    setIsProcessing(false);

    try {
      // Access user microphone
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      streamRef.current = stream;

      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      const audioCtx = new AudioCtx();
      if (audioCtx.state === "suspended") {
        await audioCtx.resume();
      }
      audioContextRef.current = audioCtx;

      const sourceNode = audioCtx.createMediaStreamSource(stream);

      // Analyser for real-time live oscilloscope waveform
      const analyserNode = audioCtx.createAnalyser();
      analyserNode.fftSize = 512;
      sourceNode.connect(analyserNode);
      setLiveAnalyser(analyserNode);

      // ScriptProcessor to capture raw mic PCM float32 samples
      const bufferSize = 4096;
      const processorNode = audioCtx.createScriptProcessor(bufferSize, 1, 1);
      processorRef.current = processorNode;

      sourceNode.connect(processorNode);
      processorNode.connect(audioCtx.destination);

      const sourceSampleRate = audioCtx.sampleRate;
      const targetSampleRate = 22050;

      processorNode.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        // Downsample to 22050 Hz
        const downsampled = downsampleBuffer(inputData, sourceSampleRate, targetSampleRate);
        recordedSamplesRef.current.push(...downsampled);
      };

      setIsRecording(true);
      onStateChange("acquisition");

      // Recording timer
      timerRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);

    } catch (err) {
      console.error("Failed to access microphone", err);
      setError("Microphone access denied or audio device error.");
      setIsRecording(false);
      onStateChange("idle");
    }
  };

  const stopRecording = async () => {
    // 1. Stop timer and audio nodes
    if (timerRef.current) clearInterval(timerRef.current);

    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    setLiveAnalyser(null);
    setIsRecording(false);

    const samples = recordedSamplesRef.current;
    if (!samples || samples.length < 22050 * 0.3) {
      setError("Recording too short. Please speak for at least 1 second.");
      onStateChange("idle");
      return;
    }

    // 2. Convert recorded PCM Float32 array to a WAV Blob
    setIsProcessing(true);
    if (setIsAnalyzing) setIsAnalyzing(true);
    onStateChange("preprocessing");

    try {
      const wavBlob = encodeWAV(new Float32Array(samples), 22050);
      const audioFile = new File([wavBlob], "microphone_recording.wav", { type: "audio/wav" });

      // Animate stages
      const stages = ["preprocessing", "features", "analysis", "decision"];
      let stageIdx = 0;
      const stageInterval = setInterval(() => {
        if (stageIdx < stages.length) {
          onStateChange(stages[stageIdx]);
          stageIdx++;
        } else {
          clearInterval(stageInterval);
        }
      }, 300);

      // 3. Post temporary recording file to backend for full 5-stage DSP analysis
      const formData = new FormData();
      formData.append("file", audioFile);

      const response = await fetch("http://localhost:8000/api/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errDetail = await response.json();
        throw new Error(errDetail.detail || "Error analyzing microphone recording");
      }

      const result = await response.json();

      setTimeout(() => {
        onAnalysisSuccess(result);
        setIsProcessing(false);
        if (setIsAnalyzing) setIsAnalyzing(false);
        onStateChange("decision");
      }, 1000);

    } catch (err) {
      console.error(err);
      setError(err.message || "Failed to analyze microphone recording.");
      setIsProcessing(false);
      if (setIsAnalyzing) setIsAnalyzing(false);
      onStateChange("idle");
    }
  };

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  // Downsample helper
  const downsampleBuffer = (buffer, fromRate, toRate) => {
    if (fromRate === toRate) return buffer;
    const ratio = fromRate / toRate;
    const newLength = Math.round(buffer.length / ratio);
    const result = new Float32Array(newLength);
    let offsetResult = 0;
    let offsetBuffer = 0;
    while (offsetResult < result.length) {
      const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
      let accum = 0;
      let count = 0;
      for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i++) {
        accum += buffer[i];
        count++;
      }
      result[offsetResult] = count > 0 ? accum / count : 0;
      offsetResult++;
      offsetBuffer = nextOffsetBuffer;
    }
    return result;
  };

  // Helper to create WAV Blob from PCM float32 array
  const encodeWAV = (samples, sampleRate) => {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };

    /* RIFF identifier */
    writeString(0, 'RIFF');
    /* RIFF chunk length */
    view.setUint32(4, 36 + samples.length * 2, true);
    /* RIFF type */
    writeString(8, 'WAVE');
    /* format chunk identifier */
    writeString(12, 'fmt ');
    /* format chunk length */
    view.setUint32(16, 16, true);
    /* sample format (raw PCM) */
    view.setUint16(20, 1, true);
    /* channel count (mono) */
    view.setUint16(22, 1, true);
    /* sample rate */
    view.setUint32(24, sampleRate, true);
    /* byte rate (sampleRate * 2) */
    view.setUint32(28, sampleRate * 2, true);
    /* block align */
    view.setUint16(32, 2, true);
    /* bits per sample */
    view.setUint16(34, 16, true);
    /* data chunk identifier */
    writeString(36, 'data');
    /* data chunk length */
    view.setUint32(40, samples.length * 2, true);

    // Convert Float32 to Int16
    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return new Blob([view], { type: 'audio/wav' });
  };

  const formatTimer = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  return (
    <div className="stream-ctrl">
      <button
        onClick={toggleRecording}
        className={`mic-btn ${isRecording ? "active" : ""}`}
        disabled={isProcessing}
        title={isRecording ? "Click to Stop & Analyze" : "Click to Start Microphone Recording"}
      >
        {isProcessing ? (
          <Loader2 size={32} className="spin-loader" style={{ animation: "spin 1s linear infinite" }} />
        ) : isRecording ? (
          <Square size={28} style={{ color: "#fff" }} />
        ) : (
          <Mic size={28} />
        )}
      </button>

      <div style={{ textAlign: "center" }}>
        <p style={{ fontSize: "0.95rem", fontWeight: "600", color: isRecording ? "var(--color-synthetic)" : "var(--text-primary)" }}>
          {isProcessing ? "Processing Recorded File..." : isRecording ? `Recording... (${formatTimer(recordingTime)})` : "Microphone Ready"}
        </p>
        <p style={{ fontSize: "0.75rem", color: "var(--text-secondary)", marginTop: "4px" }}>
          {isProcessing && "Saving temp WAV → Running 5-Stage DSP Analysis..."}
          {isRecording && "Speak into mic. Click stop button when finished to analyze."}
          {!isRecording && !isProcessing && "Click microphone button to record audio"}
        </p>
      </div>

      {error && (
        <div style={{ display: "flex", gap: "8px", alignItems: "center", background: "rgba(239, 68, 68, 0.1)", border: "1px solid rgba(239, 68, 68, 0.2)", padding: "10px 14px", borderRadius: "8px", color: "#fda4af", fontSize: "0.8rem", maxWidth: "280px" }}>
          <AlertCircle size={16} style={{ flexShrink: 0 }} />
          <span>{error}</span>
        </div>
      )}
    </div>
  );
}
