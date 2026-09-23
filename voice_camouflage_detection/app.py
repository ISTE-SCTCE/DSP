"""
app.py — Streamlit Web Application
====================================
KTU S5 DSP Project: Voice Camouflage Detection

Pure UI layer — imports from pipeline.py, never does DSP itself.
Every number shown is sourced from PipelineResult so charts and text
always agree with the computed values.

Run:
    streamlit run app.py
"""

import os
import sys
import io
import json
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Ensure this directory is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pipeline
import visualize
import transforms as tfm
from config import SAMPLE_RATE

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Voice Camouflage Detector — DSP Lab",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "KTU S5 ECE DSP Project — Voice Camouflage Detection System"},
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=Fira+Code:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
}

.main { background: #08090c; }

/* Metric card override */
[data-testid="metric-container"] {
    background: rgba(18, 22, 33, 0.8);
    border: 1px solid rgba(0, 242, 254, 0.15);
    border-radius: 12px;
    padding: 16px;
    transition: border-color 0.3s;
}
[data-testid="metric-container"]:hover {
    border-color: rgba(0, 242, 254, 0.35);
}

/* Verdict card */
.verdict-card {
    background: rgba(18, 22, 33, 0.9);
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    border: 2px solid rgba(0, 242, 254, 0.2);
    margin: 12px 0;
}
.verdict-natural  { border-color: rgba(0, 230, 118, 0.5) !important; box-shadow: 0 0 24px rgba(0,230,118,0.12); }
.verdict-ai       { border-color: rgba(255, 23, 68, 0.5) !important;  box-shadow: 0 0 24px rgba(255,23,68,0.12); }
.verdict-modified { border-color: rgba(213, 0, 249, 0.5) !important; box-shadow: 0 0 24px rgba(213,0,249,0.12); }

.verdict-label { font-size: 1.6rem; font-weight: 800; letter-spacing: 0.02em; }
.verdict-conf  { font-family: 'Fira Code', monospace; font-size: 0.95rem; color: #94a3b8; margin-top: 6px; }

/* Anomaly items */
.anomaly { font-family: 'Fira Code', monospace; font-size: 0.82rem;
           padding: 6px 10px; border-radius: 6px; margin-bottom: 4px;
           border-left: 3px solid; }
.anomaly-warning { background: rgba(253, 224, 71, 0.07); border-color: #fde047; color: #fde047; }
.anomaly-danger  { background: rgba(255, 23, 68, 0.07);  border-color: #ff1744; color: #fda4af; }
.anomaly-ok      { background: rgba(0, 230, 118, 0.07);  border-color: #00e676; color: #86efac; }

/* Section header badge */
.section-badge {
    display: inline-block;
    background: rgba(0, 242, 254, 0.12);
    border: 1px solid rgba(0, 242, 254, 0.3);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.72rem;
    font-family: 'Fira Code', monospace;
    color: #00f2fe;
    margin-bottom: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar — How This Works + Settings
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🛡️ Voice Camouflage Detector")
    st.caption("KTU S5 ECE — DSP Project")
    st.divider()

    with st.expander("ℹ️ How This Works", expanded=False):
        st.markdown("""
**5-Stage DSP Pipeline:**

1. **Pre-processing** — Butterworth HPF (80 Hz cutoff), spectral subtraction noise reduction, peak normalisation, Hamming windowing
2. **Transforms** — STFT spectrogram (FFT per frame)
3. **Feature Extraction** — MFCCs (Mel filterbank + DCT), Pitch/F0 (pYIN), HNR, ZCR, STE, Rolloff
4. **Classification** — Threshold rules + SVM + ExtraTrees
5. **Temporal Diarization** — 1.5s segment analysis for mixed audio

**Try these samples:**
- 🎙️ Your own voice recording
- 🤖 A TTS / AI voice clip (ElevenLabs, Google TTS…)
- 🎭 A voice-changer sample
        """)

    with st.expander("⚙️ Advanced Settings", expanded=False):
        show_raw = st.checkbox("Show Raw Value tables", value=True)
        show_theory = st.checkbox("Show DSP Theory notes", value=False)

    st.divider()
    st.caption("📚 KTU S5 DSP — Units 2, 3, 4, 5")
    st.caption("Built with librosa · scipy · scikit-learn · Streamlit · Plotly")


# ═══════════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="padding: 24px 0 8px 0;">
  <h1 style="font-size:2.2rem; font-weight:800; background:linear-gradient(135deg,#fff 30%,#00f2fe);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:0;">
    🛡️ VOICE CAMOUFLAGE DETECTOR
  </h1>
  <p style="color:#94a3b8; margin-top:6px; font-size:0.95rem;">
    DSP-Based Natural vs AI-Generated Voice Classification &nbsp;·&nbsp; KTU S5 ECE Project
  </p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Main Tabs
# ═══════════════════════════════════════════════════════════════════════════════

TAB_LABELS = [
    "📤 Upload / Record",
    "🔧 Preprocessing",
    "📊 Transforms",
    "🧬 Features",
    "🔬 Analysis",
    "🏁 Verdict",
    "🎓 Model Training",
]

tabs = st.tabs(TAB_LABELS)

# ─── Session state ────────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None


# ─────────────────────────────────────────────────────────────────────────────
# TAB 0: Upload / Record
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="section-badge">Step 1 — Audio Acquisition</div>', unsafe_allow_html=True)

    col_up, col_mic = st.columns(2, gap="large")

    with col_up:
        st.subheader("📁 File Upload")
        uploaded = st.file_uploader(
            "Drag & drop or browse",
            type=["wav", "mp3", "flac", "m4a", "mp4"],
            label_visibility="collapsed",
        )
        if uploaded:
            st.audio(uploaded, format=uploaded.type)
            if st.button("▶ Analyse File", type="primary", use_container_width=True):
                with st.spinner("Running DSP pipeline…"):
                    audio_bytes = uploaded.read()
                    res = pipeline.run_from_bytes(audio_bytes, filename=uploaded.name)
                    st.session_state.result = res
                st.success(f"✅ Analysis complete — {res.duration_s:.2f}s | {res.total_ms:.0f} ms total")
                st.balloons()

    with col_mic:
        st.subheader("🎙️ Microphone Recording")
        duration = st.slider("Recording duration (seconds)", 2, 15, 5)
        if st.button("⏺ Record from Microphone", use_container_width=True):
            with st.spinner(f"Recording {duration}s from mic…"):
                try:
                    y, sr = pipeline.audio_io.record_mic(duration_s=duration, sr=SAMPLE_RATE)
                    with st.spinner("Analysing recording…"):
                        res = pipeline.run(y, sr, filename="mic_recording.wav")
                        st.session_state.result = res
                    st.success(f"✅ Recording analysed — {res.total_ms:.0f} ms")
                except ImportError:
                    st.error("sounddevice not installed. Run: `pip install sounddevice`")
                except Exception as e:
                    st.error(f"Microphone error: {e}")

    # Quick status card (if result available)
    if st.session_state.result:
        r = st.session_state.result
        st.divider()
        _col_m = st.columns(4)
        _col_m[0].metric("Classification", r.label.split("(")[0].strip())
        _col_m[1].metric("Confidence", f"{r.confidence:.1f}%")
        _col_m[2].metric("Duration", f"{r.duration_s:.2f}s")
        _col_m[3].metric("Pipeline Time", f"{r.total_ms:.0f} ms")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="section-badge">Stage 1 — Pre-processing</div>', unsafe_allow_html=True)

    if not st.session_state.result:
        st.info("Upload or record audio in Tab 1 first.")
    else:
        r = st.session_state.result
        if show_theory:
            with st.expander("📖 DSP Theory: Pre-processing (KTU Unit 4)"):
                st.markdown("""
**High-Pass Filter (HPF):** Butterworth IIR filter designed via bilinear
transform. Removes DC offset and mains hum below 80 Hz.
Z-domain: H(z) = B(z)/A(z). Poles inside unit circle → BIBO stable.

**Spectral Subtraction:** Estimates noise PSD from quietest frames, subtracts
with over-subtraction factor α=1.2. Prevents musical noise via β-floor.

**Hamming Window:**  w[n] = 0.54 − 0.46·cos(2πn/(N-1))
Reduces spectral leakage by tapering frame edges.
                """)

        col_r, col_c = st.columns(2)
        with col_r:
            st.caption("Raw waveform")
            fig_raw = visualize.plot_waveform(
                np.array(r.waveform_raw), r.sample_rate, "Raw Signal")
            st.pyplot(fig_raw, use_container_width=True)

        with col_c:
            st.caption("After HPF + noise reduction + normalisation")
            fig_clean = visualize.plot_waveform(
                np.array(r.waveform_clean), r.sample_rate, "Cleaned Signal")
            st.pyplot(fig_clean, use_container_width=True)

        if show_raw:
            with st.expander("📋 Raw Waveform Values (downsampled to 4000 pts)"):
                df_w = pd.DataFrame({
                    "Sample Index": range(len(r.waveform_clean)),
                    "Raw Amplitude": r.waveform_raw,
                    "Clean Amplitude": r.waveform_clean,
                })
                st.dataframe(df_w, use_container_width=True, height=220)
                csv = df_w.to_csv(index=False).encode()
                st.download_button("⬇ Download CSV", csv, "waveform.csv", "text/csv")

        st.divider()
        st.subheader("🔬 Z-Transform: HPF Filter Analysis")

        col_pz, col_fr = st.columns(2)
        with col_pz:
            fig_pz = visualize.plot_pole_zero(r.sample_rate)
            st.pyplot(fig_pz, use_container_width=True)
        with col_fr:
            hpf = r.hpf_analysis
            if hpf:
                fig_fr = go.Figure()
                fig_fr.add_trace(go.Scatter(
                    x=list(hpf["freqs_hz"]),
                    y=list(hpf["H_magnitude_db"]),
                    line=dict(color="#4facfe", width=1.5),
                    name="|H(f)| dB",
                    hovertemplate="f=%{x:.1f}Hz  |H|=%{y:.1f}dB<extra></extra>",
                ))
                fig_fr.update_layout(
                    title="HPF Frequency Response",
                    xaxis_title="Frequency (Hz)", yaxis_title="|H| (dB)",
                    paper_bgcolor="#0f111a", plot_bgcolor="#0a0c14",
                    font=dict(color="#94a3b8"), height=280,
                    margin=dict(l=50, r=20, t=40, b=40),
                )
                st.plotly_chart(fig_fr, use_container_width=True)

        col_info = st.columns(3)
        col_info[0].metric("Frames", r.n_frames)
        col_info[1].metric("Frame len", "25 ms / 551 samples")
        col_info[2].metric("Hop len", "10 ms / 220 samples")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Transforms Comparison
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="section-badge">Stage 2 — Transform Analysis</div>', unsafe_allow_html=True)

    if not st.session_state.result:
        st.info("Upload or record audio in Tab 1 first.")
    else:
        r = st.session_state.result

        st.subheader("⚡ FFT vs DCT vs DST vs DFT Comparison")
        st.caption("Select a frame index to compare all four transforms on the same audio frame.")

        frame_idx = st.slider("Frame index", 0, max(0, r.n_frames - 1), 0)

        # Reconstruct the frame from the full clean signal
        from config import FRAME_DURATION_MS, HOP_DURATION_MS
        if r.waveform_clean_full is not None:
            y_full = np.array(r.waveform_clean_full, dtype=np.float32)
            fl = int(r.sample_rate * FRAME_DURATION_MS / 1000)
            hl = int(r.sample_rate * HOP_DURATION_MS / 1000)
            start = frame_idx * hl
            end = start + fl
            if end <= len(y_full):
                frame = y_full[start:end]
            else:
                frame = y_full[-fl:]
        else:
            frame = np.zeros(551, dtype=np.float32)

        with st.spinner("Running all 4 transforms…"):
            tc = tfm.compare_transforms(frame)

        col_t1, col_t2 = st.columns(2)
        col_t3, col_t4 = st.columns(2)

        palette = {"dft": "#ff1744", "fft": "#00f2fe", "dct": "#00e676", "dst": "#d500f9"}
        cols_map = {"dft": col_t1, "fft": col_t2, "dct": col_t3, "dst": col_t4}

        for name, col in cols_map.items():
            res_t = tc[name]
            mag = res_t["magnitude"]
            with col:
                st.markdown(f"**{name.upper()}** — `{res_t['elapsed_ms']:.4f} ms`")
                fig_t = go.Figure()
                fig_t.add_trace(go.Scatter(
                    y=list(mag[:len(mag)//2]),
                    line=dict(color=palette[name], width=1.0),
                    hovertemplate="Bin %{x}: %{y:.4f}<extra></extra>",
                ))
                fig_t.update_layout(
                    height=220,
                    paper_bgcolor="#0f111a", plot_bgcolor="#0a0c14",
                    margin=dict(l=40, r=10, t=10, b=30),
                    xaxis_title="Bin", yaxis_title="|X[k]|",
                    font=dict(color="#94a3b8", size=10),
                    showlegend=False,
                )
                st.plotly_chart(fig_t, use_container_width=True)

        # Timing comparison table
        st.divider()
        st.subheader("⏱ Timing Comparison")
        timing_data = {
            "Transform": ["DFT (direct O(N²))", "FFT (radix-2 O(N log N))", "DCT-II", "DST-II"],
            "Elapsed (ms)": [tc["dft"]["elapsed_ms"], tc["fft"]["elapsed_ms"],
                             tc["dct"]["elapsed_ms"], tc["dst"]["elapsed_ms"]],
            "Frame Size N": [tc["dft"]["n"]] * 4,
            "Complexity": ["O(N²)", "O(N log N)", "O(N log N)", "O(N log N)"],
        }
        df_t = pd.DataFrame(timing_data)
        st.dataframe(df_t, use_container_width=True)

        fft_speedup = tc["dft"]["elapsed_ms"] / max(tc["fft"]["elapsed_ms"], 0.0001)
        st.info(
            f"📐 **FFT is {fft_speedup:.0f}× faster than direct DFT** for N={tc['dft']['n']}. "
            f"Theoretical speedup = N / log₂N = {tc['dft']['n'] / np.log2(max(tc['dft']['n'],2)):.0f}×."
        )

        if show_theory:
            with st.expander("📖 DSP Theory: DFT, FFT, DCT, DST (KTU Units 2 & 3)"):
                st.markdown("""
**DFT** X[k] = Σ x[n]·e^{−j2πkn/N}  —  O(N²) — educational reference  
**FFT** Radix-2 DIT Cooley-Tukey — O(N log N) — exploits twiddle factor symmetry  
**DCT-II** C[k] = Σ x[n]·cos(π·k·(2n+1)/2N)  —  real-valued, energy compaction → MFCC  
**DST-II** D[k] = Σ x[n]·sin(π·k·(2n+1)/2N)  —  odd-symmetric extension  
                """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Features Dashboard
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="section-badge">Stage 2 — Feature Extraction</div>', unsafe_allow_html=True)

    if not st.session_state.result:
        st.info("Upload or record audio in Tab 1 first.")
    else:
        r = st.session_state.result

        # ── Metric cards ──────────────────────────────────────────────────
        cols_m = st.columns(5)
        ps = r.pitch_stats
        cols_m[0].metric("F0 Mean", f"{ps.get('f0_mean', 0):.1f} Hz", help="Mean fundamental frequency")
        cols_m[1].metric("Pitch Jitter", f"{ps.get('jitter_hz', 0):.2f} Hz", help="Frame-to-frame F0 variation")
        cols_m[2].metric("HNR", f"{r.hnr_db:.1f} dB", help="Harmonic-to-Noise Ratio")
        cols_m[3].metric("ZCR Mean", f"{r.zcr_mean:.4f}", help="Zero Crossing Rate")
        cols_m[4].metric("Spectral Rolloff", f"{r.rolloff_mean/1000:.2f} kHz", help="85th-percentile frequency")

        st.divider()

        # ── Spectrogram ───────────────────────────────────────────────────
        st.subheader("🌡️ Spectrogram (Interactive)")
        if r.spectrogram_db:
            spec_data = {"data": r.spectrogram_db,
                         "freqs": r.spectrogram_freqs,
                         "times": r.spectrogram_times}
            fig_spec = visualize.plot_spectrogram_plotly(spec_data)
            st.plotly_chart(fig_spec, use_container_width=True)
            if show_raw:
                with st.expander("📋 Spectrogram Raw Values"):
                    arr = np.array(r.spectrogram_db)
                    df_spec = pd.DataFrame(arr,
                        columns=[f"{t:.3f}s" for t in r.spectrogram_times],
                        index=[f"{f:.1f}Hz" for f in r.spectrogram_freqs])
                    st.dataframe(df_spec.head(20), use_container_width=True)
                    csv_s = df_spec.to_csv().encode()
                    st.download_button("⬇ Download Spectrogram CSV", csv_s, "spectrogram.csv")

        # ── MFCC Heatmap ──────────────────────────────────────────────────
        st.subheader("🎨 MFCC Coefficient Heatmap (Interactive)")
        if r.mfcc_matrix:
            mfcc_arr = np.array(r.mfcc_matrix)
            fig_mfcc = visualize.plot_mfcc_heatmap_plotly(mfcc_arr)
            st.plotly_chart(fig_mfcc, use_container_width=True)
            if show_raw:
                with st.expander("📋 MFCC Raw Values + Download"):
                    df_mfcc = pd.DataFrame(mfcc_arr,
                        index=[f"MFCC {i}" for i in range(mfcc_arr.shape[0])])
                    st.dataframe(df_mfcc, use_container_width=True, height=220)
                    csv_m = df_mfcc.to_csv().encode()
                    st.download_button("⬇ Download MFCC CSV", csv_m, "mfcc.csv")

        col_feat1, col_feat2 = st.columns(2)

        with col_feat1:
            st.subheader("🎵 Pitch Contour (F0)")
            fig_p = visualize.plot_pitch_contour(
                np.array(r.pitch_f0), np.array(r.pitch_times))
            st.pyplot(fig_p, use_container_width=True)
            if show_raw:
                with st.expander("📋 Pitch Statistics"):
                    st.json(r.pitch_stats)

        with col_feat2:
            st.subheader("📈 MFCC Means")
            if r.mfcc_means:
                fig_bar = visualize.plot_mfcc_means(r.mfcc_means)
                st.pyplot(fig_bar, use_container_width=True)
                if show_raw:
                    with st.expander("📋 MFCC Mean Values"):
                        df_means = pd.DataFrame({"Coefficient": [f"C{i}" for i in range(len(r.mfcc_means))],
                                                  "Mean": r.mfcc_means, "Std": r.mfcc_stds})
                        st.dataframe(df_means, use_container_width=True)

        if show_theory:
            with st.expander("📖 DSP Theory: MFCC Pipeline (KTU Unit 5)"):
                st.markdown("""
1. **FFT** per frame → power spectrum |X[k]|²  
2. **Mel filterbank** — 40 triangular filters on Mel scale  
3. **Log compression** → log Mel energies  
4. **DCT-II** → decorrelates → keep first 13 coefficients  
5. **Delta / Delta-Delta** → temporal dynamics  
Low MFCC Δ energy (< 0.015) → suspiciously smooth → AI synthesis artifact.
                """)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="section-badge">Stage 3 — Pattern Analysis</div>', unsafe_allow_html=True)

    if not st.session_state.result:
        st.info("Upload or record audio in Tab 1 first.")
    else:
        r = st.session_state.result

        # ── Qualitative analysis (numbers from PipelineResult, not canned strings) ──
        ps = r.pitch_stats
        baseline_f0_std = 18.0   # natural speech baseline
        baseline_jitter = 5.2

        f0_std_pct_diff = ((baseline_f0_std - ps.get("f0_std", baseline_f0_std))
                           / baseline_f0_std) * 100.0

        st.subheader("📝 Qualitative Analysis")
        qual_text = (
            f"Spectral envelope **MFCC Δ energy = {r.mfcc_delta_energy:.4f}** "
            f"({'significantly below' if r.mfcc_delta_energy < 0.015 else 'within'} natural baseline 0.015). "
            f"Pitch micro-variation σ = **{ps.get('f0_std', 0):.1f} Hz** "
            f"(natural baseline ≈ {baseline_f0_std:.1f} Hz, "
            f"{'suppressed' if ps.get('f0_std', 0) < NATURAL_F0_STD_MIN else 'normal'} — "
            f"{abs(f0_std_pct_diff):.0f}% {'below' if f0_std_pct_diff > 0 else 'above'} baseline). "
            f"HNR = **{r.hnr_db:.1f} dB** "
            f"({'unnaturally high → synthetic' if r.hnr_db > 40 else 'normal range'}). "
            f"Classified **{r.label}** with **{r.confidence:.1f}% confidence**."
        )
        st.info(qual_text)

        # ── Anomaly log ───────────────────────────────────────────────────
        st.subheader("🚨 Acoustic Anomaly Log")
        for anomaly in r.anomalies:
            low = anomaly.lower()
            if "anomaly" in low:
                cls = "anomaly-danger"
            elif "warning" in low:
                cls = "anomaly-warning"
            else:
                cls = "anomaly-ok"
            st.markdown(f'<div class="anomaly {cls}">▸ {anomaly}</div>',
                        unsafe_allow_html=True)

        # ── Segment Timeline ──────────────────────────────────────────────
        segs = r.segments
        if segs and segs.get("total_segments", 0) > 0:
            st.divider()
            st.subheader("🕐 Temporal Segment Diarization")
            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("Total Segments", segs["total_segments"])
            col_s2.metric("AI Synthetic", segs["syn_segments"],
                          delta=f"{r.ai_percentage:.1f}%", delta_color="inverse")
            col_s3.metric("Natural Human", segs["human_segments"],
                          delta=f"{r.human_percentage:.1f}%", delta_color="normal")

            timeline = segs.get("segment_timeline", [])
            if timeline:
                df_tl = pd.DataFrame(timeline)
                st.dataframe(df_tl, use_container_width=True)

        # ── ML prediction banner ──────────────────────────────────────────
        ml = r.ml_prediction
        if ml.get("available"):
            st.divider()
            st.subheader("🤖 ML Model Prediction")
            col_ml1, col_ml2, col_ml3 = st.columns(3)
            col_ml1.metric("Model", ml.get("model_name", "Unknown"))
            col_ml2.metric("AI Probability", f"{ml.get('ai_probability', 0):.1f}%")
            col_ml3.metric("Human Probability", f"{ml.get('human_probability', 0):.1f}%")

        # ── Benchmark ─────────────────────────────────────────────────────
        st.divider()
        st.subheader("⏱ Pipeline Benchmarks")
        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        col_b1.metric("Pre-processing", f"{r.prep_ms:.1f} ms")
        col_b2.metric("Feature Extraction", f"{r.features_ms:.1f} ms")
        col_b3.metric("Classification", f"{r.classifier_ms:.1f} ms")
        col_b4.metric("Total", f"{r.total_ms:.1f} ms")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: Final Verdict
# ─────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown('<div class="section-badge">Final Classification</div>', unsafe_allow_html=True)

    if not st.session_state.result:
        st.info("Upload or record audio in Tab 1 first.")
    else:
        r = st.session_state.result
        label = r.label

        if "Natural" in label:
            card_class = "verdict-natural"
            emoji = "✅"
            color = "#00e676"
        elif "Modified" in label:
            card_class = "verdict-modified"
            emoji = "⚠️"
            color = "#d500f9"
        else:
            card_class = "verdict-ai"
            emoji = "🚨"
            color = "#ff1744"

        st.markdown(f"""
<div class="verdict-card {card_class}">
  <div class="verdict-label" style="color:{color}">{emoji} {label}</div>
  <div class="verdict-conf">Confidence: {r.confidence:.1f}% &nbsp;|&nbsp; File: {r.filename}</div>
</div>
""", unsafe_allow_html=True)

        # Mixed audio bar
        if r.ai_percentage > 0:
            st.markdown("#### Audio Composition")
            col_bar = st.columns([r.ai_percentage / 100, r.human_percentage / 100])
            st.markdown(f"""
<div style="display:flex; height:12px; border-radius:6px; overflow:hidden; margin:8px 0 4px 0;">
  <div style="width:{r.ai_percentage}%; background:linear-gradient(90deg,#ff1744,#d500f9);"></div>
  <div style="width:{r.human_percentage}%; background:linear-gradient(90deg,#00e676,#00f2fe);"></div>
</div>
<div style="display:flex; justify-content:space-between; font-size:0.85rem;">
  <span style="color:#ff1744">🤖 AI: {r.ai_percentage:.1f}%</span>
  <span style="color:#00e676">🗣️ Human: {r.human_percentage:.1f}%</span>
</div>
""", unsafe_allow_html=True)

        st.divider()
        st.subheader("📊 Feature Summary")
        ps = r.pitch_stats
        summary_data = {
            "Feature": ["F0 Mean", "F0 Std (jitter σ)", "Pitch Jitter", "HNR", "2-4kHz HNR",
                        "MFCC Δ Energy", "ZCR Mean", "STE Mean", "Rolloff Mean"],
            "Value": [
                f"{ps.get('f0_mean', 0):.2f} Hz",
                f"{ps.get('f0_std', 0):.2f} Hz",
                f"{ps.get('jitter_hz', 0):.4f} Hz",
                f"{r.hnr_db:.2f} dB",
                f"{r.hnr_2_4khz_db:.2f} dB",
                f"{r.mfcc_delta_energy:.6f}",
                f"{r.zcr_mean:.5f}",
                f"{r.ste_mean:.6f}",
                f"{r.rolloff_mean:.1f} Hz",
            ],
            "Baseline (Natural)": [
                "100–250 Hz", "15–40 Hz", "2–8 Hz", "10–20 dB", ">5 dB",
                ">0.015", "0.02–0.15", "—", ">1800 Hz"
            ],
            "Flag if…": [
                "—", "<6 Hz = flat (AI)", "<0.8 Hz = robotic",
                ">40 dB = synthetic", "<3 dB = vocoder",
                "<0.015 = smooth", "—", "—", "<1800 Hz = narrow"
            ],
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

        # Audio player
        if r.original_wav_b64:
            st.divider()
            col_aud1, col_aud2 = st.columns(2)
            with col_aud1:
                st.caption("Original audio")
                st.audio(r.original_wav_b64, format="audio/wav")
            with col_aud2:
                st.caption("Noise-reduced audio")
                st.audio(r.clean_wav_b64, format="audio/wav")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 6: Model Training
# ─────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown('<div class="section-badge">SVM Model Training & Feedback</div>', unsafe_allow_html=True)
    st.subheader("🎓 Add Labelled Sample & Retrain SVM")

    st.markdown("""
Upload a labelled audio sample to improve the SVM classifier.
The model retains all previous feedback samples (stored in `model/feedback_data.json`).
Retraining triggers automatically when ≥4 samples exist.
    """)

    col_tr1, col_tr2 = st.columns([2, 1])
    with col_tr1:
        train_file = st.file_uploader(
            "Audio sample", type=["wav", "mp3", "flac", "m4a"],
            key="train_upload", label_visibility="visible")
    with col_tr2:
        ground_truth = st.selectbox(
            "Ground truth label",
            ["human", "ai", "synthetic", "mixed"],
            index=0,
        )

    if train_file and st.button("➕ Add Sample & Retrain", type="primary"):
        with st.spinner("Extracting features and updating model…"):
            try:
                audio_bytes = train_file.read()
                res = pipeline.run_from_bytes(audio_bytes, filename=train_file.name)
                feedback_result = pipeline.add_feedback(res._raw_features, ground_truth)
                if feedback_result.get("success"):
                    st.success(
                        f"✅ Sample added. Total samples: {feedback_result.get('n_samples')}. "
                        f"Status: {feedback_result.get('model_status')}"
                    )
                    if "accuracy" in feedback_result:
                        st.metric("Training Accuracy", f"{feedback_result['accuracy']:.1f}%")
                else:
                    st.error(f"Error: {feedback_result.get('error')}")
            except Exception as e:
                st.error(f"Training error: {e}")

    st.divider()
    st.subheader("📁 Feedback Data")
    from config import FEEDBACK_DATA_PATH
    fpath = os.path.join(os.path.dirname(__file__), FEEDBACK_DATA_PATH)
    if os.path.exists(fpath):
        with open(fpath) as f:
            data = json.load(f)
        label_counts = {"human": 0, "ai/synthetic/mixed": 0}
        for d in data:
            if d["label"] == 0:
                label_counts["human"] += 1
            else:
                label_counts["ai/synthetic/mixed"] += 1
        col_d1, col_d2 = st.columns(2)
        col_d1.metric("Human Samples", label_counts["human"])
        col_d2.metric("AI/Synthetic Samples", label_counts["ai/synthetic/mixed"])
        st.caption(f"Stored at: `{fpath}`")
    else:
        st.info("No feedback data yet. Add labelled samples above.")
