# Prompt for Claude Code

Copy everything below into Claude Code as your project instructions.

---

Build a **DSP-based Voice Camouflage Detection System** — a system that classifies a voice recording as **Natural Human Voice** or **AI-Generated / Modified Voice** (not speaker identification), for a KTU S5 ECE Digital Signal Processing course project. Wrap it in a **Streamlit** web app.

## Hard requirements

1. **Split the code into separate, single-purpose Python files** — never put everything in one giant script. Each file should be independently testable/runnable so I can debug one DSP stage at a time. Use this structure:

```
voice_camouflage_detection/
├── app.py                      # Streamlit UI only — imports and calls the modules below, no DSP logic here
├── audio_io.py                 # Recording / file upload / loading audio as a numpy array
├── preprocessing.py            # Noise removal, normalization, framing/windowing
├── transforms.py               # FFT, DFT, DCT, DST (sine transform), Z-transform utilities
├── features.py                 # MFCC, pitch/F0 tracking, HNR, short-time energy, ZCR — built on top of transforms.py
├── classifier.py                # Threshold/statistical rules + optional lightweight ML classifier (SVM)
├── visualize.py                 # Waveform, spectrum, spectrogram, feature plots (matplotlib, returned as figures for Streamlit)
├── pipeline.py                  # Orchestrates the full flow: audio -> preprocessing -> transforms -> features -> classification
├── config.py                    # Constants: sample rate, frame size, hop size, thresholds, etc.
├── utils.py                     # Small shared helpers
└── tests/
    ├── test_transforms.py       # Unit tests comparing your DFT/FFT/DCT/DST/Z-transform against numpy/scipy reference outputs
    ├── test_preprocessing.py
    └── test_features.py
```

2. **Follow the KTU S5 DSP syllabus explicitly.** For each transform, implement it in a way that shows the underlying DSP theory rather than just calling a black-box library function once and moving on:
   - **DFT** — implement the direct O(N²) definition once (`dft_direct`) for teaching/comparison purposes, AND use `numpy.fft` for actual performance-critical calls. Comment clearly which is which.
   - **FFT** — use `numpy.fft.fft`, but add a docstring/comment explaining radix-2 decimation-in-time and why FFT is O(N log N) vs DFT's O(N²).
   - **DCT** — use `scipy.fft.dct` (Type-II, the standard one used in MFCC and compression). Explain in comments why DCT is preferred over DFT for compacting energy into fewer coefficients (real-valued, no phase).
   - **DST (sine transform)** — implement using `scipy.fft.dst` or manually via the sine-basis formula. Include it as an alternative feature/comparison transform, not necessarily in the main pipeline.
   - **Z-transform** — implement a small educational module showing how the filters used in preprocessing (e.g., a bandpass/noise filter) are represented by their Z-domain transfer function `H(z)`, and plot the pole-zero diagram using `scipy.signal.tf2zpk` and `scipy.signal.freqz` for frequency response. This ties the filter design stage back to Z-transform theory.
   - Add a short **`docs/theory_notes.md`** file (or docstring block) mapping each module back to the specific KTU S5 DSP syllabus topic it demonstrates (e.g., "Unit 2: DFT properties", "Unit 3: FFT algorithms", "Unit 4: Filter design / Z-transform", "Unit 5: Applications").

3. **Preprocessing pipeline (`preprocessing.py`)**:
   - Noise removal (simple spectral subtraction or a bandpass filter designed via Z-transform / `scipy.signal`)
   - Amplitude normalization
   - Framing into 20–30 ms frames with overlap (windowing with Hamming/Hanning)
   - Each step as its own small function with a clear docstring

4. **Feature extraction (`features.py`)**, each computed per frame and aggregated:
   - FFT-based spectral analysis / spectrogram
   - MFCCs (built from Mel filterbank + DCT — show this composition explicitly rather than calling one `librosa.feature.mfcc()` line with no explanation)
   - Pitch/F0 tracking (autocorrelation method is fine and is DSP-syllabus-appropriate)
   - Harmonic-to-Noise Ratio (HNR)
   - Short-time energy
   - Zero Crossing Rate (ZCR)

5. **Transform comparison mode**: add a Streamlit toggle/section that lets me compare FFT vs DCT vs DST vs direct DFT on the same audio frame side-by-side (plots + timing), since my professor specifically wants this comparison.

6. **Classification (`classifier.py`)**:
   - Start with simple threshold/statistical rules comparing extracted features against natural-speech baselines (over-smooth spectral envelope, suppressed pitch micro-variation, vocoder artifacts → flag as AI-generated)
   - Add an optional lightweight SVM (scikit-learn) trained on a small labeled dataset, with the rule-based approach as a fallback/baseline if no trained model is available
   - Output: classification label + confidence score

7. **Streamlit app (`app.py`)**:
   - Upload audio file OR record via microphone
   - Live, per-stage visualization as the pipeline runs (waveform → filtered waveform → spectrogram → extracted features → final classification with confidence)
   - A "Compare Transforms" tab (see point 5)
   - A "Model Training" tab to retrain the SVM on new labeled samples if I add them later

8. **Code style**:
   - Every function has a short docstring explaining what it does and, where relevant, which DSP concept it demonstrates
   - Type hints on function signatures
   - No single function longer than ~40 lines — break complex logic into smaller helpers
   - Avoid unnecessary abstraction/classes where plain functions are clearer — this needs to be readable by me, an undergrad, not enterprise-grade architecture
   - Include a `requirements.txt`

9. **Deliverables**: after writing the code, give me a short README explaining how to run `app.py`, how to run the unit tests, and a one-paragraph summary per module of what DSP concept it demonstrates (for my project report).

## Frontend / UI requirements (premium, data-transparent interface)

The Streamlit front end must not just show a final label — it must expose the actual numbers the backend computed, so every plot and score is independently verifiable against the raw values.

1. **Every visualization is driven directly by backend return values, never re-estimated or hardcoded in `app.py`.**
   - `pipeline.py` should return a single structured result object (e.g. a `dataclass` or `dict`) containing every intermediate array/value: raw waveform, filtered waveform, frame boundaries, FFT/DFT/DCT/DST arrays, MFCC matrix, pitch contour, HNR, short-time energy, ZCR, per-feature baseline comparison, and final classification + confidence.
   - `app.py` only reads from this result object to render plots/tables — it must contain no signal-processing math itself. This is what makes the frontend numbers "provably" the same ones the DSP code produced.
   - Add a **"Raw Values" expander/tab** next to every chart showing the underlying numeric array (or a truncated table + a "Download as CSV" button) so I can manually verify a chart matches the numbers.

2. **Heatmaps** (use `plotly` for these instead of static matplotlib, so they're interactive with hover tooltips showing exact values):
   - Spectrogram heatmap (time vs frequency vs magnitude in dB)
   - MFCC heatmap (time vs MFCC coefficient index vs value) — this is the one my professor will look at closely, so make sure axis labels, colorbar, and coefficient indices are clearly labeled
   - Optional: DCT/DST coefficient heatmaps in the transform-comparison tab

3. **Qualitative analysis panel**: below the plots, auto-generate a short plain-English interpretation driven by the actual computed values (not a canned string) — e.g. "Spectral envelope is 34% smoother than the natural-speech baseline; pitch micro-variation is suppressed (σ = 2.1 Hz vs baseline 8.4 Hz); classified AI-Generated with 87% confidence." Pull every number in that sentence from the result object via an f-string, so the text and the charts can never disagree.

4. **Layout / visual design** — this needs to look premium and convincing, not like a default Streamlit demo:
   - Use `st.set_page_config(layout="wide")`, a custom theme (`.streamlit/config.toml` with a deliberate color palette — dark, technical, lab-instrument feel fits a DSP forensics tool well), and custom CSS injected via `st.markdown(..., unsafe_allow_html=True)` for card-style metric panels, not raw `st.write` dumps.
   - Use `st.metric` cards for the headline numbers (confidence score, classification, HNR, ZCR, energy) at the top of the page, styled cards/columns for feature comparisons.
   - Organize the page as: **Upload/Record → Preprocessing preview → Transform comparison → Feature dashboard (heatmaps + metrics) → Qualitative analysis → Final verdict**, using `st.tabs` or a sidebar step-navigation rather than one long scroll.
   - Add subtle loading spinners/progress indicators per pipeline stage so it feels responsive on longer recordings.
   - Add a small "How this works" collapsible sidebar section suggesting next steps to the user (e.g. "try a voice-changer sample to compare", "adjust the frame size in the sidebar and re-run") — genuinely useful suggestions grounded in what the tool does, not filler text.

Ask me clarifying questions only if something above is ambiguous (e.g. dataset availability for the SVM) — otherwise proceed and build it.
