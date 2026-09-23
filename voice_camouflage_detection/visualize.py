"""
visualize.py — Matplotlib/Plotly figure factories
===================================================
KTU S5 DSP Project: Voice Camouflage Detection

All functions return a Figure object (matplotlib or plotly).
They do NOT call plt.show() or st.pyplot() — that is app.py's job.
This keeps visualization logic independently testable and reusable.

Usage:
    fig = plot_waveform(y, sr)
    st.pyplot(fig)          # in app.py
    plt.savefig("out.png")  # in tests
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for Streamlit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from scipy.signal import sos2tf

import plotly.graph_objects as go

from config import SAMPLE_RATE
from preprocessing import design_high_pass, get_zpk, get_frequency_response
import transforms


# ─── Colour palette (matches the dark theme) ────────────────────────────────
C_CYAN = "#00f2fe"
C_BLUE = "#4facfe"
C_GREEN = "#00e676"
C_RED = "#ff1744"
C_MAGENTA = "#d500f9"
C_BG = "#0f111a"
C_GRID = "#1e293b"

_DARK_PARAMS = {
    "figure.facecolor": C_BG,
    "axes.facecolor": "#0a0c14",
    "axes.edgecolor": "#1e293b",
    "axes.labelcolor": "#94a3b8",
    "xtick.color": "#64748b",
    "ytick.color": "#64748b",
    "text.color": "#f8fafc",
    "grid.color": C_GRID,
    "grid.alpha": 0.5,
}


def _dark_fig(figsize=(10, 3)) -> Figure:
    """Create a dark-themed matplotlib Figure."""
    with plt.rc_context(_DARK_PARAMS):
        fig = plt.figure(figsize=figsize, facecolor=C_BG)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Waveform (time-domain)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_waveform(y: np.ndarray, sr: int = SAMPLE_RATE,
                 title: str = "Waveform") -> Figure:
    """Plot raw or cleaned audio waveform (amplitude vs time).

    Args:
        y:     Audio samples (float32).
        sr:    Sample rate.
        title: Plot title.

    Returns:
        Matplotlib Figure.
    """
    t = np.linspace(0, len(y) / sr, len(y))
    with plt.rc_context(_DARK_PARAMS):
        fig, ax = plt.subplots(figsize=(10, 3), facecolor=C_BG)
        ax.plot(t, y, color=C_CYAN, linewidth=0.6, alpha=0.9)
        ax.axhline(0, color=C_GRID, linewidth=0.5, linestyle="--")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.set_title(title, color=C_CYAN, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, len(y) / sr)
        ax.set_ylim(-1.1, 1.1)
        fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Spectrogram (Plotly interactive heatmap)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_spectrogram_plotly(spec_data: dict) -> go.Figure:
    """Interactive spectrogram heatmap using Plotly.

    Hovering shows exact time, frequency, and dB value — required by the spec.

    Args:
        spec_data: dict with 'data' (2-D list dB), 'freqs' (Hz list), 'times' (s list).

    Returns:
        Plotly Figure.
    """
    data_arr = np.array(spec_data["data"])
    freqs = spec_data["freqs"]
    times = spec_data["times"]

    fig = go.Figure(data=go.Heatmap(
        z=data_arr,
        x=times,
        y=freqs,
        colorscale="Viridis",
        colorbar=dict(title="dB", tickfont=dict(color="white")),
        hovertemplate="Time: %{x:.3f}s<br>Freq: %{y:.1f}Hz<br>Mag: %{z:.1f}dB<extra></extra>",
    ))
    fig.update_layout(
        title="Spectrogram (STFT Magnitude)", title_font_color=C_CYAN,
        xaxis_title="Time (s)", yaxis_title="Frequency (Hz)",
        paper_bgcolor=C_BG, plot_bgcolor="#0a0c14",
        font=dict(color="#94a3b8"),
        height=320,
        margin=dict(l=60, r=20, t=40, b=40),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MFCC Heatmap (Plotly interactive)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_mfcc_heatmap_plotly(mfcc_matrix: np.ndarray) -> go.Figure:
    """Interactive MFCC coefficient heatmap.

    Axes: time frames (x) × coefficient index (y).
    Colorbar labelled clearly for professor inspection.

    Args:
        mfcc_matrix: 2-D array (n_mfcc × n_frames).

    Returns:
        Plotly Figure.
    """
    n_mfcc, n_frames = mfcc_matrix.shape
    fig = go.Figure(data=go.Heatmap(
        z=mfcc_matrix,
        x=list(range(n_frames)),
        y=[f"MFCC {i}" for i in range(n_mfcc)],
        colorscale="RdBu",
        zmid=0,
        colorbar=dict(title="Coeff. Value", tickfont=dict(color="white")),
        hovertemplate="Frame: %{x}<br>%{y}<br>Value: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="MFCC Coefficient Matrix", title_font_color=C_CYAN,
        xaxis_title="Frame Index", yaxis_title="MFCC Coefficient",
        paper_bgcolor=C_BG, plot_bgcolor="#0a0c14",
        font=dict(color="#94a3b8"),
        height=320,
        margin=dict(l=80, r=20, t=40, b=40),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Pitch Contour
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pitch_contour(f0: np.ndarray, times: np.ndarray,
                       title: str = "Pitch Contour (F0)") -> Figure:
    """Plot F0 contour over time. Voiced frames shown in cyan; unvoiced gaps left blank.

    Args:
        f0:    F0 array (0 = unvoiced).
        times: Time axis (s).
        title: Plot title.

    Returns:
        Matplotlib Figure.
    """
    with plt.rc_context(_DARK_PARAMS):
        fig, ax = plt.subplots(figsize=(10, 3), facecolor=C_BG)
        voiced = f0 > 0
        ax.scatter(times[voiced], f0[voiced],
                   color=C_CYAN, s=4, alpha=0.8, linewidths=0)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("F0 (Hz)")
        ax.set_title(title, color=C_CYAN, fontsize=11)
        ax.grid(True, alpha=0.3)
        if np.any(voiced):
            ax.set_ylim(max(0, f0[voiced].min() - 20), f0[voiced].max() + 20)
        fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Transform Comparison (FFT vs DCT vs DST vs DFT)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_transform_comparison(frame: np.ndarray) -> Figure:
    """Four-panel comparison of DFT, FFT, DCT, DST magnitudes on the same frame.

    Also displays elapsed time for each to demonstrate O(N²) vs O(N log N).

    Args:
        frame: Single audio frame (1-D float array).

    Returns:
        Matplotlib Figure (2×2 grid).
    """
    result = transforms.compare_transforms(frame)
    names = list(result.keys())   # dft, fft, dct, dst
    colors = [C_RED, C_CYAN, C_GREEN, C_MAGENTA]

    with plt.rc_context(_DARK_PARAMS):
        fig, axes = plt.subplots(2, 2, figsize=(11, 6), facecolor=C_BG)
        fig.suptitle("Transform Comparison on Single Audio Frame",
                     color=C_CYAN, fontsize=12, fontweight="bold")

        for ax, name, color in zip(axes.flat, names, colors):
            mag = result[name]["magnitude"]
            n = len(mag)
            ax.plot(mag[:n // 2], color=color, linewidth=0.9)
            ms = result[name]["elapsed_ms"]
            ax.set_title(f"{name.upper()}  ({ms:.4f} ms)", color=color, fontsize=10)
            ax.set_xlabel("Bin")
            ax.set_ylabel("|X[k]|")
            ax.grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Pole-Zero Diagram + Frequency Response
# ═══════════════════════════════════════════════════════════════════════════════

def plot_pole_zero(sr: int = SAMPLE_RATE) -> Figure:
    """Plot HPF pole-zero diagram and frequency response (|H(e^jω)|).

    Demonstrates Z-transform theory: poles inside unit circle = stable filter.

    Args:
        sr: Sample rate (Hz) — used for HPF design and freq axis.

    Returns:
        Matplotlib Figure with two subplots.
    """
    z, p, k = get_zpk(sr)
    freqs, H = get_frequency_response(sr)
    H_db = 20 * np.log10(np.abs(H) + 1e-12)

    with plt.rc_context(_DARK_PARAMS):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=C_BG)

        # — Pole-Zero plot —
        theta = np.linspace(0, 2 * np.pi, 360)
        ax1.plot(np.cos(theta), np.sin(theta), color=C_GRID, linewidth=0.8)
        ax1.axhline(0, color=C_GRID, linewidth=0.5)
        ax1.axvline(0, color=C_GRID, linewidth=0.5)
        ax1.scatter(z.real, z.imag, marker="o", s=80, color=C_CYAN,
                    label="Zeros", zorder=5, edgecolors="white", linewidths=0.5)
        ax1.scatter(p.real, p.imag, marker="x", s=120, color=C_RED,
                    label="Poles", zorder=5, linewidths=2)
        ax1.set_xlim(-1.5, 1.5)
        ax1.set_ylim(-1.5, 1.5)
        ax1.set_aspect("equal")
        ax1.set_title("Pole-Zero Diagram (HPF)", color=C_CYAN, fontsize=10)
        ax1.set_xlabel("Real(z)")
        ax1.set_ylabel("Imag(z)")
        ax1.legend(fontsize=8, facecolor="#0a0c14", labelcolor="white")
        ax1.grid(True, alpha=0.3)

        # — Frequency Response —
        ax2.plot(freqs, H_db, color=C_BLUE, linewidth=1.2)
        ax2.axvline(80, color=C_RED, linewidth=0.8, linestyle="--",
                    label="Cutoff 80 Hz")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("|H(e^jω)| (dB)")
        ax2.set_title("Frequency Response H(z)", color=C_CYAN, fontsize=10)
        ax2.legend(fontsize=8, facecolor="#0a0c14", labelcolor="white")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-80, 5)

        fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Feature Bar Chart (MFCC means)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_mfcc_means(mfcc_means: list, title: str = "MFCC Means") -> Figure:
    """Bar chart of MFCC mean coefficients.

    Args:
        mfcc_means: List of 13 float values.
        title:      Plot title.

    Returns:
        Matplotlib Figure.
    """
    with plt.rc_context(_DARK_PARAMS):
        fig, ax = plt.subplots(figsize=(7, 3), facecolor=C_BG)
        x = np.arange(len(mfcc_means))
        colors_bar = [C_CYAN if v >= 0 else C_MAGENTA for v in mfcc_means]
        ax.bar(x, mfcc_means, color=colors_bar, width=0.7, alpha=0.85)
        ax.axhline(0, color=C_GRID, linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"C{i}" for i in range(len(mfcc_means))],
                           fontsize=8)
        ax.set_xlabel("MFCC Coefficient Index")
        ax.set_ylabel("Mean Value")
        ax.set_title(title, color=C_CYAN, fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
    return fig


# ─── Self-test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sr = SAMPLE_RATE
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    y = 0.5 * np.sin(2 * np.pi * 300 * t)

    fig_w = plot_waveform(y, sr)
    fig_w.savefig("_test_waveform.png")

    frame = y[:512]
    fig_tc = plot_transform_comparison(frame)
    fig_tc.savefig("_test_transforms.png")

    fig_pz = plot_pole_zero(sr)
    fig_pz.savefig("_test_polezero.png")

    plt.close("all")
    print("visualize.py self-test passed. Check _test_*.png files.")
