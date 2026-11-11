# -*- coding: utf-8 -*-
"""Shared utilities: loaders, features, phase metrics, evaluation, plotting."""
from __future__ import annotations

import os
import re

import h5py
import numpy as np
from scipy import signal as sp_signal
from sklearn.metrics import roc_auc_score, average_precision_score


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_RAW = os.path.join(ROOT, "data", "raw")


# ------------------------------------------------------------------ loading

def load_mat(path: str) -> dict:
    """Read a .mat file (v5 or v7.3/HDF5), returning a plain dict."""
    try:
        from scipy.io import loadmat
        out = loadmat(path)
    except NotImplementedError:
        out = _hdf5_to_dict(h5py.File(path, "r"))
    return {k: v for k, v in out.items() if not k.startswith("__")}


def _hdf5_to_dict(node) -> object:
    if isinstance(node, h5py.Dataset):
        return np.asarray(node[()])
    out = {}
    for k, v in node.items():
        out[k] = _hdf5_to_dict(v)
    return out


def find_data_files(subdir: str, ext: tuple = (".mat", ".edf")) -> list[str]:
    """Recursively find data files under a dataset directory."""
    base = os.path.join(DATA_RAW, subdir)
    if not os.path.isdir(base):
        return []
    hits = []
    for root, _dirs, files in os.walk(base):
        for f in sorted(files):
            if f.lower().endswith(ext):
                hits.append(os.path.join(root, f))
    return hits


def parse_chb_summary(path: str) -> list[tuple[str, float, float]]:
    """Parse the CHB-MIT seizure summary into (file, start_s, end_s)."""
    events = []
    cur_file = None
    pending_start = None
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("File Name:"):
                cur_file = line.split(":", 1)[1].strip()
            elif "Seizure Start Time" in line:
                m = re.search(r"([\d.]+)\s*seconds", line)
                pending_start = float(m.group(1)) if m else None
            elif "Seizure End Time" in line:
                m = re.search(r"([\d.]+)\s*seconds", line)
                if m and cur_file and pending_start is not None:
                    events.append((cur_file, pending_start, float(m.group(1))))
                pending_start = None
    return events


# ------------------------------------------------------------- preprocessing

def notch_filter(x: np.ndarray, fs: float, f0: float = 50.0,
                 q: float = 30.0) -> np.ndarray:
    """Zero-phase IIR notch (per channel or vector)."""
    x = np.asarray(x, dtype=float)
    b, a = sp_signal.iirnotch(f0, q, fs)
    if x.ndim == 1:
        return sp_signal.filtfilt(b, a, x)
    return np.stack([sp_signal.filtfilt(b, a, x[i]) for i in range(x.shape[0])],
                    axis=0)


def bandpass(x: np.ndarray, fs: float, low: float, high: float,
             order: int = 4) -> np.ndarray:
    """Zero-phase band-pass with a Nyquist check."""
    x = np.asarray(x, dtype=float)
    nyq = fs / 2.0
    if low <= 0 or high >= nyq:
        raise ValueError(
            f"bandpass({low}-{high} Hz) exceeds Nyquist {nyq:.1f} Hz (fs={fs})")
    b, a = sp_signal.butter(order, [low / nyq, high / nyq], btype="band")
    if x.ndim == 1:
        return sp_signal.filtfilt(b, a, x)
    return np.stack([sp_signal.filtfilt(b, a, x[i]) for i in range(x.shape[0])],
                    axis=0)


def car_reference(x: np.ndarray, bad_mask: np.ndarray | None = None) -> np.ndarray:
    """Common-average reference; bad channels are excluded and zeroed."""
    x = np.asarray(x, dtype=float)
    if bad_mask is None:
        bad_mask = np.zeros(x.shape[0], dtype=bool)
    good = ~np.asarray(bad_mask)
    mean = x[good].mean(axis=0, keepdims=True)
    out = x.copy()
    out[good] = x[good] - mean
    out[~good] = 0.0
    return out


def bipolar_reference(x: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    """Bipolar re-reference: out[i] = x[pair[i][0]] - x[pair[i][1]]."""
    x = np.asarray(x, dtype=float)
    return np.stack([x[a] - x[b] for a, b in pairs], axis=0)


# ------------------------------------------------------------------ features

def band_power(x: np.ndarray, fs: float, band: tuple[float, float]) -> np.ndarray:
    """Mean squared analytic envelope in a band (one value per channel)."""
    xb = bandpass(x, fs, band[0], band[1])
    env = np.abs(sp_signal.hilbert(xb, axis=-1)) ** 2
    return env.mean(axis=-1)


def line_length(x: np.ndarray) -> np.ndarray:
    """Per-channel line length."""
    x = np.asarray(x, dtype=float)
    return np.abs(np.diff(x, axis=-1)).sum(axis=-1)


def analytic_phase(x: np.ndarray) -> np.ndarray:
    """Instantaneous phase of the analytic signal (per channel)."""
    return np.angle(sp_signal.hilbert(np.asarray(x, dtype=float), axis=-1))


def plv(x: np.ndarray, y: np.ndarray) -> float:
    """Phase-locking value between two signals."""
    dphi = analytic_phase(x) - analytic_phase(y)
    return float(np.abs(np.exp(1j * dphi).mean()))


def plv_matrix(x: np.ndarray) -> np.ndarray:
    """Channel-by-channel PLV matrix."""
    n = x.shape[0]
    m = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = m[j, i] = plv(x[i], x[j])
    return m


def pac_modulation_index(low: np.ndarray, high: np.ndarray, fs: float,
                         low_band=(4, 12), high_band=(30, 80),
                         n_bins: int = 18) -> float:
    """Tort-style phase-amplitude coupling modulation index (simplified)."""
    lp = bandpass(low, fs, *low_band)
    hp = bandpass(high, fs, *high_band)
    phase = analytic_phase(lp)
    amp = np.abs(sp_signal.hilbert(hp))
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    idx = np.clip(np.digitize(phase, bins) - 1, 0, n_bins - 1)
    amp_mean = np.array([amp[idx == b].mean() if np.any(idx == b) else 0.0
                         for b in range(n_bins)])
    total = amp_mean.sum()
    if total <= 0:
        return 0.0
    p = amp_mean / total
    p = p[p > 0]
    kl = float((p * np.log(p * n_bins)).sum())
    return min(1.0, kl / np.log(n_bins))


def detect_ripples(x: np.ndarray, fs: float, band=(80, 250),
                   z_thresh: float = 3.0, min_dur: float = 0.01,
                   min_gap: float = 0.005) -> np.ndarray:
    """Simple ripple detector: band envelope z-score, merge, min duration."""
    xb = bandpass(x, fs, *band)
    env = np.abs(sp_signal.hilbert(xb))
    z = (env - env.mean()) / (env.std() + 1e-12)
    above = z > z_thresh
    min_len = int(min_dur * fs)
    min_gap_len = int(min_gap * fs)
    runs = []
    i = 0
    n = len(above)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            runs.append([i, j])
            i = j
        else:
            i += 1
    merged = []
    for s, e in runs:
        if merged and (s - merged[-1][1]) < min_gap_len:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    events = [(s, e) for s, e in merged if (e - s) >= min_len]
    return np.asarray(events, dtype=int).reshape(-1, 2) if events else np.zeros((0, 2), dtype=int)


def sliding_windows(x: np.ndarray, win: int, step: int) -> np.ndarray:
    """Cut (n_ch, n_t) into (n_win, n_ch, win); the last partial window is dropped."""
    x = np.asarray(x, dtype=float)
    n_ch, n_t = x.shape
    starts = np.arange(0, n_t - win + 1, step)
    return np.stack([x[:, s:s + win] for s in starts], axis=0)


# ---------------------------------------------------------------- evaluation

def chronological_split(n: int, test_frac: float = 0.2, val_frac: float = 0.1):
    """Chronological train | validation | test split (no shuffling)."""
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    idx = np.arange(n)
    test = idx[-n_test:]
    val = idx[-n_test - n_val:-n_test]
    train = idx[:-n_test - n_val]
    return train, val, test


def auc_pr(y_true: np.ndarray, y_score: np.ndarray):
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "auc_pr": float(average_precision_score(y_true, y_score)),
    }


# -------------------------------------------------------------------- plotting

def setup_matplotlib_cn() -> None:
    """Configure CJK fonts before plotting (optional)."""
    import matplotlib
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "DengXian", "Arial Unicode MS", "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False


def plot_segment(ax, x: np.ndarray, fs: float, t0: float = 0.0, n_ch: int = 8,
                 gain: float = 1.0, color: str = "k", lw: float = 0.5):
    """Overlay several channels with an offset (butterfly plot)."""
    x = np.asarray(x, dtype=float)
    n = min(x.shape[0], n_ch)
    t = t0 + np.arange(x.shape[1]) / fs
    offset = np.arange(n)[::-1] * gain * np.std(x) * 8
    for i in range(n):
        ax.plot(t, x[i] + offset[i], color=color, lw=lw)
    ax.set_yticks(offset)
    ax.set_yticklabels([f"ch{i}" for i in range(n)])
    ax.set_xlabel("time (s)")


def plot_psd(ax, x: np.ndarray, fs: float, nperseg: int | None = None):
    """Per-channel PSD with the channel mean in black."""
    x = np.asarray(x, dtype=float)
    if nperseg is None:
        nperseg = min(x.shape[1], int(fs * 2))
    f, pxx = sp_signal.welch(x, fs=fs, nperseg=nperseg, axis=-1)
    ax.semilogy(f, pxx.T, color="tab:blue", alpha=0.35, lw=0.8)
    ax.semilogy(f, pxx.mean(axis=0), color="k", lw=1.5, label="mean")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD (V^2/Hz)")
    ax.legend()
