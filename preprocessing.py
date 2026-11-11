# -*- coding: utf-8 -*-
"""iEEG preprocessing protocol v3: swappable methods with an artifact-safe
ordering, a reproducible parameter archive, and a nine-panel QC dashboard.

Fixed order (ringing- and step-safe):
  0 DC removal -> 1 bad channels -> 2 stimulation artifacts (before any
  temporal filter) -> 3 frequency filtering -> 4 line noise (base + harmonics
  + custom frequencies) -> 5 motion / EMG masks -> 6 re-referencing (9
  schemes) -> 7 IED detection (optional) -> 8 resampling -> 9 epoching ->
  10 epoch QC.

Methods per step (config key -> options):
  0 dc_method: none | mean | median
  1 bad_detect: mad | ptp | flatness | corr | ica, '+' union, 'vote:' majority
     bad_handle: none | mask | interpolate | drop (drop remaps pairs, groups,
     white-matter contacts and adjacency automatically)
  2 stim: none | template_subtract | interpolate | blank
     (stim_pre / stim_post in seconds)
  3 filter_method: none | highpass (hp_cutoff) | lowpass (lp_cutoff) |
     bandpass | detrend (Hann-weighted 50% overlap, no boundary steps)
  4 line_method: none | iir_notch | fir_notch | cleanline | fft_interp
     (line_f0 / line_q / line_width / line_win / line_harmonics /
      line_freqs: arbitrary device-noise list)
  5 motion: none | mask | interpolate (MAD-robust z, propagated to epoch QC)
     emg: none | mask (60-180 Hz broadband muscle artifacts)
  6 reference: none | car | median | group_car | shaft_car | white_matter |
     bipolar | laplacian | laplacian8
  7 ied: none | report | exclude (envelope z + amplitude + derivative
     steepness tri-criterion)
  8 resample_to: None or target rate (polyphase, event coordinates synced)
  9 events / tmin / tmax / baseline / cond_labels / evoked_sub
     (evoked_sub: none | global | per_condition)
  10 ptp_max / qc_var_z: epoch QC

Conventions: x is (n_ch, n_t), time along the last axis.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sp_signal


# ============================================================== demo data

def generate_demo_lfp(seed: int = 0, n_ch: int = 32, fs: float = 512.0,
                      duration: float = 120.0) -> dict:
    """Synthetic 32-channel LFP with realistic artifacts for protocol tests."""
    rng = np.random.default_rng(seed)
    n_t = int(duration * fs)
    t = np.arange(n_t) / fs

    noise = rng.standard_normal(n_t)
    spec = np.fft.rfft(noise, n_t)
    f = np.fft.rfftfreq(n_t, 1 / fs)
    pink = np.fft.irfft(spec * (1.0 / (1.0 + f) ** 1.2), n_t)
    pink = pink / pink.std()

    phase_grad = np.linspace(0, 6 * np.pi, n_ch)
    theta = np.sin(2 * np.pi * 6 * t[None, :] + phase_grad[:, None]) * 30
    X = theta + 12 * pink[None, :]

    for _ in range(10):
        onset = int(rng.integers(0, n_t - int(0.5 * fs)))
        dur = int(rng.uniform(0.2, 0.5) * fs)
        env = np.hanning(2 * dur)[dur:]
        gamma = np.sin(2 * np.pi * 45 * t[:dur]) * env * rng.uniform(20, 60)
        w = np.clip(rng.standard_normal(n_ch) * 0.5 + 0.5, 0, 1)
        X[:, onset:onset + dur] += gamma[None, :] * w[:, None]

    line_amp = rng.uniform(2, 12, size=(n_ch, 1))
    X += (np.sin(2 * np.pi * 50 * t)[None, :]
          + 0.4 * np.sin(2 * np.pi * 100 * t)[None, :]) * line_amp

    X[3] = 0.2 * rng.standard_normal(n_t)
    X[28] = np.clip(X[28] * 50, -800, 800)

    stim_onsets = np.array([int(20 * fs), int(55 * fs), int(90 * fs)])
    template = 400 * np.exp(-np.arange(0, int(0.15 * fs)) / (0.02 * fs))
    for on in stim_onsets:
        n = min(len(template), n_t - on)
        X[10:14, on:on + n] += template[:n][None, :]

    seg = slice(int(35 * fs), int(45 * fs))
    drift = 600 * np.sin(np.linspace(0, np.pi, seg.stop - seg.start))[None, :]
    X[5:13, seg] += drift

    return {
        "X": X.astype(np.float32),
        "fs": fs,
        "ch_names": [f"ch{i:02d}" for i in range(n_ch)],
        "stim_onsets": stim_onsets,
        "motion_seg": (int(35 * fs), int(45 * fs)),
        "bad_channels": np.array([3, 28]),
    }


# ==================================================== 0 DC / filtering

def highpass(x: np.ndarray, fs: float, cutoff: float = 0.5,
             order: int = 4) -> np.ndarray:
    """Zero-phase high-pass (removes DC and slow polarization drift)."""
    if cutoff <= 0 or cutoff >= fs / 2:
        raise ValueError(f"invalid highpass cutoff {cutoff} (fs={fs})")
    sos = sp_signal.butter(order, cutoff / (fs / 2), btype="high", output="sos")
    return sp_signal.sosfiltfilt(sos, np.asarray(x, dtype=float), axis=-1)


def lowpass(x: np.ndarray, fs: float, cutoff: float = 250.0,
            order: int = 4) -> np.ndarray:
    """Zero-phase low-pass; cutoff must stay below Nyquist."""
    if cutoff <= 0 or cutoff >= fs / 2:
        raise ValueError(f"invalid lowpass cutoff {cutoff} (fs={fs})")
    sos = sp_signal.butter(order, cutoff / (fs / 2), btype="low", output="sos")
    return sp_signal.sosfiltfilt(sos, np.asarray(x, dtype=float), axis=-1)


def bandpass_filter(x: np.ndarray, fs: float, low: float, high: float,
                    order: int = 4) -> np.ndarray:
    """Zero-phase band-pass."""
    if low <= 0 or high >= fs / 2 or low >= high:
        raise ValueError(f"invalid bandpass range {low}-{high} Hz (fs={fs})")
    sos = sp_signal.butter(order, [low / (fs / 2), high / (fs / 2)],
                           btype="band", output="sos")
    return sp_signal.sosfiltfilt(sos, np.asarray(x, dtype=float), axis=-1)


def detrend_linear(x: np.ndarray, fs: float, seg_sec: float = 10.0,
                   order: int = 1) -> np.ndarray:
    """Hann-weighted 50%-overlap polynomial detrending.

    Independent fits on hard-cut windows leave voltage steps at window
    boundaries; sliding windows with Hann weights blend smoothly instead.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    seg_n = int(seg_sec * fs)
    step = max(seg_n // 2, 1)
    trend_acc = np.zeros_like(x)
    weight = np.zeros(n)
    for start in range(0, n, step):
        end = min(start + seg_n, n)
        tt = np.arange(end - start)
        w = np.hanning(end - start)
        for ch in range(x.shape[0]):
            coef = np.polyfit(tt, x[ch, start:end], order)
            trend_acc[ch, start:end] += np.polyval(coef, tt) * w
        weight[start:end] += w
    valid = weight > 0
    out = x.copy()
    out[:, valid] = x[:, valid] - trend_acc[:, valid] / weight[valid]
    return out


def remove_dc(x: np.ndarray, method: str = "median") -> np.ndarray:
    """Remove the DC offset. 'median' is robust to stimulation pulses."""
    x = np.asarray(x, dtype=float).copy()
    if method in ("none", None):
        return x
    if method == "mean":
        return x - x.mean(axis=-1, keepdims=True)
    if method == "median":
        return x - np.median(x, axis=-1, keepdims=True)
    raise ValueError(f"unknown DC method: {method}")


def apply_filter_baseline(x: np.ndarray, fs: float, method: str = "none",
                          hp_cutoff: float = 0.5, lp_cutoff: float = 250.0,
                          detrend_seg: float = 10.0,
                          detrend_order: int = 1) -> np.ndarray:
    """Step-3 unified entry: highpass / lowpass / bandpass / detrend / none."""
    if method == "none":
        return np.asarray(x, dtype=float).copy()
    if method == "highpass":
        return highpass(x, fs, hp_cutoff)
    if method == "lowpass":
        return lowpass(x, fs, lp_cutoff)
    if method == "bandpass":
        return bandpass_filter(x, fs, hp_cutoff, lp_cutoff)
    if method == "detrend":
        return detrend_linear(x, fs, detrend_seg, detrend_order)
    raise ValueError(f"unknown filter method: {method}")


# ======================================================== 1 bad channels

def detect_bad_mad(x: np.ndarray, n_mad: float = 5.0) -> np.ndarray:
    """Two-sided log-MAD criterion: catches noisy and dead channels."""
    ch_std = np.median(np.abs(x - np.median(x, axis=-1, keepdims=True)), axis=-1)
    log_std = np.log(ch_std + 1e-12)
    med = np.median(log_std)
    mad = np.median(np.abs(log_std - med))
    z = (log_std - med) / (1.4826 * mad + 1e-12)
    return np.where(np.abs(z) > n_mad)[0]


def detect_bad_ptp(x: np.ndarray, low: float = 1e-4,
                   high: float = 1000.0) -> np.ndarray:
    """Two-sided peak-to-peak criterion: low = dead, high = saturated."""
    ptp = np.ptp(x, axis=-1)
    return np.where((ptp < low) | (ptp > high))[0]


def detect_bad_flatness(x: np.ndarray, fs: float, thresh: float = 0.75) -> np.ndarray:
    """Spectral-flatness criterion: near-white channels are suspicious."""
    bad = []
    for i in range(x.shape[0]):
        _f, pxx = sp_signal.welch(x[i], fs=fs, nperseg=min(x.shape[1], int(fs * 2)))
        p = pxx[pxx > 0]
        if len(p) < 4:
            bad.append(i)
            continue
        flat = float(np.exp(np.log(p).mean()) / p.mean())
        if flat > thresh:
            bad.append(i)
    return np.asarray(bad, dtype=int)


def detect_bad_correlation(x: np.ndarray, low: float = 0.02,
                           high: float = 0.999) -> np.ndarray:
    """Cross-channel correlation criterion: low = loose contact,
    high = bridged/shorted pair."""
    x = np.asarray(x, dtype=float)
    z = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-12)
    r = np.corrcoef(z)
    np.fill_diagonal(r, 0)
    mean_abs = np.abs(r).mean(axis=-1)
    return np.where((mean_abs < low) | (mean_abs > high))[0]


def detect_bad_channels(x: np.ndarray, fs: float, method: str = "mad",
                        **kwargs) -> np.ndarray:
    """method: single criterion, '+' union, 'vote:' majority, or 'ica'."""
    if method.startswith("vote:"):
        inner = method[5:].split("+")
        return detect_bad_vote(x, fs, inner, kwargs.get("min_votes", 2), **kwargs)[0]
    if method == "ica":
        return detect_bad_ica(x, fs, **kwargs)
    results = []
    for m in method.split("+"):
        if m == "mad":
            results.append(detect_bad_mad(x, kwargs.get("n_mad", 5.0)))
        elif m == "ptp":
            results.append(detect_bad_ptp(x, kwargs.get("low", 1e-4),
                                          kwargs.get("high", 1000.0)))
        elif m == "flatness":
            results.append(detect_bad_flatness(x, fs, kwargs.get("flat_thresh", 0.75)))
        elif m == "corr":
            results.append(detect_bad_correlation(x, kwargs.get("corr_low", 0.02),
                                                  kwargs.get("corr_high", 0.999)))
        elif m == "none":
            return np.asarray([], dtype=int)
    return np.unique(np.concatenate(results)) if results else np.asarray([], dtype=int)


def detect_bad_vote(x: np.ndarray, fs: float, methods: list[str],
                    min_votes: int = 2, **kwargs) -> tuple[np.ndarray, dict]:
    """Majority vote over criteria; every per-criterion result is returned."""
    detail = {}
    counts = np.zeros(x.shape[0], dtype=int)
    for m in methods:
        res = detect_bad_channels(x, fs, m, **kwargs)
        detail[m] = res
        for ch in np.unique(res):
            counts[ch] += 1
    final = np.where(counts >= min_votes)[0]
    return final, detail


def ica_report(x: np.ndarray, fs: float, n_components: int | None = None,
               var_explained: float = 0.95, seed: int = 0,
               max_iter: int = 500) -> dict:
    """ICA decomposition and per-channel reconstruction report.

    Component count adapts to the PCA variance explained (var_explained),
    clipped to [20, n_ch], so dense grids are not forced through a fixed
    20-component subspace.
    """
    from sklearn.decomposition import FastICA, PCA
    x = np.asarray(x, dtype=float)
    z = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-12)
    if n_components is None:
        pca = PCA(n_components=min(x.shape[0], x.shape[-1]))
        pca.fit(z.T)
        cum = np.cumsum(pca.explained_variance_ratio_)
        n_c = int(np.searchsorted(cum, var_explained) + 1)
        n_c = int(np.clip(n_c, min(20, x.shape[0]), x.shape[0]))
    else:
        n_c = int(min(n_components, x.shape[0]))
    ica = FastICA(n_components=n_c, random_state=seed, max_iter=max_iter)
    S = ica.fit_transform(z.T)
    A = ica.mixing_
    rec = (S @ A.T).T
    resid = z - rec
    vaf = 1.0 - resid.var(axis=-1)
    inv_vaf = 1.0 / np.clip(vaf, 1e-6, None)
    kurt = _kurtosis(S, axis=0)
    resid_kurt = _kurtosis(resid, axis=-1)
    return {"vaf": vaf, "inv_vaf": inv_vaf, "mixing": A, "sources": S,
            "kurtosis": kurt, "resid_kurt": resid_kurt, "n_components": n_c}


def _kurtosis(s: np.ndarray, axis: int = 0) -> np.ndarray:
    z = (s - s.mean(axis=axis, keepdims=True)) / (s.std(axis=axis, keepdims=True) + 1e-12)
    return np.asarray((z ** 4).mean(axis=axis))


def detect_bad_ica(x: np.ndarray, fs: float, inv_vaf_z: float = 4.0,
                   resid_kurt_min: float = 5.0, kurt_z: float = 3.0,
                   **kwargs) -> np.ndarray:
    """ICA criterion: robust relative 1/VAF (MAD z) + residual kurtosis.

    Fixed absolute thresholds misclassify channels on dense grids; robust
    z-scores plus a spike-kurtosis requirement are used instead. Channels
    loading most strongly on extreme-kurtosis components are also flagged.
    """
    rep = ica_report(x, fs, **kwargs)
    inv = rep["inv_vaf"]
    med = np.median(inv)
    mad = np.median(np.abs(inv - med))
    iz = (inv - med) / (1.4826 * mad + 1e-12)
    bad = set(np.where((iz > inv_vaf_z) & (rep["resid_kurt"] > resid_kurt_min))[0].tolist())
    k = rep["kurtosis"]
    if len(k) > 2:
        kz = (k - k.mean()) / (k.std() + 1e-12)
        for comp in np.where(np.abs(kz) > kurt_z)[0]:
            bad.add(int(np.argmax(np.abs(rep["mixing"][:, comp]))))
    return np.asarray(sorted(bad), dtype=int)


def handle_bad_channels(x: np.ndarray, bad: np.ndarray, method: str = "mask",
                        shafts: list[np.ndarray] | None = None,
                        n_neighbors: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Handle bad channels (none / mask / interpolate / drop).
    Returns (x, mask); the mask matches the returned rows (all-False for drop).
    """
    x = np.asarray(x, dtype=float).copy()
    bad = np.unique(np.asarray(bad, dtype=int))
    mask = np.zeros(x.shape[0], dtype=bool)
    mask[bad] = True
    if method in ("none", None):
        return x, mask
    if method == "drop":
        keep = ~mask
        return x[keep], np.zeros(int(keep.sum()), dtype=bool)
    if method == "mask":
        x[mask] = 0.0
        return x, mask
    if method == "interpolate":
        groups = shafts if shafts else [np.arange(x.shape[0])]
        for grp in groups:
            grp = np.asarray(grp, dtype=int)
            gmask = mask[grp]
            if gmask.all():
                continue
            good = grp[~gmask]
            for b in grp[gmask]:
                w = 1.0 / (np.abs(good - b) + 1.0)
                k = min(n_neighbors, len(good))
                nn = good[np.argsort(np.abs(good - b))[:k]]
                x[b] = np.average(x[nn], axis=0,
                                   weights=w[np.argsort(np.abs(good - b))[:k]])
        return x, mask
    raise ValueError(f"unknown bad-channel handling: {method}")


def compress_pairs(pairs: list[tuple[int, int]],
                   bad: np.ndarray) -> list[tuple[int, int]]:
    """Bipolar pairs: drop pairs touching bad channels."""
    bad_set = set(np.asarray(bad, dtype=int).tolist())
    return [(a, b) for a, b in pairs if a not in bad_set and b not in bad_set]


def compress_adjacency(adjacency: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """Adjacency matrix: drop bad-channel rows and columns."""
    keep = np.ones(adjacency.shape[0], dtype=bool)
    keep[np.asarray(bad, dtype=int)] = False
    return adjacency[np.ix_(keep, keep)]


# ================================================ 2 stimulation artifacts

def remove_stim_artifacts(x: np.ndarray, fs: float, onsets: np.ndarray,
                          method: str = "template_subtract",
                          pre: float = 0.05, post: float = 0.15) -> np.ndarray:
    """Remove stimulation artifacts.

    template_subtract: per-channel median template over the post-onset
    windows, subtracted from each artifact window.
    interpolate: linear bridge across artifact boundaries (short pulses).
    blank: zero the window (creates edges; not recommended).
    """
    out = np.asarray(x, dtype=float).copy()
    n_pre, n_post = int(pre * fs), int(post * fs)
    valid = np.asarray(onsets, dtype=int)
    valid = valid[(valid > n_pre) & (valid < x.shape[-1] - n_post)]
    if method == "blank":
        for on in valid:
            out[:, on:on + n_post] = 0.0
        return out
    if method == "template_subtract":
        for ch in range(x.shape[0]):
            segs = np.stack([x[ch, on:on + n_post] for on in valid])
            template = np.median(segs, axis=0)
            for on in valid:
                n = min(n_post, x.shape[-1] - on)
                k = min(n, len(template))
                out[ch, on:on + k] -= template[:k]
        return out
    if method == "interpolate":
        for ch in range(x.shape[0]):
            for on in valid:
                end = min(on + n_post, x.shape[-1])
                n = end - on
                a, b = out[ch, on - 1], out[ch, end] if end < x.shape[-1] else out[ch, on - 1]
                out[ch, on:end] = np.linspace(a, b, n)
        return out
    raise ValueError(f"unknown stimulation-artifact method: {method}")


# ============================================================ 4 line noise

def _line_freq_list(fs: float, f0: float = 50.0, harmonics: int = 1,
                    freqs: list | None = None) -> np.ndarray:
    """Line frequencies = integer harmonics + custom list (deduped, < Nyquist)."""
    nyq = fs / 2 - 1
    out = [f0 * h for h in range(1, harmonics + 1) if f0 * h < nyq]
    for f in (freqs or []):
        if 0 < float(f) < nyq:
            out.append(float(f))
    return np.unique(np.asarray(out, dtype=float))


def notch_iir(x: np.ndarray, fs: float, f0: float = 50.0, q: float = 30.0,
              harmonics: int = 1, freqs: list | None = None) -> np.ndarray:
    """Zero-phase IIR notch across harmonics + custom frequencies."""
    out = np.asarray(x, dtype=float).copy()
    for f in _line_freq_list(fs, f0, harmonics, freqs):
        b, a = sp_signal.iirnotch(f, q, fs)
        out = sp_signal.filtfilt(b, a, out, axis=-1)
    return out


def notch_fir(x: np.ndarray, fs: float, f0: float = 50.0, width: float = 2.0,
              harmonics: int = 1, freqs: list | None = None,
              numtaps: int = 401) -> np.ndarray:
    """Zero-phase FIR bandstop across harmonics + custom frequencies."""
    out = np.asarray(x, dtype=float).copy()
    for f in _line_freq_list(fs, f0, harmonics, freqs):
        b = sp_signal.firwin(numtaps, [f - width, f + width], fs=fs,
                             pass_zero="bandstop")
        out = sp_signal.filtfilt(b, [1.0], out, axis=-1)
    return out


def notch_fft_interp(x: np.ndarray, fs: float, f0: float = 50.0,
                     width: float = 2.0, harmonics: int = 1,
                     freqs: list | None = None) -> np.ndarray:
    """FFT interpolation repair: replace only the narrow line band's
    magnitude, keep the phase; leaves no notch in the log PSD."""
    out = np.asarray(x, dtype=float).copy()
    n = out.shape[-1]
    grid = np.fft.rfftfreq(n, 1 / fs)
    for ch in range(out.shape[0]):
        spec = np.fft.rfft(out[ch])
        mag = np.abs(spec)
        phase = np.angle(spec)
        for f in _line_freq_list(fs, f0, harmonics, freqs):
            band = (grid >= f - width) & (grid <= f + width)
            idx = np.where(band)[0]
            if len(idx) < 3:
                continue
            lo = max(0, idx[0] - 2)
            hi = min(len(grid) - 1, idx[-1] + 2)
            outside = np.delete(np.arange(lo, hi + 1),
                                np.where(band[lo:hi + 1])[0])
            if len(outside) >= 2:
                mag[band] = np.interp(grid[band],
                                      [grid[outside[0]], grid[outside[-1]]],
                                      [mag[outside[0]], mag[outside[-1]]])
        out[ch] = np.fft.irfft(mag * np.exp(1j * phase), n=n)
    return out


def cleanline_regression(x: np.ndarray, fs: float, f0: float = 50.0,
                         harmonics: int = 2, freqs: list | None = None,
                         win: float = 4.0) -> np.ndarray:
    """Sliding-window spectral regression (sin/cos bases per frequency),
    Hann-weighted overlap-add; robust to slow phase drift."""
    x = np.asarray(x, dtype=float)
    n_t = x.shape[-1]
    win_n = int(win * fs)
    half = win_n // 2
    clean = np.zeros_like(x)
    weight = np.zeros(n_t)
    for s in range(0, n_t, half):
        sl = slice(s, min(s + win_n, n_t))
        seg = x[..., sl]
        t = np.arange(sl.stop - sl.start) / fs
        cols = []
        for f in _line_freq_list(fs, f0, harmonics, freqs):
            cols += [np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)]
        D = np.stack(cols, axis=0)
        coef = np.linalg.lstsq(D.T, seg.T, rcond=None)[0]
        residual = seg - (coef.T @ D)
        win_w = np.hanning(sl.stop - sl.start)
        clean[..., sl] += residual * win_w
        weight[sl] += win_w
    return clean / (weight + 1e-12)


def remove_line_noise(x: np.ndarray, fs: float, method: str = "iir_notch",
                      **kwargs) -> np.ndarray:
    """method: iir_notch / fir_notch / cleanline / fft_interp / none."""
    if method == "none":
        return np.asarray(x, dtype=float).copy()
    if method == "iir_notch":
        return notch_iir(x, fs, kwargs.get("f0", 50.0), kwargs.get("q", 30.0),
                         kwargs.get("harmonics", 1), kwargs.get("freqs"))
    if method == "fir_notch":
        return notch_fir(x, fs, kwargs.get("f0", 50.0), kwargs.get("width", 2.0),
                         kwargs.get("harmonics", 1), kwargs.get("freqs"),
                         kwargs.get("numtaps", 401))
    if method == "cleanline":
        return cleanline_regression(x, fs, kwargs.get("f0", 50.0),
                                    kwargs.get("harmonics", 2), kwargs.get("freqs"),
                                    kwargs.get("win", 4.0))
    if method == "fft_interp":
        return notch_fft_interp(x, fs, kwargs.get("f0", 50.0),
                                kwargs.get("width", 2.0), kwargs.get("harmonics", 1),
                                kwargs.get("freqs"))
    raise ValueError(f"unknown line-noise method: {method}")


# ====================================================== 5 motion / EMG

def detect_motion_segments(x: np.ndarray, fs: float, win: float = 1.0,
                           z_thresh: float = 4.0, lf: float = 2.0) -> np.ndarray:
    """Slow-drift energy mask with median/MAD robust z-scores (a violent
    artifact cannot inflate the threshold and hide milder segments)."""
    x = np.asarray(x, dtype=float)
    sos = sp_signal.butter(4, lf / (fs / 2), btype="low", output="sos")
    low = sp_signal.sosfiltfilt(sos, x, axis=-1)
    win_n = int(win * fs)
    half = win_n // 2
    idx = np.arange(half, x.shape[-1] - half, win_n)
    seg_std = np.array([low[:, i - half:i + half].std(axis=-1).mean() for i in idx])
    med = np.median(seg_std)
    mad = np.median(np.abs(seg_std - med))
    z = (seg_std - med) / (1.4826 * mad + 1e-12)
    mask = np.zeros(x.shape[-1], dtype=bool)
    for i in idx[z > z_thresh]:
        mask[max(0, i - half):min(x.shape[-1], i + half)] = True
    return mask


def detect_emg_segments(x: np.ndarray, fs: float, lo: float = 60.0,
                        hi: float = 180.0, win: float = 1.0,
                        z_thresh: float = 5.0, min_dur: float = 0.1) -> np.ndarray:
    """High-frequency EMG mask: band envelope window RMS with MAD-robust z."""
    x = np.asarray(x, dtype=float)
    if hi >= fs / 2 - 1:
        hi = fs / 2 - 2
    if lo <= 0 or hi <= lo:
        raise ValueError(f"invalid EMG band {lo}-{hi} Hz (fs={fs})")
    sos = sp_signal.butter(4, [lo / (fs / 2), hi / (fs / 2)],
                           btype="band", output="sos")
    xb = sp_signal.sosfiltfilt(sos, x, axis=-1)
    env = np.abs(sp_signal.hilbert(xb, axis=-1))
    win_n = int(win * fs)
    half = win_n // 2
    idx = np.arange(half, x.shape[-1] - half, win_n)
    seg_rms = np.array([env[:, i - half:i + half].mean(axis=-1).mean() for i in idx])
    med = np.median(seg_rms)
    mad = np.median(np.abs(seg_rms - med))
    z = (seg_rms - med) / (1.4826 * mad + 1e-12)
    mask = np.zeros(x.shape[-1], dtype=bool)
    for i in idx[z > z_thresh]:
        mask[max(0, i - half):min(x.shape[-1], i + half)] = True
    runs, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    min_n = int(min_dur * fs)
    for s, e in runs:
        if e - s < min_n:
            mask[s:e] = False
    return mask


def interpolate_masked_segments(x: np.ndarray, mask: np.ndarray, fs: float,
                                max_gap: float = 0.5) -> np.ndarray:
    """Bridge short masked segments by linear interpolation; longer segments
    stay untouched and are excluded later via the propagated mask."""
    out = np.asarray(x, dtype=float).copy()
    mask = np.asarray(mask, dtype=bool)
    max_n = int(max_gap * fs)
    runs, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    for s, e in runs:
        if e - s > max_n:
            continue
        for ch in range(out.shape[0]):
            a = out[ch, s - 1] if s > 0 else out[ch, e]
            b = out[ch, e] if e < n else out[ch, s - 1]
            out[ch, s:e] = np.linspace(a, b, e - s)
    return out


# ====================================================== 6 re-referencing

def parse_seeg_shafts(ch_names: list[str]) -> list[np.ndarray]:
    """Parse depth-electrode shafts from channel names ('LA1'..'LA12')."""
    shafts: dict[str, list[int]] = {}
    for i, name in enumerate(ch_names):
        m = re.match(r"^([A-Za-z]+)[0-9]+$", str(name))
        key = m.group(1) if m else "?"
        shafts.setdefault(key, []).append(i)
    return [np.asarray(v, dtype=int) for v in shafts.values()]


def chain_bipolar_pairs(n_ch: int,
                        shafts: list[np.ndarray] | None = None) -> list[tuple[int, int]]:
    """Bipolar pairs; within-shaft only when shafts are provided."""
    if shafts is None:
        return [(i, i + 1) for i in range(n_ch - 1)]
    pairs = []
    for grp in shafts:
        g = np.sort(np.asarray(grp, dtype=int))
        pairs += [(int(g[i]), int(g[i + 1])) for i in range(len(g) - 1)]
    return pairs


def car_reference(x: np.ndarray, bad_mask: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    if bad_mask is None:
        bad_mask = np.zeros(x.shape[0], dtype=bool)
    good = ~np.asarray(bad_mask)
    mean = x[good].mean(axis=0, keepdims=True)
    out = x.copy()
    out[good] = x[good] - mean
    out[~good] = 0.0
    return out


def median_reference(x: np.ndarray, bad_mask: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    if bad_mask is None:
        bad_mask = np.zeros(x.shape[0], dtype=bool)
    good = ~np.asarray(bad_mask)
    med = np.median(x[good], axis=0, keepdims=True)
    out = x.copy()
    out[good] = x[good] - med
    out[~good] = 0.0
    return out


def shaft_car_reference(x: np.ndarray, shafts: list[np.ndarray],
                        bad_mask: np.ndarray | None = None) -> np.ndarray:
    """Per-shaft common-average reference (depth electrodes)."""
    out = np.asarray(x, dtype=float).copy()
    if bad_mask is None:
        bad_mask = np.zeros(x.shape[0], dtype=bool)
    bad = np.asarray(bad_mask)
    for grp in shafts:
        g = np.asarray(grp, dtype=int)
        good = g[~bad[g]]
        if len(good):
            out[good] = out[good] - out[good].mean(axis=0, keepdims=True)
    return out


def group_car_reference(x: np.ndarray, groups: list[np.ndarray],
                        bad_mask: np.ndarray | None = None) -> np.ndarray:
    """Per-group common-average reference; distant grids must not share a
    common mode, otherwise one grid's activity leaks into the other."""
    out = np.asarray(x, dtype=float).copy()
    if bad_mask is None:
        bad_mask = np.zeros(x.shape[0], dtype=bool)
    bad = np.asarray(bad_mask)
    for grp in groups:
        g = np.asarray(grp, dtype=int)
        good = g[~bad[g]]
        if len(good):
            out[good] = out[good] - out[good].mean(axis=0, keepdims=True)
    return out


def white_matter_reference(x: np.ndarray, wm_channels: list[int]) -> np.ndarray:
    """Subtract the mean of designated white-matter contacts."""
    wm = np.asarray(wm_channels, dtype=int)
    return np.asarray(x, dtype=float) - x[wm].mean(axis=0, keepdims=True)


def bipolar_reference(x: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    return np.stack([x[a] - x[b] for a, b in pairs], axis=0)


def laplacian_reference(x: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
    """Subtract the neighbor average (normalized by true degree)."""
    x = np.asarray(x, dtype=float)
    A = np.asarray(adjacency, dtype=float)
    deg = A.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    return x - (A @ x) / deg


def apply_reference(x: np.ndarray, method: str = "none",
                    bad_mask: np.ndarray | None = None,
                    pairs: list | None = None, adjacency: np.ndarray | None = None,
                    shafts: list[np.ndarray] | None = None,
                    wm_channels: list[int] | None = None,
                    groups: list[np.ndarray] | None = None) -> np.ndarray:
    """Unified reference entry. The Laplacian variants zero bad rows and fall
    back to the good-channel mean for isolated nodes (all-neighbors-bad)."""
    if bad_mask is None:
        bad_mask = np.zeros(x.shape[0], dtype=bool)
    bad = np.asarray(bad_mask)
    if method == "none":
        return np.asarray(x, dtype=float).copy()
    if method == "car":
        return car_reference(x, bad)
    if method == "median":
        return median_reference(x, bad)
    if method == "shaft_car":
        if shafts is None:
            raise ValueError("shaft_car requires shafts")
        return shaft_car_reference(x, shafts, bad)
    if method == "group_car":
        if groups is None:
            raise ValueError("group_car requires groups")
        return group_car_reference(x, groups, bad)
    if method == "white_matter":
        if wm_channels is None:
            raise ValueError("white_matter requires wm_channels")
        return white_matter_reference(x, wm_channels)
    if method == "bipolar":
        if pairs is None:
            raise ValueError("bipolar requires pairs")
        return bipolar_reference(x, compress_pairs(pairs, np.where(bad)[0]))
    if method in ("laplacian", "laplacian8"):
        if adjacency is None:
            raise ValueError(f"{method} requires adjacency")
        adj = np.asarray(adjacency, dtype=float).copy()
        adj[:, bad] = 0.0
        deg = adj.sum(axis=1)
        out = np.asarray(laplacian_reference(x, adj), dtype=float)
        out[bad] = 0.0
        isolated = (deg == 0) & (~bad)
        if isolated.any() and (~bad).any():
            mean_good = np.asarray(x, dtype=float)[~bad].mean(axis=0, keepdims=True)
            out[isolated] = np.asarray(x, dtype=float)[isolated] - mean_good
        return out
    raise ValueError(f"unknown reference method: {method}")


# ============================================================ 8 resampling

def resample(x: np.ndarray, fs: float, target_fs: float) -> tuple[np.ndarray, float]:
    """Polyphase resampling. Returns (data, new sampling rate)."""
    from fractions import Fraction
    ratio = Fraction(target_fs / fs).limit_denominator(1000)
    out = sp_signal.resample_poly(np.asarray(x, dtype=float), ratio.numerator,
                                  ratio.denominator, axis=-1)
    return out, fs * ratio.numerator / ratio.denominator


# ===================================================== 9-10 epochs / QC

def make_epochs(x: np.ndarray, fs: float, events: np.ndarray, tmin: float,
                tmax: float, baseline: tuple[float, float] | None = None,
                bad_time_mask: np.ndarray | None = None,
                cond_labels: np.ndarray | None = None,
                evoked_sub: str = "none") -> tuple[np.ndarray, list[int]]:
    """Event-locked epoching with optional baseline correction, artifact
    overlap rejection, and evoked subtraction (none / global / per_condition).
    cond_labels must align with events (one label per event)."""
    events = np.asarray(events, dtype=int)
    n_pre, n_post = int(-tmin * fs), int(tmax * fs)
    epochs, kept, kept_idx = [], [], []
    for i, on in enumerate(events):
        if on - n_pre < 0 or on + n_post > x.shape[-1]:
            continue
        if bad_time_mask is not None and bad_time_mask[on - n_pre:on + n_post].any():
            continue
        seg = x[:, on - n_pre:on + n_post].copy()
        if baseline is not None:
            b0 = int((baseline[0] - tmin) * fs)
            b1 = int((baseline[1] - tmin) * fs)
            seg = seg - seg[:, b0:b1].mean(axis=-1, keepdims=True)
        epochs.append(seg)
        kept.append(int(on))
        kept_idx.append(i)
    if epochs:
        epochs = np.stack(epochs)
        if evoked_sub != "none":
            labs = None
            if cond_labels is not None:
                labs = np.asarray(cond_labels)[kept_idx]
            epochs = subtract_evoked(epochs, labs, evoked_sub)
        return epochs, kept
    return np.zeros((0, x.shape[0], n_pre + n_post)), kept


def subtract_evoked(epochs: np.ndarray, cond_labels: np.ndarray | None,
                    mode: str) -> np.ndarray:
    """Evoked subtraction. global: subtract the grand trial average;
    per_condition: subtract each condition's own trial average. Averages use
    only the retained (boundary- and artifact-free) trials."""
    epochs = np.asarray(epochs, dtype=float).copy()
    n = epochs.shape[0]
    if mode == "global":
        return epochs - epochs.mean(axis=0, keepdims=True)
    if mode == "per_condition":
        if cond_labels is None:
            raise ValueError("evoked_sub='per_condition' requires cond_labels")
        labs = np.asarray(cond_labels)
        if len(labs) != n:
            raise ValueError(f"condition labels {len(labs)} != trials {n}")
        for lab in np.unique(labs):
            sel = labs == lab
            epochs[sel] -= epochs[sel].mean(axis=0, keepdims=True)
        return epochs
    raise ValueError(f"unknown evoked subtraction mode: {mode}")


def epoch_qc_mask(epochs: np.ndarray, ptp_max: float = 800.0,
                  var_z: float = 4.0) -> np.ndarray:
    """Epoch QC: reject peak-to-peak or variance outliers."""
    ptp = np.ptp(epochs, axis=-1).max(axis=-1)
    var = epochs.var(axis=-1).mean(axis=-1)
    z = (var - var.mean()) / (var.std() + 1e-12)
    return (ptp < ptp_max) & (z < var_z)


# ============================================================== IED / 7

def detect_ied_spikes(x: np.ndarray, fs: float, hp: float = 20.0,
                      lp: float = 80.0, z_thresh: float = 5.0,
                      min_dist: float = 0.05,
                      report_only: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Interictal epileptiform discharge detection (tri-criterion):
    1) 20-80 Hz envelope z-score (MAD-robust), 2) amplitude threshold,
    3) first-derivative steepness. Returns (event samples, time mask).
    report_only keeps the flag; set False to merge into artifact masks."""
    sos = sp_signal.butter(4, [hp / (fs / 2), lp / (fs / 2)],
                           btype="band", output="sos")
    xb = sp_signal.sosfiltfilt(sos, np.asarray(x, dtype=float), axis=-1)
    env = np.abs(sp_signal.hilbert(xb, axis=-1))
    med = np.median(env, axis=-1, keepdims=True)
    mad = np.median(np.abs(env - med), axis=-1, keepdims=True)
    z = (env - med) / (1.4826 * mad + 1e-12)
    diff = np.abs(np.diff(xb, axis=-1, prepend=xb[:, :1]))
    dth = diff.mean(axis=-1) + 5 * diff.std(axis=-1)
    strong = ((z > z_thresh)
              & (np.abs(xb) > np.abs(xb).mean(axis=-1, keepdims=True)
                 + 4 * np.abs(xb).std(axis=-1, keepdims=True))
              & (diff > dth[:, None]))
    hit = strong.any(axis=0)
    n_t = x.shape[-1]
    mask = np.zeros(n_t, dtype=bool)
    idx = np.where(hit)[0]
    min_gap = int(min_dist * fs)
    keep = []
    for i in idx:
        if not keep or i - keep[-1] > min_gap:
            keep.append(i)
    for i in keep:
        mask[max(0, i - min_gap):min(n_t, i + min_gap)] = True
    return np.asarray(keep, dtype=int), mask


# ============================================================== config

@dataclass
class PreprocessConfig:
    dc_method: str = "none"            # none/mean/median; before all filtering
    filter_method: str = "none"        # none/highpass/lowpass/bandpass/detrend
    hp_cutoff: float = 0.5
    lp_cutoff: float = 250.0
    detrend_seg: float = 10.0
    detrend_order: int = 1
    bad_detect: str = "mad+ptp"
    manual_bad: list = field(default_factory=list)  # union with auto results
    bad_handle: str = "mask"
    stim: str = "none"
    stim_pre: float = 0.05
    stim_post: float = 0.15
    line_method: str = "iir_notch"
    line_f0: float = 50.0
    line_q: float = 30.0
    line_width: float = 2.0
    line_win: float = 4.0
    line_harmonics: int = 1
    line_freqs: list = field(default_factory=list)  # extra device-noise Hz
    motion: str = "none"
    motion_z: float = 4.0
    motion_win: float = 1.0
    motion_lf: float = 2.0
    motion_max_gap: float = 0.5
    emg: str = "none"
    emg_z: float = 5.0
    emg_win: float = 1.0
    emg_lo: float = 60.0
    emg_hi: float = 180.0
    ied: str = "none"
    ied_z: float = 5.0
    reference: str = "none"
    resample_to: float | None = None
    events: np.ndarray | None = None
    tmin: float = -1.0
    tmax: float = 1.0
    baseline: tuple[float, float] | None = None
    cond_labels: np.ndarray | None = None
    evoked_sub: str = "none"
    ptp_max: float = 800.0
    qc_var_z: float = 4.0
    detect_kwargs: dict = field(default_factory=dict)


# ========================================================= orchestration

def run_preprocessing(x: np.ndarray, fs: float, cfg: PreprocessConfig,
                      bipolar_pairs: list | None = None,
                      adjacency: np.ndarray | None = None,
                      stim_onsets: np.ndarray | None = None,
                      ch_names: list[str] | None = None,
                      wm_channels: list[int] | None = None,
                      groups: list[np.ndarray] | None = None,
                      force_bad: np.ndarray | None = None) -> dict:
    """Run the full protocol, returning {'data': ..., 'report': ...}.

    force_bad supplies a globally locked bad-channel set for batch runs.
    drop mode remaps bipolar pairs, groups, white-matter contacts, channel
    names and adjacency to the new row indices. The report carries the full
    parameter snapshot, bad channels, out_ch_map (output row -> original
    channel or bipolar pair), artifact coverage and epoch QC."""
    report: dict = {}
    xw = np.asarray(x, dtype=float).copy()
    n_ch_in = xw.shape[0]
    report["input_shape"] = list(xw.shape)
    report["fs_in"] = float(fs)
    report["config"] = {
        "dc_method": cfg.dc_method,
        "filter_method": cfg.filter_method,
        "hp_cutoff": cfg.hp_cutoff,
        "lp_cutoff": cfg.lp_cutoff,
        "detrend_seg": cfg.detrend_seg,
        "detrend_order": cfg.detrend_order,
        "bad_detect": cfg.bad_detect,
        "manual_bad": [int(v) for v in cfg.manual_bad],
        "bad_handle": cfg.bad_handle,
        "stim": cfg.stim,
        "stim_pre": cfg.stim_pre,
        "stim_post": cfg.stim_post,
        "line_method": cfg.line_method,
        "line_f0": cfg.line_f0,
        "line_q": cfg.line_q,
        "line_width": cfg.line_width,
        "line_win": cfg.line_win,
        "line_harmonics": cfg.line_harmonics,
        "line_freqs": [float(v) for v in cfg.line_freqs],
        "motion": cfg.motion,
        "motion_z": cfg.motion_z,
        "motion_win": cfg.motion_win,
        "motion_lf": cfg.motion_lf,
        "motion_max_gap": cfg.motion_max_gap,
        "emg": cfg.emg,
        "emg_z": cfg.emg_z,
        "emg_win": cfg.emg_win,
        "emg_lo": cfg.emg_lo,
        "emg_hi": cfg.emg_hi,
        "ied": cfg.ied,
        "ied_z": cfg.ied_z,
        "reference": cfg.reference,
        "resample_to": cfg.resample_to,
        "n_events": 0 if cfg.events is None else int(len(np.atleast_1d(cfg.events))),
        "n_cond_labels": 0 if cfg.cond_labels is None else int(len(np.atleast_1d(cfg.cond_labels))),
        "tmin": cfg.tmin,
        "tmax": cfg.tmax,
        "baseline": None if cfg.baseline is None else [float(v) for v in cfg.baseline],
        "evoked_sub": cfg.evoked_sub,
        "ptp_max": cfg.ptp_max,
        "qc_var_z": cfg.qc_var_z,
        "detect_kwargs": cfg.detect_kwargs,
    }
    shafts = parse_seeg_shafts(ch_names) if ch_names else None

    # 0 DC removal (before any temporal filter or template estimate)
    xw = remove_dc(xw, cfg.dc_method)
    report["dc_method"] = cfg.dc_method

    # 1 bad channels
    if force_bad is not None:
        bad = np.unique(np.asarray(force_bad, dtype=int))
        report["bad_detect_note"] = "locked by batch (force_bad)"
        report["bad_vote_detail"] = {}
        report["min_votes"] = None
    elif cfg.bad_detect.startswith("vote:"):
        detect_kwargs = dict(cfg.detect_kwargs)
        min_votes = detect_kwargs.pop("min_votes", 2)
        bad, vote_detail = detect_bad_vote(
            xw, fs, cfg.bad_detect[5:].split("+"),
            min_votes, **detect_kwargs)
        report["bad_vote_detail"] = {k: v.tolist() for k, v in vote_detail.items()}
        report["min_votes"] = min_votes
    else:
        bad = detect_bad_channels(xw, fs, cfg.bad_detect, **cfg.detect_kwargs)
    bad = np.unique(np.concatenate([bad, np.asarray(cfg.manual_bad, dtype=int)]))
    bad = bad[(bad >= 0) & (bad < n_ch_in)]
    report["bad_channels"] = bad.tolist()
    xw, bad_mask = handle_bad_channels(xw, bad, cfg.bad_handle, shafts)
    report["bad_handle"] = cfg.bad_handle
    if cfg.bad_handle == "drop":
        keep = np.ones(n_ch_in, dtype=bool)
        keep[bad] = False
        remap = np.full(n_ch_in, -1, dtype=int)
        remap[keep] = np.arange(int(keep.sum()))
        if bipolar_pairs is not None:
            bipolar_pairs = [(int(remap[a]), int(remap[b]))
                             for a, b in bipolar_pairs
                             if remap[a] >= 0 and remap[b] >= 0]
        if adjacency is not None:
            adjacency = compress_adjacency(np.asarray(adjacency), bad)
        if ch_names is not None:
            ch_names = [n for i, n in enumerate(ch_names) if i not in bad]
        if groups is not None:
            groups = [np.asarray([remap[c] for c in grp if remap[c] >= 0], dtype=int)
                      for grp in groups]
            groups = [g for g in groups if len(g)]
        if wm_channels is not None:
            wm_channels = [int(remap[c]) for c in wm_channels if remap[c] >= 0]
        shafts = parse_seeg_shafts(ch_names) if ch_names else None
    if cfg.bad_handle == "drop":
        ch_origin = np.asarray([i for i in range(n_ch_in)
                                if i not in set(bad.tolist())], dtype=int)
    else:
        ch_origin = np.arange(n_ch_in)

    # 2 stimulation artifacts (before any temporal filter)
    if cfg.stim != "none" and stim_onsets is not None:
        xw = remove_stim_artifacts(xw, fs, np.asarray(stim_onsets), cfg.stim,
                                   cfg.stim_pre, cfg.stim_post)
    report["stim_method"] = cfg.stim

    # 3 frequency filtering (after stimulation artifacts)
    xw = apply_filter_baseline(xw, fs, cfg.filter_method, cfg.hp_cutoff,
                               cfg.lp_cutoff, cfg.detrend_seg, cfg.detrend_order)
    report["filter_method"] = cfg.filter_method

    # 4 line noise
    xw = remove_line_noise(xw, fs, cfg.line_method, f0=cfg.line_f0,
                           q=cfg.line_q, width=cfg.line_width,
                           win=cfg.line_win, harmonics=cfg.line_harmonics,
                           freqs=list(cfg.line_freqs))
    report["line_method"] = cfg.line_method

    # 5 motion + EMG masks (never zero the signal)
    artifact_mask = np.zeros(xw.shape[-1], dtype=bool)
    motion_mask = None
    if cfg.motion in ("mask", "interpolate"):
        motion_mask = detect_motion_segments(xw, fs, win=cfg.motion_win,
                                             z_thresh=cfg.motion_z, lf=cfg.motion_lf)
        report["motion_frac"] = float(motion_mask.mean())
        artifact_mask |= motion_mask
        report["motion_mask"] = motion_mask
        if cfg.motion == "interpolate":
            xw = interpolate_masked_segments(xw, motion_mask, fs, cfg.motion_max_gap)
            report["motion_note"] = (f"short segments (<= {cfg.motion_max_gap}s) "
                                     "interpolated; long ones excluded via mask")
    report["motion"] = cfg.motion
    if cfg.emg == "mask":
        emg_mask = detect_emg_segments(xw, fs, lo=cfg.emg_lo, hi=cfg.emg_hi,
                                       win=cfg.emg_win, z_thresh=cfg.emg_z)
        report["emg_frac"] = float(emg_mask.mean())
        artifact_mask |= emg_mask
    report["artifact_frac"] = float(artifact_mask.mean())

    # 6 re-referencing
    if cfg.reference in ("laplacian", "laplacian8") and adjacency is not None:
        adj0 = np.asarray(adjacency, dtype=float).copy()
        adj0[:, np.asarray(bad_mask)] = 0.0
        deg = adj0.sum(axis=1)
        report["laplacian_isolated"] = np.where(
            (deg == 0) & (~np.asarray(bad_mask)))[0].tolist()
    xw = apply_reference(xw, cfg.reference, bad_mask,
                         bipolar_pairs, adjacency, shafts, wm_channels, groups)
    report["reference"] = cfg.reference
    report["n_ch_out"] = xw.shape[0]
    if cfg.reference == "bipolar" and bipolar_pairs is not None:
        report["out_ch_map"] = [[int(ch_origin[a]), int(ch_origin[b])]
                                for a, b in bipolar_pairs]
    else:
        report["out_ch_map"] = [int(ch_origin[i]) for i in range(xw.shape[0])]

    # 7 IED detection (on cleaned data)
    if cfg.ied in ("report", "exclude"):
        ied_events, ied_mask = detect_ied_spikes(
            xw, fs, z_thresh=cfg.ied_z, report_only=(cfg.ied == "report"))
        report["ied_count"] = int(len(ied_events))
        report["ied_frac"] = float(ied_mask.mean())
        if cfg.ied == "exclude":
            artifact_mask |= ied_mask
    report["ied"] = cfg.ied

    # 8 resampling
    fs_out = fs
    events_out = cfg.events
    if cfg.resample_to is not None:
        xw, fs_out = resample(xw, fs, cfg.resample_to)
        if events_out is not None:
            events_out = (np.asarray(events_out) * (fs_out / fs)).astype(int)
    report["fs"] = float(fs_out)

    # 9-10 epoching and QC
    report["epochs_kept"] = None
    report["qc_mask"] = None
    if events_out is not None:
        epochs, kept = make_epochs(xw, fs_out, events_out, cfg.tmin, cfg.tmax,
                                   cfg.baseline, artifact_mask,
                                   cond_labels=cfg.cond_labels,
                                   evoked_sub=cfg.evoked_sub)
        report["evoked_sub"] = cfg.evoked_sub
        qc = epoch_qc_mask(epochs, cfg.ptp_max, cfg.qc_var_z)
        report["epochs_total"] = len(kept)
        report["epochs_kept"] = int(qc.sum())
        report["qc_mask"] = qc.tolist()
        data_out = {"continuous": xw, "epochs": epochs, "kept_events": kept}
    else:
        data_out = {"continuous": xw}

    report["shape_out"] = list(xw.shape)
    return {"data": data_out, "report": report}


# ======================================================== QC metrics

def _psd_channel_median(x: np.ndarray, fs: float, nperseg: int | None = None):
    """Per-channel Welch PSDs, then the across-channel median in the linear
    domain (zero-variance channels excluded). Vectorized rfft implementation."""
    x = np.asarray(x, dtype=float)
    n_t = x.shape[-1]
    if nperseg is None or nperseg > n_t:
        nperseg = min(n_t, 2048)
    f = np.fft.rfftfreq(nperseg, 1.0 / fs)
    keep = x.std(axis=-1) > 0
    if not keep.any():
        return f, np.zeros(len(f))
    xg = x[keep]
    noverlap = nperseg // 2
    step = max(nperseg - noverlap, 1)
    n_seg = (n_t - nperseg) // step + 1
    win = np.hanning(nperseg)
    acc = np.zeros((xg.shape[0], len(f)))
    for s in range(n_seg):
        seg = xg[:, s * step:s * step + nperseg]
        seg = seg - seg.mean(axis=-1, keepdims=True)
        acc += np.abs(np.fft.rfft(seg * win, axis=-1)) ** 2
    scale = 2.0 / (fs * (win ** 2).sum() * n_seg)
    return f, np.median(acc * scale, axis=0)


def _line_peak_ratio(f: np.ndarray, p: np.ndarray, f0: float,
                     half: float = 0.5, nb: float = 2.5) -> float:
    """Line peak ratio: peak within +/-half Hz over the neighbor-band mean."""
    peak = float(p[(f >= f0 - half) & (f <= f0 + half)].max())
    nlo = (f >= f0 - nb - half) & (f < f0 - half)
    nhi = (f > f0 + half) & (f <= f0 + nb + half)
    base = float(np.concatenate([p[nlo], p[nhi]]).mean())
    return peak / (base + 1e-30)


def _zscore_rows(x: np.ndarray) -> np.ndarray:
    z = x - x.mean(axis=-1, keepdims=True)
    sd = z.std(axis=-1, keepdims=True)
    return z / (sd + 1e-12)


def _per_channel_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-channel Pearson r between aligned signals."""
    za, zb = _zscore_rows(a), _zscore_rows(b)
    return (za * zb).sum(axis=-1) / max(a.shape[-1], 1)


def _rank1_ev(x: np.ndarray) -> float:
    """Rank-1 variance explained (residual common-mode proxy)."""
    z = x - x.mean(axis=-1, keepdims=True)
    c = (z @ z.T) / max(z.shape[-1], 1)
    ev = np.linalg.eigvalsh(c)
    total = ev.sum()
    return float(ev[-1] / total) if total > 0 else 0.0


def _mean_abs_r(x: np.ndarray) -> float:
    """Mean absolute pairwise r (negative correlations count positive)."""
    return _pairwise_r(x)[1]


def _mean_r(x: np.ndarray) -> float:
    """Mean signed pairwise r; negative = spatial high-pass side effect."""
    return _pairwise_r(x)[0]


def _pairwise_r(x: np.ndarray) -> tuple[float, float]:
    z = _zscore_rows(x)
    c = (z @ z.T) / max(z.shape[-1], 1)
    c = np.clip(c, -1.0, 1.0)
    np.fill_diagonal(c, 0.0)
    return float(c.mean()), float(np.abs(c).mean())


def render_protocol_report(report: dict) -> str:
    """Render the protocol report as readable text (the reproducible archive)."""
    cfg = report.get("config", {})
    lines = []
    lines.append("=" * 72)
    lines.append("iEEG PREPROCESSING PROTOCOL REPORT")
    lines.append("=" * 72)
    lines.append(f"input : shape={report.get('input_shape')}  fs={report.get('fs_in')} Hz  (n_ch, n_t)")
    lines.append(f"output: shape={report.get('shape_out')}  fs={report.get('fs')} Hz")
    lines.append("")
    lines.append(f"[0] DC removal       : {cfg.get('dc_method')}")
    lines.append(f"[3] filter/baseline : {cfg.get('filter_method')}")
    if cfg.get("filter_method") not in (None, "none"):
        lines.append(f"    hp_cutoff={cfg.get('hp_cutoff')} Hz | lp_cutoff={cfg.get('lp_cutoff')} Hz | "
                     f"detrend_seg={cfg.get('detrend_seg')} s | detrend_order={cfg.get('detrend_order')}")
    lines.append(f"[1] bad channels    : {cfg.get('bad_detect')}"
                 + (f"  (>= {report.get('min_votes', 2)} votes)"
                    if str(cfg.get('bad_detect', '')).startswith('vote:') else ""))
    if report.get("bad_vote_detail"):
        lines.append("    vote detail     : "
                     + "; ".join(f"{k}={v}" for k, v in report["bad_vote_detail"].items()))
    lines.append(f"    manual (union)  : {cfg.get('manual_bad')}")
    lines.append(f"    final (union)   : {report.get('bad_channels')}")
    lines.append(f"    handling        : {cfg.get('bad_handle')}")
    lines.append(f"[2] stim artifacts  : {cfg.get('stim')}")
    if cfg.get("stim") not in (None, "none"):
        lines.append(f"    stim_pre={cfg.get('stim_pre')} s | stim_post={cfg.get('stim_post')} s")
    lines.append(f"[4] line noise      : {cfg.get('line_method')}")
    if cfg.get("line_method") not in (None, "none"):
        lines.append(f"    f0={cfg.get('line_f0')} Hz | q={cfg.get('line_q')} | "
                     f"width={cfg.get('line_width')} Hz | win={cfg.get('line_win')} s | "
                     f"harmonics={cfg.get('line_harmonics')} | "
                     f"extra_freqs={cfg.get('line_freqs')}")
    lines.append(f"[5] motion artifacts: {cfg.get('motion')}"
                 + (f"  (coverage={100 * float(report.get('motion_frac', 0.0)):.2f}%)"
                    if report.get("motion_frac") is not None else ""))
    if cfg.get("motion") not in (None, "none"):
        lines.append(f"    z={cfg.get('motion_z')} | win={cfg.get('motion_win')} s | "
                     f"lf={cfg.get('motion_lf')} Hz | max_gap={cfg.get('motion_max_gap')} s")
    if cfg.get("emg") not in (None, "none"):
        lines.append(f"[5] EMG artifacts    : {cfg.get('emg')}"
                     f"  (coverage={100 * float(report.get('emg_frac', 0.0)):.2f}%)")
        lines.append(f"    band={cfg.get('emg_lo')}-{cfg.get('emg_hi')} Hz | "
                     f"z={cfg.get('emg_z')} | win={cfg.get('emg_win')} s")
    if report.get("artifact_frac") is not None:
        lines.append(f"    combined artifact coverage = "
                     f"{100 * float(report.get('artifact_frac')):.2f}%")
    lines.append(f"[6] re-reference    : {cfg.get('reference')}")
    lines.append(f"    output channels : {report.get('n_ch_out')}")
    if report.get("laplacian_isolated"):
        lines.append(f"    laplacian isolated rows (CAR fallback): {report.get('laplacian_isolated')}")
    if report.get("out_ch_map") is not None:
        lines.append(f"    out_ch_map      : {report.get('out_ch_map')[:12]}"
                     + ("..." if len(report.get('out_ch_map', [])) > 12 else ""))
    lines.append(f"[7] IED detection   : {cfg.get('ied')}"
                 + (f"  (n={report.get('ied_count')}, "
                    f"coverage={100 * float(report.get('ied_frac', 0.0)):.2f}%)"
                    if cfg.get('ied') not in (None, 'none') else ""))
    lines.append(f"[8] resampling      : {cfg.get('resample_to') or 'none'}")
    lines.append(f"[9] epoching        : {'none' if cfg.get('n_events') == 0 else str(cfg.get('n_events')) + ' events'}"
                 f"  (tmin={cfg.get('tmin')}, tmax={cfg.get('tmax')}, baseline={cfg.get('baseline')}, "
                 f"evoked_sub={cfg.get('evoked_sub')}, n_cond_labels={cfg.get('n_cond_labels')})")
    if report.get("epochs_total") is not None:
        lines.append(f"[10] epoch QC       : {report.get('epochs_kept')}/{report.get('epochs_total')} kept"
                     f"  (ptp_max={cfg.get('ptp_max')}, var_z={cfg.get('qc_var_z')})")
    else:
        lines.append("[10] epoch QC       : skipped (no events)")
    lines.append("=" * 72)
    return "\n".join(lines)


def protocol_qc_metrics(x_raw: np.ndarray, x_clean: np.ndarray, fs: float,
                        bad_channels: list | None = None,
                        motion_mask: np.ndarray | None = None,
                        harmonics: int = 3, f0: float = 50.0,
                        ch_map: list | None = None) -> dict:
    """Before/after QC metrics; retention is aligned via ch_map so drop and
    bipolar outputs never compare mismatched rows."""
    x_raw = np.asarray(x_raw, dtype=float)
    x_clean = np.asarray(x_clean, dtype=float)
    nyq = fs / 2.0
    f, pr = _psd_channel_median(x_raw, fs)
    f, pc = _psd_channel_median(x_clean, fs)
    bands = [("delta", 0.5, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0),
             ("beta", 13.0, 30.0), ("gamma", 30.0, 80.0), ("high-gamma", 80.0, 150.0)]
    band_retention = {}
    for name, lo, hi in bands:
        hi_ = min(hi, nyq - 0.5)
        if hi_ <= lo:
            continue
        sel = (f >= lo) & (f <= hi_)
        band_retention[name] = float(pc[sel].mean() / (pr[sel].mean() + 1e-30))
    h_list = [h for h in range(1, harmonics + 1) if f0 * h < nyq - 1]
    lr_b = [_line_peak_ratio(f, pr, f0 * h) for h in h_list]
    lr_a = [_line_peak_ratio(f, pc, f0 * h) for h in h_list]
    ref = _aligned_reference(x_raw, ch_map) if ch_map is not None else x_raw
    nc = min(ref.shape[0], x_clean.shape[0])
    n_t = min(ref.shape[-1], x_clean.shape[-1])
    retention = _per_channel_corr(ref[:nc, :n_t], x_clean[:nc, :n_t])
    bad_rows = _output_rows_in_bad(ch_map, x_raw.shape[0],
                                   [] if bad_channels is None else bad_channels)
    retention = np.where(bad_rows, np.nan, retention)
    amp_before = np.median(np.abs(ref), axis=-1)
    return {
        "f": f,
        "psd_raw": pr,
        "psd_clean": pc,
        "band_retention": band_retention,
        "harmonics_hz": [f0 * h for h in h_list],
        "line_ratios_before": lr_b,
        "line_ratios_after": lr_a,
        "retention": retention,
        "rank1_before": _rank1_ev(x_raw),
        "rank1_after": _rank1_ev(x_clean),
        "mean_r_before": _mean_r(x_raw[:nc]),
        "mean_r_after": _mean_r(x_clean[:nc]),
        "mean_abs_r_before": _mean_abs_r(x_raw[:nc]),
        "mean_abs_r_after": _mean_abs_r(x_clean[:nc]),
        "amp_before": amp_before,
        "amp_after": np.median(np.abs(x_clean), axis=-1),
        "bad_channels": [] if bad_channels is None else [int(v) for v in bad_channels],
        "ch_map": ch_map,
        "motion_frac": float(np.asarray(motion_mask).mean()) if motion_mask is not None else None,
        "motion_mask": motion_mask,
    }


def _aligned_reference(x_raw: np.ndarray, ch_map: list) -> np.ndarray:
    """Rebuild the reference signal aligned to each output row via ch_map
    (int = original channel; [a, b] = bipolar pair a-b)."""
    x_raw = np.asarray(x_raw, dtype=float)
    rows = []
    for m in ch_map:
        if isinstance(m, (int, np.integer)):
            rows.append(x_raw[int(m)] if 0 <= int(m) < x_raw.shape[0]
                        else np.zeros(x_raw.shape[-1]))
        else:
            a, b = int(m[0]), int(m[1])
            if 0 <= a < x_raw.shape[0] and 0 <= b < x_raw.shape[0]:
                rows.append(x_raw[a] - x_raw[b])
            else:
                rows.append(np.zeros(x_raw.shape[-1]))
    return np.stack(rows) if rows else np.zeros((0, x_raw.shape[-1]))


def _output_rows_in_bad(ch_map: list | None, n_raw: int,
                        bad_channels: list) -> np.ndarray:
    if ch_map is None:
        flags = np.zeros(n_raw, dtype=bool)
        for b in bad_channels:
            if 0 <= b < n_raw:
                flags[b] = True
        return flags
    bad_set = set(int(b) for b in bad_channels)
    flags = []
    for m in ch_map:
        if isinstance(m, (int, np.integer)):
            flags.append(int(m) in bad_set)
        else:
            flags.append(int(m[0]) in bad_set or int(m[1]) in bad_set)
    return np.asarray(flags, dtype=bool)


# ============================================================ QC plots

def plot_protocol_qc(x_raw: np.ndarray, x_clean: np.ndarray, fs: float,
                     report: dict | None = None, motion_mask: np.ndarray | None = None,
                     n_panels: int = 9, figsize: tuple = (16, 12),
                     save_path: str | None = None):
    """Nine-panel before/after QC dashboard (English labels)."""
    import matplotlib.pyplot as plt
    bad = (report or {}).get("bad_channels", []) if report is not None else []
    if motion_mask is None and report is not None:
        motion_mask = report.get("motion_mask")
    ch_map = (report or {}).get("out_ch_map")
    m = protocol_qc_metrics(x_raw, x_clean, fs, bad_channels=bad,
                            motion_mask=motion_mask, ch_map=ch_map)
    if n_panels <= 6:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.ravel()
    else:
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        axes = axes.ravel()

    ax = axes[0]
    ax.semilogy(m["f"], m["psd_raw"], lw=1.2, label="raw", color="tab:blue")
    ax.semilogy(m["f"], m["psd_clean"], lw=1.2, label="after", color="tab:orange", alpha=0.85)
    ax.set_xlim(0, min(250.0, fs / 2 - 1))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD ($\\mu V^2$/Hz)")
    ax.set_title("1) PSD, median over channels (raw vs after)")
    ax.legend(fontsize=8)

    ax = axes[1]
    n = min(x_raw.shape[-1], int(2 * fs))
    t = np.arange(n) / fs
    ax.plot(t, x_raw[0, :n], lw=0.6, label="raw", color="tab:blue")
    ax.plot(t, x_clean[0, :n], lw=0.6, label="after", color="tab:orange", alpha=0.85)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude ($\\mu V$)")
    ax.set_title("2) Waveform, channel 0, first 2 s")
    ax.legend(fontsize=8)

    ax = axes[2]
    hz = np.asarray(m["harmonics_hz"], dtype=float)
    w = 0.35
    ax.bar(hz - w / 2, m["line_ratios_before"], w, label="raw", color="tab:blue")
    ax.bar(hz + w / 2, m["line_ratios_after"], w, label="after", color="tab:orange")
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Harmonic frequency (Hz)")
    ax.set_ylabel("Peak / neighbor-band mean")
    ax.set_title("3) Line-noise peak ratio (target = 1.0)")
    ax.legend(fontsize=8)

    ax = axes[3]
    ret = m["retention"][~np.isnan(m["retention"])]
    ax.hist(ret, bins=40, color="tab:green", alpha=0.7)
    if len(ret):
        ax.axvline(np.median(ret), color="k", ls="--", lw=1.2,
                   label=f"median={np.median(ret):.3f}")
    ax.set_xlabel("Pearson r(raw, clean) per channel")
    ax.set_ylabel("Channel count")
    ax.set_title("4) Signal retention per output row, ch-map aligned")
    ax.legend(fontsize=8)

    ax = axes[4]
    names = list(m["band_retention"].keys())
    vals = [m["band_retention"][k] for k in names]
    ax.bar(range(len(names)), vals, color="tab:cyan", alpha=0.8)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Power after / before")
    ax.set_title("5) Band power retention (1.0 = unchanged)")

    ax = axes[5]
    ax.bar([0, 1], [100 * m["rank1_before"], 100 * m["rank1_after"]],
           color=["tab:blue", "tab:orange"], alpha=0.85)
    for i, v in enumerate([m["rank1_before"], m["rank1_after"]]):
        ax.text(i, 100 * v + 1, f"{100 * v:.1f}%", ha="center", fontsize=9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["raw", "after"])
    ax.set_ylabel("Rank-1 variance explained (%)")
    ax.set_title("6) Residual common mode (lower = more removed)")

    ax = axes[6]
    ax.bar([0, 1], [m["mean_r_before"], m["mean_r_after"]],
           color=["tab:blue", "tab:orange"], alpha=0.85)
    ax.text(0, m["mean_r_before"] + 0.01, f"{m['mean_r_before']:.3f}", ha="center")
    ax.text(1, m["mean_r_after"] + 0.01, f"{m['mean_r_after']:.3f}", ha="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["raw", "after"])
    ax.set_ylabel("Mean pairwise r (signed)")
    ax.set_title("7) Spatial coherence (signed r; after <= 0 = decorrelated)")

    ax = axes[7]
    chs = np.arange(len(m["amp_before"]))
    ax.bar(chs, m["amp_before"], width=0.9, color="tab:blue", alpha=0.65, label="raw")
    ax.bar(chs, m["amp_after"][:len(chs)], width=0.9, color="tab:orange",
           alpha=0.65, label="after")
    bad_rows = _output_rows_in_bad(m["ch_map"], x_raw.shape[0], m["bad_channels"])
    if bad_rows.any():
        ax.bar(chs[bad_rows], m["amp_before"][bad_rows], width=1.0,
               color="red", alpha=0.9, label="bad")
    isolated = (report or {}).get("laplacian_isolated", [])
    if isolated:
        iso = np.asarray([i for i in isolated if 0 <= i < len(chs)], dtype=int)
        if len(iso):
            ax.bar(chs[iso], m["amp_before"][iso], width=1.0,
                   color="orange", alpha=0.9, label="isolated (CAR fallback)")
    ax.set_xlabel("Channel index")
    ax.set_ylabel("Median |amplitude| ($\\mu V$)")
    ax.set_title("8) Per-output-row amplitude (ch-map aligned; red = bad)")
    ax.legend(fontsize=8)

    ax = axes[8]
    n = min(x_raw.shape[-1], int(4 * fs))
    t = np.arange(n) / fs
    ax.plot(t, x_raw[0, :n], lw=0.5, color="tab:blue", label="raw")
    if m["motion_mask"] is not None and m["motion_mask"].sum() > 0:
        ax.fill_between(t, float(np.nanmin(x_raw[0, :n])),
                        float(np.nanmax(x_raw[0, :n])),
                        where=m["motion_mask"][:n], color="red", alpha=0.25,
                        label="motion mask")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no motion mask detected", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="gray")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude ($\\mu V$)")
    extra = []
    if (report or {}).get("emg_frac") is not None:
        extra.append(f"EMG={100 * report['emg_frac']:.2f}%")
    if (report or {}).get("ied_frac") is not None:
        extra.append(f"IED={100 * report['ied_frac']:.2f}%")
    ax.set_title(f"9) Motion mask coverage={100 * (m['motion_frac'] or 0):.2f}%"
                 + (" | " + " ".join(extra) if extra else "")
                 + f" | bad channels={m['bad_channels']}")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_config_comparison(results: dict, figsize: tuple = (16, 12),
                           save_path: str | None = None):
    """Nine-panel comparison across protocol configurations (English labels)."""
    import matplotlib.pyplot as plt
    names = list(results.keys())
    labels = [n.split(":")[0] for n in names]

    def _box(ax, key, title, ylabel, target=None, pct=False):
        data = [np.asarray(results[n][key]) for n in names]
        if pct:
            data = [100 * d for d in data]
        ax.boxplot(data, tick_labels=labels)
        if target is not None:
            ax.axhline(target, color="k", ls="--", lw=0.8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)

    fig, axes = plt.subplots(3, 3, figsize=figsize)
    _box(axes[0, 0], "line_ratio_after", "1) Line-noise peak ratio after (target = 1)",
         "Peak / neighbor-band mean", target=1.0)
    _box(axes[0, 1], "retention", "2) Signal retention r(raw, clean), higher = better",
         "Pearson r")
    _box(axes[0, 2], "gamma_retention", "3) Gamma power retention after/before (1 = unchanged)",
         "Ratio", target=1.0)
    _box(axes[1, 0], "theta_retention",
         "4) Theta power retention (drop matches spatial high-pass strength)",
         "Ratio", target=1.0)
    data = [100 * np.asarray(results[n]["common_mode_after"]) for n in names]
    axes[1, 1].boxplot(data, tick_labels=labels)
    ref = float(np.median(np.concatenate([100 * np.asarray(results[n]["common_mode_before"])
                                          for n in names])))
    axes[1, 1].axhline(ref, color="gray", ls="--", lw=0.8)
    axes[1, 1].text(0.02, 0.98, f"raw median = {ref:.1f}%",
                    transform=axes[1, 1].transAxes, va="top", fontsize=8, color="gray")
    axes[1, 1].set_title("5) Residual common mode after (rank-1 EV %, lower = better)")
    axes[1, 1].set_ylabel("Rank-1 variance explained (%)")
    _box(axes[1, 2], "mean_r_after",
         "6) Mean pairwise r after (signed; <= 0 = decorrelated)",
         "Mean pairwise r")
    _box(axes[2, 0], "bad_counts", "7) Bad channels flagged per trial", "Channel count")
    _box(axes[2, 1], "motion_fracs", "8) Motion mask coverage per trial (%)",
         "Coverage (%)", pct=True)
    axes[2, 2].bar(labels, [results[n]["runtime_s"] for n in names],
                   color="tab:gray", alpha=0.8)
    axes[2, 2].set_title("9) Runtime (s)")
    axes[2, 2].set_ylabel("Seconds")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


# ============================================ global bad lock + batch

def detect_bad_channels_global(X: np.ndarray, fs: float, method: str,
                               min_votes: int = 2, **kwargs) -> tuple[np.ndarray, dict]:
    """Global (session-wide) bad-channel detection: per-criterion statistics
    are aggregated across trials before a single vote, so every trial shares
    the same mask and outputs always stack into a consistent tensor."""
    X = np.asarray(X, dtype=float)
    trials = [X[k] for k in range(len(X))]
    n_ch = X.shape[1]
    detail = {}
    counts = np.zeros(n_ch, dtype=int)
    for name in method.split("+"):
        name = name.strip()
        if not name or name == "none":
            continue
        res = _global_criterion(trials, fs, name, kwargs)
        detail[name] = res
        for ch in res:
            counts[ch] += 1
    final = np.where(counts >= min_votes)[0]
    return final, detail


def _global_criterion(trials: list, fs: float, name: str,
                      kw: dict) -> np.ndarray:
    n_ch = trials[0].shape[0]
    if name == "mad":
        ch_mad = np.median(np.asarray([
            np.median(np.abs(t - np.median(t, axis=-1, keepdims=True)), axis=-1)
            for t in trials]), axis=0)
        log_std = np.log(ch_mad + 1e-12)
        med = np.median(log_std)
        mad = np.median(np.abs(log_std - med))
        z = (log_std - med) / (1.4826 * mad + 1e-12)
        return np.where(np.abs(z) > kw.get("n_mad", 5.0))[0]
    if name == "ptp":
        ptp = np.max(np.asarray([np.ptp(t, axis=-1) for t in trials]), axis=0)
        return np.where((ptp < kw.get("low", 1e-4)) | (ptp > kw.get("high", 1000.0)))[0]
    if name == "flatness":
        per_trial = []
        for t in trials:
            vals = []
            for i in range(n_ch):
                _f, p = sp_signal.welch(t[i], fs=fs, nperseg=min(t.shape[-1], int(fs * 2)))
                p = p[p > 0]
                vals.append(float(np.exp(np.log(p).mean()) / p.mean())
                            if len(p) >= 4 else 1.0)
            per_trial.append(vals)
        flat = np.median(np.asarray(per_trial), axis=0)
        return np.where(flat > kw.get("flat_thresh", 0.75))[0]
    if name == "corr":
        per_trial = []
        for t in trials:
            z = _zscore_rows(t)
            c = np.clip(z @ z.T / max(z.shape[-1], 1), -1.0, 1.0)
            np.fill_diagonal(c, 0.0)
            per_trial.append(np.abs(c).mean(axis=-1))
        mean_abs = np.median(np.asarray(per_trial), axis=0)
        return np.where((mean_abs < kw.get("corr_low", 0.02))
                        | (mean_abs > kw.get("corr_high", 0.999)))[0]
    if name == "ica":
        iv, rk = [], []
        for t in trials:
            rep = ica_report(t, fs, **kw)
            iv.append(rep["inv_vaf"])
            rk.append(rep["resid_kurt"])
        inv = np.median(np.asarray(iv), axis=0)
        med = np.median(inv)
        mad = np.median(np.abs(inv - med))
        iz = (inv - med) / (1.4826 * mad + 1e-12)
        resid_kurt = np.median(np.asarray(rk), axis=0)
        return np.where((iz > kw.get("inv_vaf_z", 4.0))
                        & (resid_kurt > kw.get("resid_kurt_min", 5.0)))[0]
    raise ValueError(f"unknown bad-channel criterion: {name}")


def batch_preprocess(X: np.ndarray, fs: float, cfg: PreprocessConfig,
                     run_kwargs: dict | None = None, verbose: bool = True) -> dict:
    """Batch preprocessing with a globally locked bad-channel set."""
    import time
    X = np.asarray(X, dtype=float)
    if cfg.bad_detect == "none":
        global_bad, detail = np.asarray([], dtype=int), {}
    elif cfg.bad_detect.startswith("vote:"):
        detect_kwargs = dict(cfg.detect_kwargs)
        min_votes = detect_kwargs.pop("min_votes", 2)
        global_bad, detail = detect_bad_channels_global(
            X, fs, cfg.bad_detect[5:],
            min_votes, **detect_kwargs)
    else:
        global_bad, detail = detect_bad_channels_global(
            X, fs, cfg.bad_detect, 1, **cfg.detect_kwargs)
    global_bad = np.unique(np.concatenate(
        [global_bad, np.asarray(cfg.manual_bad, dtype=int)]))
    outs, reports = [], []
    epochs_all, kept_all = [], []
    t0 = time.time()
    for k in range(len(X)):
        res = run_preprocessing(X[k], fs, cfg, force_bad=global_bad,
                                **(run_kwargs or {}))
        outs.append(res["data"]["continuous"])
        reports.append(res["report"])
        if "epochs" in res["data"]:
            epochs_all.append(res["data"]["epochs"])
            kept_all.append(res["data"]["kept_events"])
        if verbose and (k + 1) % 10 == 0:
            print(f"  batch: {k + 1}/{len(X)} trials, {time.time() - t0:.0f}s",
                  flush=True)
    X_clean = np.stack(outs)
    out = {
        "X_clean": X_clean,
        "reports": reports,
        "bad_channels": global_bad.tolist(),
        "bad_vote_detail": {k: v.tolist() for k, v in detail.items()},
        "out_ch_map": reports[0]["out_ch_map"] if reports else [],
        "fs_out": float(reports[0]["fs"]) if reports else float(fs),
        "shape_out": list(X_clean.shape),
    }
    if epochs_all:
        out["epochs"] = np.concatenate(epochs_all, axis=0)
        out["kept_events"] = [e for sub in kept_all for e in sub]
    return out


def evaluate_config(X: np.ndarray, fs: float, cfg: PreprocessConfig,
                    max_trials: int | None = None, verbose: bool = True,
                    run_kwargs: dict | None = None) -> dict:
    """Evaluate one protocol over trials with a globally locked bad-channel
    set; returns per-trial QC metrics plus the first trial's full report."""
    import time
    n = len(X) if max_trials is None else min(max_trials, len(X))
    if cfg.bad_detect == "none":
        global_bad = np.asarray([], dtype=int)
    elif cfg.bad_detect.startswith("vote:"):
        detect_kwargs = dict(cfg.detect_kwargs)
        min_votes = detect_kwargs.pop("min_votes", 2)
        global_bad, _ = detect_bad_channels_global(
            X, fs, cfg.bad_detect[5:],
            min_votes, **detect_kwargs)
    else:
        global_bad, _ = detect_bad_channels_global(
            X, fs, cfg.bad_detect, 1, **cfg.detect_kwargs)
    global_bad = np.unique(np.concatenate(
        [global_bad, np.asarray(cfg.manual_bad, dtype=int)]))
    bad_counts, motion_fracs = [], []
    bad_flags = []
    std_b, std_a = [], []
    lr_b, lr_a = [], []
    th_b, th_a = [], []
    ga_b, ga_a = [], []
    ret = []
    cm_b, cm_a = [], []
    mr_b, mr_a = [], []
    mar_b, mar_a = [], []
    first_report = None
    t0 = time.time()
    for k in range(n):
        x = X[k].astype(np.float64)
        res = run_preprocessing(x, fs, cfg, force_bad=global_bad,
                                **(run_kwargs or {}))
        xc = res["data"]["continuous"]
        rep = res["report"]
        if k == 0:
            first_report = rep
        bad_counts.append(len(rep["bad_channels"]))
        flag = np.zeros(X.shape[1], dtype=bool)
        flag[np.asarray(rep["bad_channels"], dtype=int)] = True
        bad_flags.append(flag)
        motion_fracs.append(float(rep.get("motion_frac", 0.0)))
        std_b.append(float(np.median(np.abs(x).mean(axis=-1))))
        std_a.append(float(np.median(np.abs(xc).mean(axis=-1))))
        f, p = _psd_channel_median(x, fs)
        th_b.append(float(p[(f >= 4) & (f <= 8)].mean()))
        ga_b.append(float(p[(f >= 30) & (f <= 80)].mean()))
        lr_b.append(_line_peak_ratio(f, p, 50.0))
        f, p = _psd_channel_median(xc, fs)
        th_a.append(float(p[(f >= 4) & (f <= 8)].mean()))
        ga_a.append(float(p[(f >= 30) & (f <= 80)].mean()))
        lr_a.append(_line_peak_ratio(f, p, 50.0))
        ch_map = rep.get("out_ch_map")
        ref = _aligned_reference(x, ch_map) if ch_map is not None else x
        nc = min(ref.shape[0], xc.shape[0])
        n_t = min(ref.shape[-1], xc.shape[-1])
        rr = _per_channel_corr(ref[:nc, :n_t], xc[:nc, :n_t])
        bad_rows = _output_rows_in_bad(ch_map, x.shape[0], rep["bad_channels"])
        rr = rr[~bad_rows]
        if len(rr):
            ret.append(float(np.median(rr)))
        cm_b.append(_rank1_ev(x))
        cm_a.append(_rank1_ev(xc))
        mr_b.append(_mean_r(x[:min(x.shape[0], xc.shape[0])]))
        mr_a.append(_mean_r(xc[:nc]))
        mar_b.append(_mean_abs_r(x[:min(x.shape[0], xc.shape[0])]))
        mar_a.append(_mean_abs_r(xc[:nc]))
        if verbose and (k + 1) % 50 == 0:
            print(f"  {cfg.reference}/{cfg.line_method}: {k + 1}/{n} trials, "
                  f"{time.time() - t0:.0f}s elapsed", flush=True)
    ga_b = np.asarray(ga_b)
    ga_a = np.asarray(ga_a)
    th_b = np.asarray(th_b)
    th_a = np.asarray(th_a)
    return {
        "bad_counts": np.asarray(bad_counts),
        "bad_flags": np.stack(bad_flags) if bad_flags else np.zeros((0, X.shape[1]), dtype=bool),
        "motion_fracs": np.asarray(motion_fracs),
        "std_median_before": np.asarray(std_b),
        "std_median_after": np.asarray(std_a),
        "line_ratio_before": np.asarray(lr_b),
        "line_ratio_after": np.asarray(lr_a),
        "theta_power_before": th_b,
        "theta_power_after": th_a,
        "gamma_power_before": ga_b,
        "gamma_power_after": ga_a,
        "theta_retention": th_a / (th_b + 1e-30),
        "gamma_retention": ga_a / (ga_b + 1e-30),
        "retention": np.asarray(ret),
        "common_mode_before": np.asarray(cm_b),
        "common_mode_after": np.asarray(cm_a),
        "mean_r_before": np.asarray(mr_b),
        "mean_r_after": np.asarray(mr_a),
        "mean_abs_r_before": np.asarray(mar_b),
        "mean_abs_r_after": np.asarray(mar_a),
        "first_report": first_report,
        "runtime_s": float(time.time() - t0),
    }
