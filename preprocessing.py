# -*- coding: utf-8 -*-
"""iEEG 预处理协议 v3：多方法可切换、可组合，含 sEEG 专属模块。

固定顺序（run_preprocessing）：
  0 去直流(dc_method, 可选) -> 1 坏通道 -> 2 刺激伪迹(先于一切时域滤波)
  -> 3 频率滤波(highpass/lowpass/bandpass/平滑去趋势, 在刺激伪迹之后)
  -> 4 工频去除(基频+谐波+自定义频率列表) -> 5 运动/EMG 伪迹掩码
  -> 6 重参考(8 种) -> 7 IED 检测(可选, report/exclude)
  -> 8 重采样(可选) -> 9 事件分段 -> 10 段级质检(含伪迹重叠剔除)

约定：x 为 (n_ch, n_t) float 数组，时间沿最后一维。

每一步的全部实现路径（配置项 -> 可选值）：
  0 dc_method: none | mean(去均值) | median(去中位数, 对刺激脉冲稳健)
  3 filter_method: none | highpass(hp_cutoff, 建议 0.1-1.0 Hz)
                   | lowpass(lp_cutoff, 建议 150-250 Hz 或 fs/2-1)
                   | bandpass(hp_cutoff-lp_cutoff)
                   | detrend(汉宁窗 50% 重叠滑动多项式去趋势,
                            detrend_seg/detrend_order, 无窗口边界阶跃)
  1 bad_detect: mad(Median Absolute Deviation, 双侧对数标准差)
                | ptp(Peak-to-Peak, 峰峰幅值双侧)
                | flatness(Spectral Flatness, 谱平坦度)
                | corr(Cross-channel Correlation, 跨通道相关性)
                | ica(ICA-based 1/VAF, 重构残差与伪迹成分负载)
                | 任意 '+' 组合取并集(任一报警即坏)
                | vote:mad+ptp+corr+ica(多数投票, min_votes 默认 2: 三取二、
                  四取三需显式传 min_votes=3)
    bad_handle: none | mask(置零) | interpolate(针道内邻居加权) | drop(剔除并重映射)
  2 stim: none | template_subtract(Template Subtraction, 刺激后窗中位数模板)
           | interpolate(Boundary Interpolation, 伪迹边界线性插值)
           | blank(Blanking, 置零，不推荐)
          参数: stim_pre(事件前边界, 默认 0.05s) / stim_post(伪迹窗口, 默认 0.15s)
  4 line_method: none | iir_notch(Infinite Impulse Response Notch, 零相位陷波)
                 | fir_notch(Finite Impulse Response Bandstop, 带阻陷波)
                 | cleanline(Spectral Regression, 滑窗谱回归, 抗相位漂移)
                 | fft_interp(FFT Interpolation Repair, 频谱插值修复, 无凹陷)
          参数: line_f0(基频, 默认 50Hz) / line_q(IIR 品质因数, 默认 30)
                / line_width(FIR 带宽, 默认 2Hz) / line_win(CleanLine 滑窗, 默认 4s)
                / line_harmonics(谐波数, 默认 1 = 只去 50Hz 本身)
                  / line_freqs(自定义干扰频率列表, 如 [60,65,78,140])
  5 motion: none | mask(MAD 稳健 z 分数慢漂移掩码，传播至分段质检，不修改波形)
            | interpolate(短段边界插值修复，长段仍掩码剔除)
            参数: motion_win(检测窗长, 默认 1s) / motion_z(z 阈值, 默认 4)
                  / motion_lf(慢漂移低通截止, 默认 2Hz)
                  / motion_max_gap(允许修复的最长段, 默认 0.5s)
     emg: none | mask(60-180Hz 肌电包络 MAD 稳健掩码, emg_lo/emg_hi/emg_z/emg_win)
  6 reference: none | car | median | group_car(分组均值, 多脑区/多栅格)
               | shaft_car(针道内均值) | white_matter(白质参考)
               | bipolar(针道内配对) | laplacian(邻接矩阵)
  7 ied: none | report(发作间期放电检测, 只报告) | exclude(报告并剔除重叠试次)
  8 resample_to: None | 任意目标采样率(多相重采样，自动同步事件)
  9 events/tmin/tmax/baseline: 事件触发分段 + 可选基线均值校正
  10 ptp_max/qc_var_z: 段级峰峰与方差 z 分数质检

  sEEG 专属：parse_seeg_shafts(针道解析)、detect_ied_spikes(发作间期放电检测，
  包络+一阶差分陡峭度联合判据，已接入 run_preprocessing)、
  detect_bad_correlation(松脱/桥接触点)、detect_emg_segments(高频肌电掩码)。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sp_signal


# ================================================================ 合成演示数据

def generate_demo_lfp(seed: int = 0, n_ch: int = 32, fs: float = 512.0,
                      duration: float = 120.0) -> dict:
    """生成带真实形态伪迹的 32 通道合成 LFP，用于检验预处理协议。"""
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


# ================================================================ 0 去直流/去趋势

def highpass(x: np.ndarray, fs: float, cutoff: float = 0.5, order: int = 4) -> np.ndarray:
    """零相位高通滤波（去直流与超低频极化漂移）。"""
    if cutoff <= 0 or cutoff >= fs / 2:
        raise ValueError(f"highpass cutoff {cutoff} 非法（fs={fs}）")
    sos = sp_signal.butter(order, cutoff / (fs / 2), btype="high", output="sos")
    return sp_signal.sosfiltfilt(sos, np.asarray(x, dtype=float), axis=-1)


def lowpass(x: np.ndarray, fs: float, cutoff: float = 250.0, order: int = 4) -> np.ndarray:
    """零相位低通滤波。cutoff 必须小于奈奎斯特频率（fs/2）。"""
    if cutoff <= 0 or cutoff >= fs / 2:
        raise ValueError(f"lowpass cutoff {cutoff} 非法（fs={fs}，须小于 {fs/2}）")
    sos = sp_signal.butter(order, cutoff / (fs / 2), btype="low", output="sos")
    return sp_signal.sosfiltfilt(sos, np.asarray(x, dtype=float), axis=-1)


def bandpass_filter(x: np.ndarray, fs: float, low: float, high: float,
                    order: int = 4) -> np.ndarray:
    """零相位带通滤波，低/高通截止均可自定义。"""
    if low <= 0 or high >= fs / 2 or low >= high:
        raise ValueError(f"bandpass 范围 {low}-{high} Hz 非法（fs={fs}）")
    sos = sp_signal.butter(order, [low / (fs / 2), high / (fs / 2)],
                           btype="band", output="sos")
    return sp_signal.sosfiltfilt(sos, np.asarray(x, dtype=float), axis=-1)


def detrend_linear(x: np.ndarray, fs: float, seg_sec: float = 10.0,
                   order: int = 1) -> np.ndarray:
    """汉宁窗 50% 重叠滑动多项式去趋势（无窗口边界阶跃）。

    旧实现把连续信号按 seg_sec 秒硬切成互不重叠的窗口，每个窗口独立最小二乘
    拟合后直接减去，相邻窗口的多项式系数不同，拼接边界必然产生电压跳变，
    这种人造方波阶跃会污染下游陷波与时频分析。
    新实现：窗口以 50% 步长滑动，每个窗口拟合的趋势按汉宁窗权重叠加，
    边界处权重趋近于 0，相邻窗口平滑过渡，不再产生阶跃伪迹。
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
    """去直流偏置。method: none | mean(减均值) | median(减中位数)。

    减中位数对刺激脉冲/瞬态伪迹更稳健；去直流必须在任何时域滤波与刺激伪迹
    处理之前完成，否则数百微伏的直流偏置会进入滤波器振铃与模板估计。
    """
    x = np.asarray(x, dtype=float).copy()
    if method in ("none", None):
        return x
    if method == "mean":
        return x - x.mean(axis=-1, keepdims=True)
    if method == "median":
        return x - np.median(x, axis=-1, keepdims=True)
    raise ValueError(f"未知去直流方法: {method}")


def apply_filter_baseline(x: np.ndarray, fs: float, method: str = "none",
                          hp_cutoff: float = 0.5, lp_cutoff: float = 250.0,
                          detrend_seg: float = 10.0,
                          detrend_order: int = 1) -> np.ndarray:
    """步骤 0 统一入口：滤波与基线处理。

    method:
      none      不做处理
      highpass  高通（去直流与超低频漂移），截止 hp_cutoff
      lowpass   低通（去高频，如研究低频动力学时），截止 lp_cutoff
      bandpass  带通 hp_cutoff-lp_cutoff
      detrend   分段多项式去趋势（不滤波），seg_sec/detrend_order 可调
    """
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
    raise ValueError(f"未知去直流方法: {method}")


# ================================================================ 1 坏通道

def detect_bad_mad(x: np.ndarray, n_mad: float = 5.0) -> np.ndarray:
    """双侧 MAD：对数标准差双侧检验，同时抓高噪道与坏死道。"""
    ch_std = np.median(np.abs(x - np.median(x, axis=-1, keepdims=True)), axis=-1)
    log_std = np.log(ch_std + 1e-12)
    med = np.median(log_std)
    mad = np.median(np.abs(log_std - med))
    z = (log_std - med) / (1.4826 * mad + 1e-12)
    return np.where(np.abs(z) > n_mad)[0]


def detect_bad_ptp(x: np.ndarray, low: float = 1e-4, high: float = 1000.0) -> np.ndarray:
    """峰峰幅值双侧判据：过低=坏死，过高=饱和。"""
    ptp = np.ptp(x, axis=-1)
    return np.where((ptp < low) | (ptp > high))[0]


def detect_bad_flatness(x: np.ndarray, fs: float, thresh: float = 0.75) -> np.ndarray:
    """谱平坦度判据：接近白噪声的通道可疑。"""
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


def detect_bad_correlation(x: np.ndarray, low: float = 0.02, high: float = 0.999) -> np.ndarray:
    """跨通道相关性判据：与所有其他通道平均 |r| 过低=脱接触点，过高=桥接短路。"""
    x = np.asarray(x, dtype=float)
    z = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-12)
    r = np.corrcoef(z)
    np.fill_diagonal(r, 0)
    mean_abs = np.abs(r).mean(axis=-1)
    return np.where((mean_abs < low) | (mean_abs > high))[0]


def detect_bad_channels(x: np.ndarray, fs: float, method: str = "mad", **kwargs) -> np.ndarray:
    """method 支持单判据、'+' 并集组合、'vote:' 多数投票、'ica'。

    - "mad+ptp"：并集（任何一个判据报警即坏）。
    - "vote:mad+ptp+corr"：多数投票（min_votes 个判据同时报警才算坏，
      默认 min_votes=2，即三取二）。
    - "ica"：ICA 重构 1/VAF 与伪迹成分负载判据。
    """
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
            results.append(detect_bad_ptp(x, kwargs.get("low", 1e-4), kwargs.get("high", 1000.0)))
        elif m == "flatness":
            results.append(detect_bad_flatness(x, fs, kwargs.get("flat_thresh", 0.75)))
        elif m == "corr":
            results.append(detect_bad_correlation(x, kwargs.get("corr_low", 0.02),
                                                  kwargs.get("corr_high", 0.999)))
        elif m == "none":
            return np.asarray([], dtype=int)
    return np.unique(np.concatenate(results)) if results else np.asarray([], dtype=int)


def detect_bad_vote(x: np.ndarray, fs: float, methods: list[str],
                    min_votes: int = 2, **kwargs) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """多数投票坏通道判定：每种方法独立输出结果，至少 min_votes 种方法报警才算坏。

    返回 (最终坏通道, {方法名: 该方法检出的坏通道})，每种方法的结果都保留供人工复核。
    """
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
    """ICA 分解与每通道重构质量报告（VAF 与 1/VAF）。

    VAF（方差解释率）= 1 - 残差方差/原始方差；1/VAF 表示"剩下的噪声是信号的多
    少倍"，越大说明该通道越难被公共成分解释（往往是独立噪声/伪迹通道）。
    同时返回各独立成分的时间峰度（kurtosis）与通道负载矩阵，供伪迹成分判定。
    成分数自适应：取 PCA 中解释累计方差 var_explained 所需的主成分数
    （限制在 20..n_ch 之间），避免高密度阵列（128/192 通道）用固定 20 成分
    导致正常局灶通道重构 VAF 系统性偏低、被大面积误判为坏道。
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
    S = ica.fit_transform(z.T)          # (n_t, n_c) 独立成分时间序列
    A = ica.mixing_                     # (n_ch, n_c) 通道负载
    rec = (S @ A.T).T                   # 重构的标准化信号 (n_ch, n_t)
    resid = z - rec
    vaf = 1.0 - resid.var(axis=-1)
    inv_vaf = 1.0 / np.clip(vaf, 1e-6, None)
    kurt = _kurtosis(S, axis=0)
    resid_kurt = _kurtosis(resid, axis=-1)
    return {"vaf": vaf, "inv_vaf": inv_vaf, "mixing": A, "sources": S,
            "kurtosis": kurt, "resid_kurt": resid_kurt, "n_components": n_c}


def _kurtosis(s: np.ndarray, axis: int = 0) -> np.ndarray:
    z = (s - s.mean(axis=axis, keepdims=True)) / (s.std(axis=axis, keepdims=True) + 1e-12)
    m4 = (z ** 4).mean(axis=axis)
    return np.asarray(m4)


def detect_bad_ica(x: np.ndarray, fs: float, inv_vaf_z: float = 4.0,
                   resid_kurt_min: float = 5.0, kurt_z: float = 3.0,
                   **kwargs) -> np.ndarray:
    """ICA 判据（稳健版）：重构残差的相对异常 + 残差峰度联合判据。

    为什么不用固定 1/VAF > 5.0：高密度 iEEG 局部特异度高，正常局灶通道的
    全局重构 VAF 天然偏低，固定绝对阈值会大面积误杀；而高幅伪迹通道在 PCA
    白化阶段霸占前几个成分、重构 VAF 反而接近 100%，固定阈值会漏检。
    新判据：1/VAF 以全体通道的中位数与 MAD 做稳健 z 分数，超过 inv_vaf_z 且
    残差时间序列峰度 > resid_kurt_min（独立尖刺噪声特征）才算坏；
    同时保留"对峰度极端成分负载最大的通道"这一补充判据。
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
    """处置坏通道。返回 (处理后 x, 与输出行对应的坏通道掩码)。
    drop 模式返回的掩码已压缩到剩余行（全 False），并建议通过
    compress_pairs/compress_adjacency 同步重映射双极对与邻接矩阵。
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
    raise ValueError(f"未知坏通道处置方式: {method}")


def compress_pairs(pairs: list[tuple[int, int]], bad: np.ndarray) -> list[tuple[int, int]]:
    """双极对：剔除涉及坏通道的对。"""
    bad_set = set(np.asarray(bad, dtype=int).tolist())
    return [(a, b) for a, b in pairs if a not in bad_set and b not in bad_set]


def compress_adjacency(adjacency: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """邻接矩阵：删除坏通道行列。"""
    keep = np.ones(adjacency.shape[0], dtype=bool)
    keep[np.asarray(bad, dtype=int)] = False
    return adjacency[np.ix_(keep, keep)]


# ================================================================ 2 刺激伪迹

def remove_stim_artifacts(x: np.ndarray, fs: float, onsets: np.ndarray,
                          method: str = "template_subtract",
                          pre: float = 0.05, post: float = 0.15) -> np.ndarray:
    """刺激伪迹处理。
    template_subtract: 对每个通道取各次刺激后窗(on..on+post)的中位数模板，从
    各自的刺激后窗减去——模板来自伪迹本身而不是基线。
    interpolate: 用伪迹前后边界的线性插值平滑过渡（短伪迹首选，避免阶跃振铃）。
    blank: 置零（会制造阶跃，不推荐用于连续信号）。
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
    raise ValueError(f"未知刺激伪迹方法: {method}")


# ================================================================ 3 工频去除

def _line_freq_list(fs: float, f0: float = 50.0, harmonics: int = 1,
                    freqs: list | None = None) -> np.ndarray:
    """工频列表 = 基频整数倍谐波 + 自定义干扰频率（去重、去奈奎斯特以上）。"""
    nyq = fs / 2 - 1
    out = [f0 * h for h in range(1, harmonics + 1) if f0 * h < nyq]
    for f in (freqs or []):
        if 0 < float(f) < nyq:
            out.append(float(f))
    return np.unique(np.asarray(out, dtype=float))


def notch_iir(x: np.ndarray, fs: float, f0: float = 50.0, q: float = 30.0,
              harmonics: int = 1, freqs: list | None = None) -> np.ndarray:
    """IIR 陷波（零相位双向），支持基频谐波 + 自定义多频点。"""
    out = np.asarray(x, dtype=float).copy()
    for f in _line_freq_list(fs, f0, harmonics, freqs):
        b, a = sp_signal.iirnotch(f, q, fs)
        out = sp_signal.filtfilt(b, a, out, axis=-1)
    return out


def notch_fir(x: np.ndarray, fs: float, f0: float = 50.0, width: float = 2.0,
              harmonics: int = 1, freqs: list | None = None,
              numtaps: int = 401) -> np.ndarray:
    """FIR 带阻陷波（零相位双向），支持基频谐波 + 自定义多频点。"""
    out = np.asarray(x, dtype=float).copy()
    for f in _line_freq_list(fs, f0, harmonics, freqs):
        b = sp_signal.firwin(numtaps, [f - width, f + width], fs=fs, pass_zero="bandstop")
        out = sp_signal.filtfilt(b, [1.0], out, axis=-1)
    return out


def notch_fft_interp(x: np.ndarray, fs: float, f0: float = 50.0,
                     width: float = 2.0, harmonics: int = 1,
                     freqs: list | None = None) -> np.ndarray:
    """FFT 插值修复去工频：只替换工频窄带内的频谱幅度，相位保留。

    与 IIR/FIR 陷波的关键区别：陷波器把窄带内的**全部功率**（含底噪）一并
    挖掉，对数坐标的 PSD 上会出现深谷；本方法用带边两侧的幅度线性插值填充
    该窄带，只去掉线频的正弦成分、保留底噪，频谱平滑无凹陷。
    实现上是逐通道、逐频率点对 rFFT 的幅度做带内插值后反变换，天然零相位。
    支持基频谐波 + 自定义多频点；相邻频点的插值带重叠时按顺序依次处理。
    """
    out = np.asarray(x, dtype=float).copy()
    n = out.shape[-1]
    freqs = np.fft.rfftfreq(n, 1 / fs)
    for ch in range(out.shape[0]):
        spec = np.fft.rfft(out[ch])
        mag = np.abs(spec)
        phase = np.angle(spec)
        for f in _line_freq_list(fs, f0, harmonics, freqs):
            band = (freqs >= f - width) & (freqs <= f + width)
            idx = np.where(band)[0]
            if len(idx) < 3:
                continue
            lo = max(0, idx[0] - 2)
            hi = min(len(freqs) - 1, idx[-1] + 2)
            # 取带外两侧的幅度值做带内线性插值
            outside = np.delete(np.arange(lo, hi + 1),
                                np.where(band[lo:hi + 1])[0])
            if len(outside) >= 2:
                mag[band] = np.interp(freqs[band],
                                      [freqs[outside[0]], freqs[outside[-1]]],
                                      [mag[outside[0]], mag[outside[-1]]])
        spec = mag * np.exp(1j * phase)
        out[ch] = np.fft.irfft(spec, n=n)
    return out


def cleanline_regression(x: np.ndarray, fs: float, f0: float = 50.0,
                         harmonics: int = 2, freqs: list | None = None,
                         win: float = 4.0) -> np.ndarray:
    """谱回归去工频：滑动窗分段拟合正弦/余弦基，重叠相加，适应相位漂移。

    支持基频谐波 + 自定义多频点；每个频率用 sin/cos 一对基拟合该段内的
    幅度与相位，对相位漂移的鲁棒性优于全时程固定相位拟合。
    """
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


def remove_line_noise(x: np.ndarray, fs: float, method: str = "iir_notch", **kwargs) -> np.ndarray:
    """method: iir_notch / fir_notch / cleanline / fft_interp / none。

    通用参数：f0(基频)、harmonics(谐波数)、freqs(自定义频率列表)、
    q(IIR)、width(FIR/插值带宽)、win(CleanLine 滑窗)。
    """
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
    raise ValueError(f"未知工频去除方法: {method}")


# ================================================================ 4 运动伪迹

def detect_motion_segments(x: np.ndarray, fs: float, win: float = 1.0,
                           z_thresh: float = 4.0, lf: float = 2.0) -> np.ndarray:
    """慢漂移能量检测。返回 (n_t,) 布尔掩码（供下游剔除重叠试次，不置零）。

    z 分数使用中位数与 MAD 而不是均值/标准差：一段幅度极大的剧烈伪迹会
    拉高均值与标准差，把阈值垫高、漏掉中等强度的伪迹段；MAD 不受离群
    窗口影响，检测灵敏度稳定。
    """
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


def detect_emg_segments(x: np.ndarray, fs: float, lo: float = 60.0, hi: float = 180.0,
                        win: float = 1.0, z_thresh: float = 5.0,
                        min_dur: float = 0.1) -> np.ndarray:
    """高频肌电（EMG）伪迹掩码：带通包络的窗 RMS 做 MAD 稳健 z 分数。

    颞叶底面/外侧的颅内电极易受咀嚼肌、颞肌收缩产生的 30-200 Hz 宽带肌电
    污染，以及电极微动产生的高频尖峰。这类伪迹不出现在低频慢漂移中，必须
    在 60-180 Hz 频带单独检测；掩码传播到分段质检剔除，保证 Gamma 分析纯净。
    返回 (n_t,) 布尔掩码。
    """
    x = np.asarray(x, dtype=float)
    if hi >= fs / 2 - 1:
        hi = fs / 2 - 2
    if lo <= 0 or hi <= lo:
        raise ValueError(f"EMG 频带 {lo}-{hi} Hz 非法（fs={fs}）")
    sos = sp_signal.butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos")
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
    # 丢弃短于 min_dur 的孤立碎段，降低误报
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
    """对掩码覆盖的**短**伪迹段做边界线性插值修复。

    超过 max_gap 秒的长段不修复（避免伪造大段脑信号），保持原样，
    继续由掩码传播到分段质检剔除。
    """
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
            continue                       # 长伪迹段不修复
        for ch in range(out.shape[0]):
            a = out[ch, s - 1] if s > 0 else out[ch, e]
            b = out[ch, e] if e < n else out[ch, s - 1]
            out[ch, s:e] = np.linspace(a, b, e - s)
    return out


# ================================================================ 5 重参考

def parse_seeg_shafts(ch_names: list[str]) -> list[np.ndarray]:
    """按电极名解析针道：'LA1'..'LA12' -> 针道 LA。返回各针道的通道索引。"""
    shafts: dict[str, list[int]] = {}
    for i, name in enumerate(ch_names):
        m = re.match(r"^([A-Za-z]+)[0-9]+$", str(name))
        key = m.group(1) if m else "?"
        shafts.setdefault(key, []).append(i)
    return [np.asarray(v, dtype=int) for v in shafts.values()]


def chain_bipolar_pairs(n_ch: int, shafts: list[np.ndarray] | None = None) -> list[tuple[int, int]]:
    """双极配对：给定针道时仅在针道内部配对。"""
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
    """针道内局部平均参考：每根针道独立减去本针道好通道均值。"""
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
    """分组共同平均参考：每个电极组（不同脑区/不同植入栅格）独立减去本组好通道均值。

    为什么：两块相距很远的电极栅格共模成分不同，全局 CAR 会把 A 组的共同活动
    混进 B 组；分组 CAR 分别去除各组的共模，适合多脑区/多植入物数据。
    """
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
    """白质参考：减去指定白质触点平均信号。"""
    wm = np.asarray(wm_channels, dtype=int)
    return np.asarray(x, dtype=float) - x[wm].mean(axis=0, keepdims=True)


def bipolar_reference(x: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    return np.stack([x[a] - x[b] for a, b in pairs], axis=0)


def laplacian_reference(x: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
    """拉普拉斯参考：减去邻居加权平均。"""
    x = np.asarray(x, dtype=float)
    A = np.asarray(adjacency, dtype=float)
    deg = A.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1.0
    return x - (A @ x) / deg


def apply_reference(x: np.ndarray, method: str = "none", bad_mask: np.ndarray | None = None,
                    pairs: list | None = None, adjacency: np.ndarray | None = None,
                    shafts: list[np.ndarray] | None = None,
                    wm_channels: list[int] | None = None,
                    groups: list[np.ndarray] | None = None) -> np.ndarray:
    """统一参考入口。bipolar/laplacian 会结合坏通道掩码规避坏触点。"""
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
            raise ValueError("shaft_car 参考需要提供 shafts")
        return shaft_car_reference(x, shafts, bad)
    if method == "group_car":
        if groups is None:
            raise ValueError("group_car 参考需要提供 groups（每组的通道索引列表）")
        return group_car_reference(x, groups, bad)
    if method == "white_matter":
        if wm_channels is None:
            raise ValueError("white_matter 参考需要提供 wm_channels")
        return white_matter_reference(x, wm_channels)
    if method == "bipolar":
        if pairs is None:
            raise ValueError("bipolar 参考需要提供 pairs")
        return bipolar_reference(x, compress_pairs(pairs, np.where(bad)[0]))
    if method == "laplacian":
        if adjacency is None:
            raise ValueError("laplacian 参考需要提供 adjacency")
        adj = np.asarray(adjacency, dtype=float).copy()
        adj[:, bad] = 0.0            # 切断坏通道向好通道的扩散
        deg = adj.sum(axis=1)
        out = np.asarray(laplacian_reference(x, adj), dtype=float)
        out[bad] = 0.0               # 坏通道自身的输出置零，与 mask 处置一致
        # 孤立好节点（四个邻居全为坏道）：laplacian_reference 内部把 deg=0
        # 改写成 1 会原样输出未参考的单极信号；这里退化为减去全体好通道均值
        isolated = (deg == 0) & (~bad)
        if isolated.any() and (~bad).any():
            mean_good = np.asarray(x, dtype=float)[~bad].mean(axis=0, keepdims=True)
            out[isolated] = np.asarray(x, dtype=float)[isolated] - mean_good
        return out
    raise ValueError(f"未知参考方式: {method}")


# ================================================================ 6 重采样

def resample(x: np.ndarray, fs: float, target_fs: float) -> tuple[np.ndarray, float]:
    """多相重采样。返回 (新数据, 新采样率)。"""
    from fractions import Fraction
    ratio = Fraction(target_fs / fs).limit_denominator(1000)
    out = sp_signal.resample_poly(np.asarray(x, dtype=float), ratio.numerator,
                                  ratio.denominator, axis=-1)
    return out, fs * ratio.numerator / ratio.denominator


# ================================================================ 7-8 分段与质检

def make_epochs(x: np.ndarray, fs: float, events: np.ndarray, tmin: float, tmax: float,
                baseline: tuple[float, float] | None = None,
                bad_time_mask: np.ndarray | None = None) -> tuple[np.ndarray, list[int]]:
    """事件触发分段；baseline 为 (t0, t1) 秒，做基线均值校正；
    bad_time_mask 为时间掩码，与伪迹重叠的试次直接剔除。"""
    events = np.asarray(events, dtype=int)
    n_pre, n_post = int(-tmin * fs), int(tmax * fs)
    epochs, kept = [], []
    for on in events:
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
    if epochs:
        return np.stack(epochs), kept
    return np.zeros((0, x.shape[0], n_pre + n_post)), kept


def epoch_qc_mask(epochs: np.ndarray, ptp_max: float = 800.0,
                  var_z: float = 4.0) -> np.ndarray:
    """段级质检：峰峰超阈或方差 z 分数超阈的片段剔除。"""
    ptp = np.ptp(epochs, axis=-1).max(axis=-1)
    var = epochs.var(axis=-1).mean(axis=-1)
    z = (var - var.mean()) / (var.std() + 1e-12)
    return (ptp < ptp_max) & (z < var_z)


# ================================================================ IED 检测

def detect_ied_spikes(x: np.ndarray, fs: float, hp: float = 20.0, lp: float = 80.0,
                      z_thresh: float = 5.0, min_dist: float = 0.05,
                      report_only: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """发作间期癫痫样放电（IED）检测：三判据联合。

    判据（逐通道）：
      1) 20-80 Hz 带通包络的 z 分数 > z_thresh；
      2) 带通信号幅值超均值 + 4 标准差；
      3) 一阶差分陡峭度超均值 + 5 标准差（棘波/尖波的陡峭上升沿）。
    三个判据同时满足才计数——第 3 条是棘波形态的核心特征，不能省。
    返回 (事件采样点, 事件时间掩码)。注意：IED 是癫痫研究的信号而非噪声，
    report_only=True 时只报告不剔除；report_only=False 时把掩码合并进
    伪迹掩码、在分段质检中剔除重叠试次（由 run_preprocessing 的 cfg.ied 控制）。
    """
    sos = sp_signal.butter(4, [hp / (fs / 2), lp / (fs / 2)], btype="band", output="sos")
    xb = sp_signal.sosfiltfilt(sos, np.asarray(x, dtype=float), axis=-1)
    env = np.abs(sp_signal.hilbert(xb, axis=-1))
    med = np.median(env, axis=-1, keepdims=True)
    mad = np.median(np.abs(env - med), axis=-1, keepdims=True)
    z = (env - med) / (1.4826 * mad + 1e-12)
    # 一阶差分用 prepend 对齐时间轴（长度保持 n_t），陡峭度判据真正参与联合
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


# ================================================================ 协议编排

@dataclass
class PreprocessConfig:
    dc_method: str = "none"             # none/mean/median；先于一切时域滤波与刺激伪迹
    filter_method: str = "none"        # none/highpass/lowpass/bandpass/detrend
    hp_cutoff: float = 0.5             # 高通/带通下限 (Hz)
    lp_cutoff: float = 250.0           # 低通/带通上限 (Hz)，须小于 fs/2
    detrend_seg: float = 10.0          # detrend 分段长度 (s)
    detrend_order: int = 1             # detrend 多项式阶数
    bad_detect: str = "mad+ptp"        # 见模块 docstring：'+' 并集，'vote:' 多数投票，'ica' ICA(1/VAF)
    manual_bad: list = field(default_factory=list)   # 手动追加的坏通道，与自动判据取并集
    bad_handle: str = "mask"           # none/mask/interpolate/drop
    stim: str = "none"                 # none/template_subtract/interpolate/blank
    stim_pre: float = 0.05             # 刺激伪迹：事件前边界 (s)
    stim_post: float = 0.15            # 刺激伪迹：伪迹窗口长度 (s)
    line_method: str = "iir_notch"     # iir_notch/fir_notch/cleanline/fft_interp/none
    line_f0: float = 50.0              # 工频基频 (Hz)
    line_q: float = 30.0               # IIR 陷波品质因数
    line_width: float = 2.0            # FIR 带阻半宽 (Hz)
    line_win: float = 4.0              # CleanLine 滑窗长度 (s)
    line_harmonics: int = 1
    line_freqs: list = field(default_factory=list)  # 自定义干扰频率列表 (Hz), 如 [60,65,78,140]
    motion: str = "none"               # none/mask(掩码传播)/interpolate(短段修复)
    motion_z: float = 4.0
    motion_win: float = 1.0            # 检测窗长 (s)
    motion_lf: float = 2.0             # 慢漂移低通截止 (Hz)
    motion_max_gap: float = 0.5        # 允许插值修复的最长伪迹段 (s)
    emg: str = "none"                  # none/mask(60-180Hz 肌电包络掩码)
    emg_z: float = 5.0
    emg_win: float = 1.0               # EMG 检测窗长 (s)
    emg_lo: float = 60.0               # EMG 频带下限 (Hz)
    emg_hi: float = 180.0              # EMG 频带上限 (Hz)
    ied: str = "none"                  # none/report/exclude(报告并剔除重叠试次)
    ied_z: float = 5.0                 # IED 包络 z 阈值
    reference: str = "none"            # none/car/median/shaft_car/white_matter/bipolar/laplacian
    resample_to: float | None = None
    events: np.ndarray | None = None
    tmin: float = -1.0
    tmax: float = 1.0
    baseline: tuple[float, float] | None = None
    ptp_max: float = 800.0
    qc_var_z: float = 4.0
    detect_kwargs: dict = field(default_factory=dict)


def run_preprocessing(x: np.ndarray, fs: float, cfg: PreprocessConfig,
                      bipolar_pairs: list | None = None,
                      adjacency: np.ndarray | None = None,
                      stim_onsets: np.ndarray | None = None,
                      ch_names: list[str] | None = None,
                      wm_channels: list[int] | None = None,
                      groups: list[np.ndarray] | None = None,
                      force_bad: np.ndarray | None = None) -> dict:
    """按固定顺序执行完整协议，返回 {'data': ..., 'report': ...}。

    固定顺序（防振铃 + 防阶跃）：
      0 去直流(dc_method) -> 1 坏通道 -> 2 刺激伪迹(先于一切时域滤波)
      -> 3 频率滤波 -> 4 工频 -> 5 运动/EMG 掩码 -> 6 重参考
      -> 7 IED(可选) -> 8 重采样 -> 9 分段 -> 10 段级质检

    force_bad：批处理用的全局锁定坏通道集合（跳过自动检测，仍与 manual_bad
    取并集）。drop 模式下 bipolar_pairs/groups/wm_channels/ch_names/adjacency
    全部按"旧行号->新行号"重映射，杜绝索引错位与越界。
    report 包含：config 全参数快照、bad_channels/bad_vote_detail、
    out_ch_map（输出行->原始通道的映射，供 QC 保留度对齐）、
    motion_frac/emg_frac/artifact_frac/ied_count、laplacian_isolated 等。
    用 render_protocol_report(report) 渲染成可读文本。
    """
    report: dict = {}
    xw = np.asarray(x, dtype=float).copy()
    n_ch_in = xw.shape[0]
    report["input_shape"] = list(xw.shape)
    report["fs_in"] = float(fs)
    # 完整参数快照：记录"对这段原始数据做了什么"，供后续特征提取与复现引用
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
        "tmin": cfg.tmin,
        "tmax": cfg.tmax,
        "baseline": None if cfg.baseline is None else [float(v) for v in cfg.baseline],
        "ptp_max": cfg.ptp_max,
        "qc_var_z": cfg.qc_var_z,
        "detect_kwargs": cfg.detect_kwargs,
    }
    shafts = parse_seeg_shafts(ch_names) if ch_names else None

    # 0 去直流（先于一切时域滤波与刺激伪迹处理，避免偏置进滤波器/模板）
    xw = remove_dc(xw, cfg.dc_method)
    report["dc_method"] = cfg.dc_method

    # 1 坏通道
    if force_bad is not None:
        bad = np.unique(np.asarray(force_bad, dtype=int))
        report["bad_detect_note"] = "locked by batch (force_bad)"
        report["bad_vote_detail"] = {}
        report["min_votes"] = None
    elif cfg.bad_detect.startswith("vote:"):
        bad, vote_detail = detect_bad_vote(
            xw, fs, cfg.bad_detect[5:].split("+"),
            cfg.detect_kwargs.get("min_votes", 2), **cfg.detect_kwargs)
        report["bad_vote_detail"] = {k: v.tolist() for k, v in vote_detail.items()}
        report["min_votes"] = cfg.detect_kwargs.get("min_votes", 2)
    else:
        bad = detect_bad_channels(xw, fs, cfg.bad_detect, **cfg.detect_kwargs)
    # 手动坏通道与自动判据结果取并集（绝不覆盖自动结果）
    bad = np.unique(np.concatenate([bad, np.asarray(cfg.manual_bad, dtype=int)]))
    bad = bad[(bad >= 0) & (bad < n_ch_in)]
    report["bad_channels"] = bad.tolist()
    xw, bad_mask = handle_bad_channels(xw, bad, cfg.bad_handle, shafts)
    report["bad_handle"] = cfg.bad_handle
    if cfg.bad_handle == "drop":
        # 旧行号 -> 新行号重映射：双极对/分组/白质触点/针道全部按新行号
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

    # 输出行 -> 原始通道索引（供 QC 保留度/幅值对齐）
    if cfg.bad_handle == "drop":
        ch_origin = np.asarray([i for i in range(n_ch_in)
                                if i not in set(bad.tolist())], dtype=int)
    else:
        ch_origin = np.arange(n_ch_in)

    # 2 刺激伪迹（先于一切时域滤波，避免振铃扩散；去直流之后）
    if cfg.stim != "none" and stim_onsets is not None:
        xw = remove_stim_artifacts(xw, fs, np.asarray(stim_onsets), cfg.stim,
                                   cfg.stim_pre, cfg.stim_post)
    report["stim_method"] = cfg.stim

    # 3 频率滤波（在刺激伪迹之后：脉冲先进插值/模板扣除，再进滤波器）
    xw = apply_filter_baseline(xw, fs, cfg.filter_method, cfg.hp_cutoff,
                               cfg.lp_cutoff, cfg.detrend_seg, cfg.detrend_order)
    report["filter_method"] = cfg.filter_method

    # 4 工频（基频+谐波+自定义频率列表）
    xw = remove_line_noise(xw, fs, cfg.line_method, f0=cfg.line_f0,
                           q=cfg.line_q, width=cfg.line_width,
                           win=cfg.line_win, harmonics=cfg.line_harmonics,
                           freqs=list(cfg.line_freqs))
    report["line_method"] = cfg.line_method

    # 5 运动 + EMG 伪迹：只生成掩码，不置零；合并为 artifact_mask
    artifact_mask = np.zeros(xw.shape[-1], dtype=bool)
    motion_mask = None
    if cfg.motion in ("mask", "interpolate"):
        motion_mask = detect_motion_segments(xw, fs, win=cfg.motion_win,
                                             z_thresh=cfg.motion_z, lf=cfg.motion_lf)
        report["motion_frac"] = float(motion_mask.mean())
        artifact_mask |= motion_mask
        # 掩码放进报告：绘图（第 9 张 QC 面板）与下游分段质检都要用它
        report["motion_mask"] = motion_mask
        if cfg.motion == "interpolate":
            xw = interpolate_masked_segments(xw, motion_mask, fs, cfg.motion_max_gap)
            report["motion_note"] = (f"短段(<= {cfg.motion_max_gap}s)已插值修复，"
                                     "长段保留并传播掩码剔除")
    report["motion"] = cfg.motion
    if cfg.emg == "mask":
        emg_mask = detect_emg_segments(xw, fs, lo=cfg.emg_lo, hi=cfg.emg_hi,
                                       win=cfg.emg_win, z_thresh=cfg.emg_z)
        report["emg_frac"] = float(emg_mask.mean())
        artifact_mask |= emg_mask
    report["artifact_frac"] = float(artifact_mask.mean())

    # 6 重参考（拉普拉斯的孤立节点信息记入报告）
    if cfg.reference == "laplacian" and adjacency is not None:
        adj0 = np.asarray(adjacency, dtype=float).copy()
        adj0[:, np.asarray(bad_mask)] = 0.0
        deg = adj0.sum(axis=1)
        report["laplacian_isolated"] = np.where((deg == 0) & (~np.asarray(bad_mask)))[0].tolist()
    xw = apply_reference(xw, cfg.reference, bad_mask,
                         bipolar_pairs, adjacency, shafts, wm_channels, groups)
    report["reference"] = cfg.reference
    report["n_ch_out"] = xw.shape[0]
    if cfg.reference == "bipolar" and bipolar_pairs is not None:
        report["out_ch_map"] = [[int(ch_origin[a]), int(ch_origin[b])]
                                for a, b in bipolar_pairs]
    else:
        report["out_ch_map"] = [int(ch_origin[i]) for i in range(xw.shape[0])]

    # 7 IED 检测（在重参考后的干净数据上；report 只报告，exclude 并入伪迹掩码）
    if cfg.ied in ("report", "exclude"):
        ied_events, ied_mask = detect_ied_spikes(
            xw, fs, z_thresh=cfg.ied_z, report_only=(cfg.ied == "report"))
        report["ied_count"] = int(len(ied_events))
        report["ied_frac"] = float(ied_mask.mean())
        if cfg.ied == "exclude":
            artifact_mask |= ied_mask
    report["ied"] = cfg.ied

    # 8 重采样（局部事件坐标，不修改 cfg）
    fs_out = fs
    events_out = cfg.events
    if cfg.resample_to is not None:
        xw, fs_out = resample(xw, fs, cfg.resample_to)
        if events_out is not None:
            events_out = (np.asarray(events_out) * (fs_out / fs)).astype(int)
    report["fs"] = float(fs_out)

    # 9-10 分段与质检（伪迹重叠试次剔除）
    report["epochs_kept"] = None
    report["qc_mask"] = None
    if events_out is not None:
        epochs, kept = make_epochs(xw, fs_out, events_out, cfg.tmin, cfg.tmax,
                                   cfg.baseline, artifact_mask)
        qc = epoch_qc_mask(epochs, cfg.ptp_max, cfg.qc_var_z)
        report["epochs_total"] = len(kept)
        report["epochs_kept"] = int(qc.sum())
        report["qc_mask"] = qc.tolist()
        data_out = {"continuous": xw, "epochs": epochs, "kept_events": kept}
    else:
        data_out = {"continuous": xw}

    report["shape_out"] = list(xw.shape)
    return {"data": data_out, "report": report}


# ================================================================ 协议评估

def _psd_channel_median(x: np.ndarray, fs: float, nperseg: int | None = None):
    """逐通道 PSD 的通道中位数（线性域，先剔除零方差坏通道）。

    为什么不用"空间中位数通道"：CAR 之后每个时刻的通道均值恒为 0，若数据
    对称/去相关，中位数通道也近似为 0，其 PSD 会整条塌缩，令"频段保留比"
    全变成 0（度量假象，不是真的把信号删光了）。
    为什么不用逐通道几何平均：对数域平均对接近零的通道过度敏感，一个被掩码
    置零或相位抵消的通道就能把整个频段的均值拖成 0。
    逐通道 PSD 的中位数（线性域）既稳健又不会塌缩，反映"典型通道"的功率。
    用向量化 rfft 实现 Welch 平均，比逐通道调用 scipy.welch 快一个数量级。
    """
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
    """工频峰-邻带比：中心 ±half Hz 内峰值 / 两侧 ±nb Hz 邻带均值。

    等于 1 表示没有孤立谱峰（完全去干净）；原始数据里有工频时通常 > 1。
    """
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
    """逐通道皮尔逊相关（处理前 vs 处理后），即信号保留度。"""
    za, zb = _zscore_rows(a), _zscore_rows(b)
    return (za * zb).sum(axis=-1) / max(a.shape[-1], 1)


def _rank1_ev(x: np.ndarray) -> float:
    """秩 1 方差解释率 = 最大特征值 / 全部特征值之和。

    它是"残余共模"的定量度量：所有通道共享同一个时间波形时可达 100%，
    CAR / 分组 CAR / 拉普拉斯的目的就是把这一项压下去。
    """
    z = x - x.mean(axis=-1, keepdims=True)
    c = (z @ z.T) / max(z.shape[-1], 1)
    ev = np.linalg.eigvalsh(c)
    total = ev.sum()
    return float(ev[-1] / total) if total > 0 else 0.0


def _mean_abs_r(x: np.ndarray) -> float:
    """通道间平均 |相关系数|。

    注意：取绝对值后，负相关也算正数。CAR/拉普拉斯给已去相关数据注入的
    负相关会被算成"相关升高"，判读时优先用带符号的 _mean_r。
    """
    return _pairwise_r(x)[1]


def _mean_r(x: np.ndarray) -> float:
    """通道间平均相关系数（带符号）。负值=空间高通注入的负相关。"""
    return _pairwise_r(x)[0]


def _pairwise_r(x: np.ndarray) -> tuple[float, float]:
    """返回 (通道间平均相关系数[带符号], 通道间平均 |r|)。"""
    z = _zscore_rows(x)
    # 手工计算相关系数并裁剪到 [-1,1]：全零坏通道（mask 处置）标准差为 0，
    # 用 np.corrcoef 会触发除零警告，这里直接得到 0 相关且无警告
    c = (z @ z.T) / max(z.shape[-1], 1)
    c = np.clip(c, -1.0, 1.0)
    np.fill_diagonal(c, 0.0)
    return float(c.mean()), float(np.abs(c).mean())


def render_protocol_report(report: dict) -> str:
    """把 run_preprocessing 的 report 渲染成可读的协议档案文本。

    用途：记录"这段原始数据经历了什么处理、用了哪些参数"，后续特征提取
    （03 及以后）直接引用这份档案即可保证可复现；这也是协议报告的语义：
    一份完整参数清单 + 每一步的处理结果摘要，而不是原始 JSON。
    """
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
                 f"  (tmin={cfg.get('tmin')}, tmax={cfg.get('tmax')}, baseline={cfg.get('baseline')})")
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
    """单次运行"处理前 vs 处理后"的 QC 指标集（plot_protocol_qc 的数据来源）。

    返回 dict：
      f / psd_raw / psd_clean          逐通道 PSD 的中位数（典型通道）
      band_retention                   各频段功率保留比 after/before（1=完全保留）
      line_ratios_before/after         各谐波(50/100/150Hz)峰-邻带比
       retention                        输出行 vs 对应原始信号的 Pearson r
                                        （经 ch_map 对齐；drop/双极不再错位）
      rank1_before/after               残余共模（秩 1 方差解释率，越低越好）
      mean_r_before/after              通道间平均相关系数（带符号，空间相干性）
      mean_abs_r_before/after          通道间平均 |r|（负数也计正，仅作参考）
      amp_before/after                 逐通道中位绝对幅值
      bad_channels / motion_frac / motion_mask
    ch_map: 输出行 -> 原始通道索引（int）或双极对 [a, b]（run_preprocessing
    的 report["out_ch_map"]）。None 时按行号一一对应（仅限无 drop/双极场景）。
    """
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

    # 输出行与原始信号的逐行对齐：drop 后行号前移、双极输出为差分对，
    # 直接按行号对齐会算出无物理意义的"保留度"；这里按 ch_map 重建参考信号
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
    """按 ch_map 重建与输出逐行对齐的参考信号矩阵。

    ch_map 元素为 int（单通道输出，对应原始通道索引）或 [a, b]（双极输出，
    参考信号 = 原始通道 a - b）。越界/缺失的映射用全零行占位（保留度记 NaN）。
    """
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
    """输出行是否涉及坏通道（用于把对应保留度记 NaN，避免误导）。"""
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


def plot_protocol_qc(x_raw: np.ndarray, x_clean: np.ndarray, fs: float,
                     report: dict | None = None, motion_mask: np.ndarray | None = None,
                     n_panels: int = 9, figsize: tuple = (16, 12),
                     save_path: str | None = None):
    """单次运行的 3x3 QC 面板（9 张图；n_panels=6 时用 2x3）。

    面板内容（全部英文标注，避免字体缺失）：
      1 PSD 前后叠加       2 时域波形前后叠加       3 工频峰比(50/100/150Hz)
      4 逐通道信号保留度   5 各频段功率保留比       6 残余共模(秩1 EV%)
      7 空间相干性(平均|r|) 8 逐通道幅值(坏道标红)   9 伪迹覆盖/坏道摘要
    save_path 非 None 时保存 PNG 并关闭图形。
    """
    import matplotlib.pyplot as plt
    bad = (report or {}).get("bad_channels", []) if report is not None else []
    # 运动掩码优先取显式参数；UI/notebook 没传时回退到报告里的 motion_mask
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

    # 1) PSD 前后叠加（通道中位数）
    ax = axes[0]
    ax.semilogy(m["f"], m["psd_raw"], lw=1.2, label="raw", color="tab:blue")
    ax.semilogy(m["f"], m["psd_clean"], lw=1.2, label="after", color="tab:orange", alpha=0.85)
    ax.set_xlim(0, min(250.0, fs / 2 - 1))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD ($\\mu V^2$/Hz)")
    ax.set_title("1) PSD, median over channels (raw vs after)")
    ax.legend(fontsize=8)

    # 2) 时域波形（通道 0 前 2 秒）
    ax = axes[1]
    n = min(x_raw.shape[-1], int(2 * fs))
    t = np.arange(n) / fs
    ax.plot(t, x_raw[0, :n], lw=0.6, label="raw", color="tab:blue")
    ax.plot(t, x_clean[0, :n], lw=0.6, label="after", color="tab:orange", alpha=0.85)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude ($\\mu V$)")
    ax.set_title("2) Waveform, channel 0, first 2 s")
    ax.legend(fontsize=8)

    # 3) 工频峰比（谐波 50/100/150 Hz）
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

    # 4) 逐通道信号保留度 r(raw, clean)
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

    # 5) 各频段功率保留比
    ax = axes[4]
    names = list(m["band_retention"].keys())
    vals = [m["band_retention"][k] for k in names]
    ax.bar(range(len(names)), vals, color="tab:cyan", alpha=0.8)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Power after / before")
    ax.set_title("5) Band power retention (1.0 = unchanged)")

    # 6) 残余共模：秩 1 方差解释率（前后对比）
    ax = axes[5]
    ax.bar([0, 1], [100 * m["rank1_before"], 100 * m["rank1_after"]],
           color=["tab:blue", "tab:orange"], alpha=0.85)
    for i, v in enumerate([m["rank1_before"], m["rank1_after"]]):
        ax.text(i, 100 * v + 1, f"{100 * v:.1f}%", ha="center", fontsize=9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["raw", "after"])
    ax.set_ylabel("Rank-1 variance explained (%)")
    ax.set_title("6) Residual common mode (lower = more removed)")

    # 7) 空间相干性：通道间平均相关系数（带符号）
    ax = axes[6]
    ax.bar([0, 1], [m["mean_r_before"], m["mean_r_after"]],
           color=["tab:blue", "tab:orange"], alpha=0.85)
    ax.text(0, m["mean_r_before"] + 0.01, f"{m['mean_r_before']:.3f}", ha="center")
    ax.text(1, m["mean_r_after"] + 0.01, f"{m['mean_r_after']:.3f}", ha="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["raw", "after"])
    ax.set_ylabel("Mean pairwise r (signed)")
    ax.set_title("7) Spatial coherence (signed r; after <= 0 = decorrelated)")

    # 8) 逐通道幅值（前后对比，坏道红色描边）
    ax = axes[7]
    chs = np.arange(len(m["amp_before"]))
    ax.bar(chs, m["amp_before"], width=0.9, color="tab:blue", alpha=0.65, label="raw")
    ax.bar(chs, m["amp_after"][:len(chs)], width=0.9, color="tab:orange",
           alpha=0.65, label="after")
    bad_rows = _output_rows_in_bad(m["ch_map"], x_raw.shape[0], m["bad_channels"])
    if bad_rows.any():
        ax.bar(chs[bad_rows], m["amp_before"][bad_rows], width=1.0,
               color="red", alpha=0.9, label="bad")
    ax.set_xlabel("Channel index")
    ax.set_ylabel("Median |amplitude| ($\\mu V$)")
    ax.set_title("8) Per-output-row amplitude (ch-map aligned; red = bad)")
    ax.legend(fontsize=8)

    # 9) 伪迹覆盖 / 坏通道摘要：始终绘制波形；有掩码则叠加红色阴影
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
    """多协议对比面板（3x3，全部英文标注）。

    results = {协议名: evaluate_config(...) 返回值}。
    面板：
      1 工频峰比 after      2 信号保留度 r        3 gamma 保留比
      4 theta 保留比        5 残余共模 after      6 通道间平均|r| after
      7 坏通道数/trial      8 运动覆盖/trial      9 耗时
    判读原则见各面板标题：峰比越接近 1、保留度越高、共模越低、坏道越集中越好。
    """
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
    ax = axes[0, 0]
    _box(ax, "line_ratio_after", "1) Line-noise peak ratio after (target = 1)",
         "Peak / neighbor-band mean", target=1.0)
    ax = axes[0, 1]
    _box(ax, "retention", "2) Signal retention r(raw, clean), higher = better",
         "Pearson r")
    ax = axes[0, 2]
    _box(ax, "gamma_retention", "3) Gamma power retention after/before (1 = unchanged)",
         "Ratio", target=1.0)
    ax = axes[1, 0]
    _box(ax, "theta_retention",
         "4) Theta power retention (drop matches spatial high-pass strength)",
         "Ratio", target=1.0)
    ax = axes[1, 1]
    data = [100 * np.asarray(results[n]["common_mode_after"]) for n in names]
    ax.boxplot(data, tick_labels=labels)
    ref = float(np.median(np.concatenate([100 * np.asarray(results[n]["common_mode_before"])
                                          for n in names])))
    ax.axhline(ref, color="gray", ls="--", lw=0.8)
    ax.text(0.02, 0.98, f"raw median = {ref:.1f}%", transform=ax.transAxes,
            va="top", fontsize=8, color="gray")
    ax.set_title("5) Residual common mode after (rank-1 EV %, lower = better)")
    ax.set_ylabel("Rank-1 variance explained (%)")
    ax = axes[1, 2]
    _box(ax, "mean_r_after",
         "6) Mean pairwise r after (signed; <= 0 = decorrelated)",
         "Mean pairwise r")
    ax = axes[2, 0]
    _box(ax, "bad_counts", "7) Bad channels flagged per trial", "Channel count")
    ax = axes[2, 1]
    _box(ax, "motion_fracs", "8) Motion mask coverage per trial (%)",
         "Coverage (%)", pct=True)
    ax = axes[2, 2]
    ax.bar(labels, [results[n]["runtime_s"] for n in names], color="tab:gray", alpha=0.8)
    ax.set_title("9) Runtime (s)")
    ax.set_ylabel("Seconds")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def detect_bad_channels_global(X: np.ndarray, fs: float, method: str,
                               min_votes: int = 2, **kwargs) -> tuple[np.ndarray, dict]:
    """多试次全局坏通道判定：逐 trial 统计量聚合后一次性锁定同一套通道。

    为什么必须全局锁定：逐 trial 独立检测会导致第 1 个试次检出 2 个坏道、
    第 2 个试次检出 5 个坏道；在 drop/双极模式下各试次输出维度不一致，
    无法拼接成 3 维张量供下游 SVM/dPCA。这里对每个判据先在全部试次上
    计算逐通道统计量（MAD/峰峰取跨 trial 中位数或最大值，谱平坦度、跨通道
    相关、ICA 残差取跨 trial 中位数），再做一次统一判定与多数投票。
    返回 (最终坏通道, {方法名: 该方法判定的坏通道})。
    """
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
    """单判据的全局（跨 trial 聚合）坏通道判定。"""
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
    raise ValueError(f"未知坏通道判据: {name}")


def batch_preprocess(X: np.ndarray, fs: float, cfg: PreprocessConfig,
                     run_kwargs: dict | None = None, verbose: bool = True) -> dict:
    """多试次批量预处理：坏通道在全会话上全局锁定，全部试次用同一掩码。

    返回 dict：
      X_clean     (n_trials, n_ch_out, n_t_out) 清洗后连续信号
      reports     逐试次完整报告
      bad_channels / bad_vote_detail  全局锁定的坏通道与各判据明细
      out_ch_map / fs_out / shape_out
      epochs / kept_events            若 cfg.events 非 None
    """
    import time
    X = np.asarray(X, dtype=float)
    if cfg.bad_detect == "none":
        global_bad, detail = np.asarray([], dtype=int), {}
    elif cfg.bad_detect.startswith("vote:"):
        global_bad, detail = detect_bad_channels_global(
            X, fs, cfg.bad_detect[5:],
            cfg.detect_kwargs.get("min_votes", 2), **cfg.detect_kwargs)
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
    """对 trial 矩阵 X(n_trials, n_ch, n_t) 逐 trial 运行协议并汇总 QC 指标。

    坏通道在全会话上全局锁定（同一套通道对所有 trial 生效），避免逐 trial
    独立检测造成 drop/双极模式下各试次维度不一致。

    返回 dict（新增字段用 [new] 标记）：
      bad_counts / bad_flags / motion_fracs       坏通道（全局锁定，恒定）与运动覆盖
      line_ratio_before/after                     50Hz 峰-邻带比
      theta_power_before/after、gamma_power_before/after（逐通道 PSD 中位数）
      theta_retention / gamma_retention [new]     after/before 功率比
      retention [new]                             逐 trial 信号保留度（ch_map 对齐）
      common_mode_before/after [new]              残余共模（秩 1 EV%）
      mean_r_before/after [new]                   通道间平均相关系数（带符号）
      mean_abs_r_before/after [new]               通道间平均 |r|（负数也计正，仅参考）
      std_median_before/after                     通道中位绝对幅值
      first_report [new]                          第 1 个 trial 的完整报告
      runtime_s
    run_kwargs: 透传给 run_preprocessing 的额外参数（如 bipolar_pairs / adjacency）。
    """
    import time
    n = len(X) if max_trials is None else min(max_trials, len(X))
    if cfg.bad_detect == "none":
        global_bad = np.asarray([], dtype=int)
    elif cfg.bad_detect.startswith("vote:"):
        global_bad, _ = detect_bad_channels_global(
            X, fs, cfg.bad_detect[5:],
            cfg.detect_kwargs.get("min_votes", 2), **cfg.detect_kwargs)
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
        # PSD 用逐通道中位数（线性域）：CAR 后中位数通道近似 0，会让频段
        # 保留比全部塌缩成 0（度量假象），逐通道中位数反映典型通道功率
        f, p = _psd_channel_median(x, fs)
        th_b.append(float(p[(f >= 4) & (f <= 8)].mean()))
        ga_b.append(float(p[(f >= 30) & (f <= 80)].mean()))
        lr_b.append(_line_peak_ratio(f, p, 50.0))
        f, p = _psd_channel_median(xc, fs)
        th_a.append(float(p[(f >= 4) & (f <= 8)].mean()))
        ga_a.append(float(p[(f >= 30) & (f <= 80)].mean()))
        lr_a.append(_line_peak_ratio(f, p, 50.0))
        # 信号保留度按 out_ch_map 对齐（drop/双极不会错位）；坏通道行记 NaN
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
