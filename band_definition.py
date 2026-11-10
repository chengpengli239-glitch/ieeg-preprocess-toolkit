# -*- coding: utf-8 -*-
"""数据驱动的频段划分（对 EChc Star Protocols 方法的修订版）。

与 EChc 原版（P2_Freq_clustering_v02.m）的关键差异：
  1. 邻接约束聚类：只允许相邻频率 bin 合并，保证频段在频率轴上连续
     （原版对不连续簇按 min/max 截断会得到错误频段）；
  2. 频段数 k 自动选择：轮廓系数 + bootstrap 稳定性，替代原版的
     任意阈值（max(distance)*0.75 / cophenet*0.8）；
  3. 移除硬编码的">40 Hz 合并成一个频段"；
  4. 峰锚定频段作为第二条独立证据，与聚类结果互相对照；
  5. 支持按状态（发作前/发作中/发作后、任务条件）分别划分；
  6. 固定频段对照组始终输出，供下游特征/解码公平比较。

数据约定：x 为 (n_ch, n_t) float 数组，时间沿最后一维。
"""
from __future__ import annotations

import json

import numpy as np
from scipy import signal as sp_signal
from scipy.cluster import hierarchy as sp_hierarchy
from scipy.spatial.distance import pdist, squareform


def setup_matplotlib_cn() -> None:
    """绘图前的中文字体配置（缺字体会刷大量 Glyph 警告）。"""
    import matplotlib
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "DengXian", "Arial Unicode MS", "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False


FIXED_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 80.0),
}


# ---------------------------------------------------------------- 谱估计

def power_timecourses(x: np.ndarray, fs: float, fmin: float = 0.5,
                      fmax: float = 100.0, win_sec: float = 0.5,
                      hop_sec: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """短时傅里叶功率：返回 (freqs, times, power)，power 形状 (n_ch, n_freq, n_time)。

    win_sec 对应 EChc 的 max_timwin（恒定时间窗）；要更细的低频分辨率可调大。
    """
    x = np.asarray(x, dtype=float)
    nperseg = int(win_sec * fs)
    noverlap = nperseg - int(hop_sec * fs)
    f, t, z = sp_signal.stft(x, fs=fs, nperseg=nperseg, noverlap=noverlap,
                             window="hann", axis=-1)
    power = np.abs(z) ** 2
    sel = (f >= fmin) & (f <= fmax)
    return f[sel], t, power[:, sel, :]


def to_db_relative(power: np.ndarray, baseline_mask: np.ndarray | None = None,
                   baseline_span: tuple[float, float] | None = None,
                   times: np.ndarray | None = None) -> np.ndarray:
    """功率 -> 相对基线的 dB：10*log10(power / baseline_mean)。

    baseline_mask 优先；否则用 baseline_span（秒）在 times 上取窗。
    与 EChc 原代码一致（log10 比值）；如需 z 分数化另做。
    """
    power = np.asarray(power, dtype=float)
    if baseline_mask is None:
        if baseline_span is not None and times is not None:
            baseline_mask = (times >= baseline_span[0]) & (times <= baseline_span[1])
        else:
            baseline_mask = np.ones(power.shape[-1], dtype=bool)
    base = power[..., baseline_mask].mean(axis=-1, keepdims=True)
    return 10.0 * np.log10(power / (base + 1e-30))


def assemble_freq_matrix(power_db: np.ndarray, channels: list | None = None) -> np.ndarray:
    """把 (n_ch, n_freq, n_time) 的 dB 功率重排成 EChc 的 powspctrm3：
    (n_freq, n_ch * n_time)，行=频率，列=通道×时间拼接。
    """
    p = np.asarray(power_db, dtype=float)
    if channels is not None:
        p = p[np.asarray(channels, dtype=int)]
    n_ch, n_freq, n_time = p.shape
    return p.transpose(1, 0, 2).reshape(n_freq, n_ch * n_time)


def freq_corr_dist(M: np.ndarray) -> np.ndarray:
    """频率行之间的相关距离矩阵 (1 - Pearson r)，对称方阵。"""
    return squareform(pdist(M, metric="correlation"))


def contiguity_adjacency(n_freq: int, max_step: int = 1) -> object:
    """只允许在频率轴上相邻（或相距 <= max_step）的 bin 合并的连接矩阵。"""
    from scipy import sparse
    rows, cols = [], []
    for i in range(n_freq):
        for d in range(1, max_step + 1):
            if i - d >= 0:
                rows += [i, i - d]
                cols += [i - d, i]
            if i + d < n_freq:
                rows += [i, i + d]
                cols += [i + d, i]
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)),
                             shape=(n_freq, n_freq))


# ---------------------------------------------------------------- 聚类与选 k

def cluster_freqs(dist_corr: np.ndarray, k: int, connectivity: object | None = None,
                  linkage: str = "average", seed: int = 0) -> np.ndarray:
    """对相关距离矩阵做凝聚聚类，返回每个频率 bin 的簇标签。"""
    from sklearn.cluster import AgglomerativeClustering
    model = AgglomerativeClustering(n_clusters=int(k), metric="precomputed",
                                    linkage=linkage, connectivity=connectivity,
                                    compute_full_tree=True)
    return model.fit_predict(dist_corr)


def silhouette_of(dist_corr: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import silhouette_score
    if len(set(labels)) < 2:
        return float("nan")
    return float(silhouette_score(dist_corr, labels, metric="precomputed"))


def _base_labels(dist_corr, k, connectivity, linkage):
    return cluster_freqs(dist_corr, k, connectivity, linkage, seed=0)


def bootstrap_stability(M: np.ndarray, dist_corr: np.ndarray, k: int,
                        connectivity: object | None = None, linkage: str = "average",
                        n_boot: int = 15, seed: int = 0) -> float:
    """对列（通道×时间）做有放回重采样，重跑聚类，用调整兰德指数衡量频段划分稳定性。"""
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(seed)
    n_col = M.shape[1]
    base = _base_labels(dist_corr, k, connectivity, linkage)
    ari = []
    for _ in range(n_boot):
        cols = rng.integers(0, n_col, size=n_col)
        Mb = M[:, cols]
        db = freq_corr_dist(Mb)
        lab_b = cluster_freqs(db, k, connectivity, linkage, seed)
        ari.append(adjusted_rand_score(base, lab_b))
    return float(np.mean(ari))


def select_k(M: np.ndarray, dist_corr: np.ndarray, k_range: range,
             connectivity: object | None = None, linkage: str = "average",
             n_boot: int = 15, freqs: np.ndarray | None = None,
             min_band_hz: float = 6.0) -> dict:
    """在 k_range 上比较轮廓系数与稳定性，返回诊断表与最优 k。

    轮廓系数会偏爱碎片化划分；min_band_hz 约束"最窄频段至少多少 Hz"，
    在满足约束的候选中选轮廓系数最高者。
    """
    rows = []
    for k in k_range:
        labels = cluster_freqs(dist_corr, k, connectivity, linkage)
        sil = silhouette_of(dist_corr, labels)
        stab = bootstrap_stability(M, dist_corr, k, connectivity, linkage,
                                   n_boot=n_boot)
        min_w = np.inf
        if freqs is not None:
            df = float(np.median(np.diff(np.asarray(freqs, dtype=float))))
            for b in bands_from_labels(labels, freqs):
                min_w = min(min_w, b["high"] - b["low"] + df)
        rows.append({"k": k, "silhouette": sil, "stability": stab,
                     "min_band_hz": min_w, "labels": labels})
    valid = [r for r in rows if not np.isnan(r["silhouette"])]
    constrained = [r for r in valid if r["min_band_hz"] >= min_band_hz]
    pool = constrained if constrained else valid
    best = max(pool, key=lambda r: r["silhouette"])
    return {"rows": rows,
            "best_k": int(best["k"]),
            "best_labels": best["labels"]}


def bands_from_labels(labels: np.ndarray, freqs: np.ndarray) -> list[dict]:
    """把簇标签转成连续频段 [(lo, hi, center)]，按频率升序。"""
    freqs = np.asarray(freqs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    order = np.argsort(freqs)
    freqs, labels = freqs[order], labels[order]
    bands, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            lo, hi = float(freqs[start]), float(freqs[i - 1])
            bands.append({"low": lo, "high": hi,
                          "center": float(np.sqrt(lo * hi))})
            start = i
    return bands


def merge_narrow_bands(bands: list[dict], min_band_hz: float = 6.0) -> list[dict]:
    """把窄于 min_band_hz 的频段并入带宽更小的相邻频段（迭代至无窄带）。

    与 EChc 原版"所有 >40 Hz 合并成一段"的硬编码不同，这里只合并窄带，
    且并入哪一侧由相邻带宽决定（更窄的一侧更可能是被过度切分的碎带）。
    """
    bands = [dict(b) for b in bands]
    while len(bands) > 1:
        widths = [b["high"] - b["low"] for b in bands]
        i = int(np.argmin(widths))
        if widths[i] >= min_band_hz:
            break
        if i == 0:
            j = 1
        elif i == len(bands) - 1:
            j = i - 1
        else:
            j = i - 1 if widths[i - 1] <= widths[i + 1] else i + 1
        lo = min(bands[i]["low"], bands[j]["low"])
        hi = max(bands[i]["high"], bands[j]["high"])
        merged = {"low": lo, "high": hi, "center": float(np.sqrt(lo * hi))}
        bands[min(i, j)] = merged
        del bands[max(i, j)]
    return bands


def peak_anchored_bands(power_db: np.ndarray, freqs: np.ndarray,
                        min_prominence: float | None = None,
                        min_sep_hz: float = 1.5,
                        max_bands: int = 8) -> list[dict]:
    """峰锚定频段：平均谱找峰 + 跨通道投票，边界取相邻峰中点。

    显著性阈值自适应（max(min_prominence, 0.3 * 谱 std)），并统计"多少通道
    在该频率附近有峰"，投票比例低于 0.3 的峰丢弃——单一通道的伪峰不会定界。
    """
    spec = np.asarray(power_db).mean(axis=(0, -1))
    spec = np.nan_to_num(spec, nan=-np.inf, posinf=-np.inf)
    thresh = max(min_prominence or 0.0, 0.3 * float(np.nanstd(spec)))
    peaks, props = sp_signal.find_peaks(
        spec, prominence=thresh,
        distance=max(1, int(min_sep_hz / np.median(np.diff(freqs)))))
    if power_db.ndim == 3:
        n_ch = power_db.shape[0]
        votes = []
        for ch in range(n_ch):
            cs = np.nan_to_num(power_db[ch].mean(axis=-1), nan=-np.inf, posinf=-np.inf)
            ch_thresh = max(min_prominence or 0.0, 0.3 * float(np.nanstd(cs)))
            pk, _ = sp_signal.find_peaks(
                cs, prominence=ch_thresh,
                distance=max(1, int(min_sep_hz / np.median(np.diff(freqs)))))
            votes.append(set(pk.tolist()))
        keep = [i for i in peaks if sum(1 for v in votes if i in v) / n_ch >= 0.3]
        if keep:
            peaks = np.asarray(keep, dtype=int)
    if len(peaks) > max_bands:
        keep = np.argsort(props["prominences"])[::-1][:max_bands]
        peaks = np.sort(peaks[keep])
    if len(peaks) == 0:
        return [{"low": float(freqs[0]), "high": float(freqs[-1]),
                 "center": float(np.sqrt(freqs[0] * freqs[-1]))}]
    edges = [float(freqs[0])]
    for a, b in zip(peaks[:-1], peaks[1:]):
        edges.append(float(0.5 * (freqs[a] + freqs[b])))
    edges.append(float(freqs[-1]))
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi > lo:
            bands.append({"low": lo, "high": hi, "center": float(np.sqrt(lo * hi))})
    return bands


# ---------------------------------------------------------------- 编排与绘图

def define_bands(power_db: np.ndarray, freqs: np.ndarray, times: np.ndarray,
                 channels: list | None = None, k_range: range = range(2, 9),
                 linkage: str = "average", max_step: int = 1,
                 state_spans: dict | None = None,
                 peak_params: dict | None = None,
                 min_band_hz: float = 6.0) -> dict:
    """频段划分主入口。

    state_spans: 可选，{'pre': (t0,t1), 'ictal': (...), ...}（秒），
                 为每个状态分别聚类；None 时对全部时间做一次。
    """
    out = {"freqs": np.asarray(freqs, dtype=float),
           "fixed_bands": FIXED_BANDS}
    if state_spans:
        out["states"] = {}
        for name, (t0, t1) in state_spans.items():
            mask = (times >= t0) & (times <= t1)
            if mask.sum() == 0:
                continue
            sub = define_bands(power_db[..., mask], freqs, times[mask],
                               channels=channels, k_range=k_range,
                               linkage=linkage, max_step=max_step,
                               peak_params=peak_params, min_band_hz=min_band_hz)
            out["states"][name] = sub["bands"]
        out["bands"] = out["states"][list(out["states"].keys())[0]] if out["states"] else []
        out["peak_bands"] = peak_anchored_bands(power_db, freqs, **(peak_params or {}))
        out["diagnostics"] = {"per_state": {k: v for k, v in out["states"].items()}}
        return out
    M = assemble_freq_matrix(power_db, channels)
    dist = freq_corr_dist(M)
    conn = contiguity_adjacency(M.shape[0], max_step)
    sel = select_k(M, dist, k_range, conn, linkage, freqs=freqs,
                   min_band_hz=min_band_hz)
    raw_bands = bands_from_labels(sel["best_labels"], freqs)
    bands = merge_narrow_bands(raw_bands, min_band_hz)
    peaks = peak_anchored_bands(power_db, freqs, **(peak_params or {}))
    diag = {"best_k": sel["best_k"],
            "table": [{k: r[k] for k in ("k", "silhouette", "stability")}
                      for r in sel["rows"]],
            "raw_bands": raw_bands,
            "dist_corr": dist, "M": M, "connectivity": conn}
    out.update({"bands": bands, "peak_bands": peaks, "diagnostics": diag})
    return out


def plot_band_definition(result: dict, power_db: np.ndarray, freqs: np.ndarray,
                         save_path: str | None = None):
    """三面板：树状图 + 平均谱与频段着色 + k 诊断。"""
    import matplotlib.pyplot as plt
    setup_matplotlib_cn()
    dist = result["diagnostics"].get("dist_corr")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    if dist is not None:
        Z = sp_hierarchy.linkage(squareform(dist, checks=False), method="average")
        sp_hierarchy.dendrogram(Z, ax=axes[0], orientation="left",
                                labels=[f"{f:.1f}" for f in result["freqs"]],
                                color_threshold=0.55 * np.max(Z[:, 2]),
                                leaf_font_size=7)
    axes[0].set_title("频率层次聚类树（邻接约束）")
    axes[0].set_xlabel("相关距离")

    spec = np.asarray(power_db).mean(axis=(0, -1))
    axes[1].plot(result["freqs"], spec, color="k", lw=1.2)
    colors = plt.cm.tab10.colors
    for i, b in enumerate(result.get("bands", [])):
        sel = (result["freqs"] >= b["low"]) & (result["freqs"] <= b["high"])
        axes[1].fill_between(result["freqs"][sel], float(spec.min()), spec[sel],
                             color=colors[i % 10], alpha=0.25,
                             label=f"{b['low']:.1f}-{b['high']:.1f} Hz")
    for name, (lo, hi) in result["fixed_bands"].items():
        axes[1].axvline(lo, color="gray", ls=":", lw=0.8)
        axes[1].axvline(hi, color="gray", ls=":", lw=0.8)
    axes[1].set_xlabel("频率 (Hz)")
    axes[1].set_ylabel("相对功率 (dB)")
    axes[1].set_title("数据驱动频段 vs 固定频段边界（虚线）")
    axes[1].legend(fontsize=7)

    table = result["diagnostics"].get("table", [])
    if table:
        ks = [r["k"] for r in table]
        axes[2].plot(ks, [r["silhouette"] for r in table], "o-",
                     label="轮廓系数", color="tab:blue")
        axes[2].plot(ks, [r["stability"] for r in table], "s-",
                     label="bootstrap 稳定性", color="tab:orange")
        axes[2].axvline(result["diagnostics"]["best_k"], color="k", ls="--", lw=0.8)
        axes[2].set_xlabel("频段数 k")
        axes[2].set_title("k 选择诊断（竖虚线=所选 k）")
        axes[2].legend()

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_state_bands(result: dict, power_db: np.ndarray, freqs: np.ndarray,
                     times: np.ndarray, state_spans: dict,
                     save_path: str | None = None):
    """状态模式：每个状态一张谱图，叠加该状态的频段着色与固定频段虚线。"""
    import matplotlib.pyplot as plt
    setup_matplotlib_cn()
    names = [n for n in state_spans if n in result.get("states", {})]
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5), squeeze=False)
    colors = plt.cm.tab10.colors
    for ax, name in zip(axes[0], names):
        t0, t1 = state_spans[name]
        mask = (times >= t0) & (times <= t1)
        spec = np.asarray(power_db[..., mask]).mean(axis=(0, -1))
        ax.plot(freqs, spec, color="k", lw=1.2)
        for i, b in enumerate(result["states"][name]):
            sel = (freqs >= b["low"]) & (freqs <= b["high"])
            ax.fill_between(freqs[sel], float(spec.min()), spec[sel],
                            color=colors[i % 10], alpha=0.25,
                            label=f"{b['low']:.1f}-{b['high']:.1f}")
        for lo, hi in result["fixed_bands"].values():
            ax.axvline(lo, color="gray", ls=":", lw=0.7)
            ax.axvline(hi, color="gray", ls=":", lw=0.7)
        ax.set_title(f"{name} 频段划分")
        ax.set_xlabel("频率 (Hz)")
        ax.set_ylabel("相对功率 (dB)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    return fig


def bands_to_json(result: dict, path: str) -> None:
    """把频段定义落盘成 JSON，供下游特征提取引用。"""
    payload = {
        "fixed_bands": result["fixed_bands"],
        "bands": result.get("bands", []),
        "peak_bands": result.get("peak_bands", []),
        "states": result.get("states", {}),
        "best_k": result.get("diagnostics", {}).get("best_k"),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=float)
