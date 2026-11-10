# -*- coding: utf-8 -*-
"""iEEG 预处理协议 UI（Tkinter 单文件，三栏布局）。

运行：conda activate bci_env && python preprocess_ui.py

布局：左栏 = 步骤 0-3；中栏 = 步骤 4-7；右栏 = 加载 / 运行 / 报告 / 导出。

执行顺序（与 preprocessing.py 一致，防振铃 + 防阶跃）：
  去直流 -> 坏通道 -> 刺激伪迹 -> 频率滤波 -> 工频 -> 运动/EMG 掩码
  -> 重参考 -> IED -> 重采样 -> 分段 -> 段质检

步骤 5 重参考的空间参数：
  - group_car / laplacian：加载电极排布表 (.xlsx)；表中首列若是针道名
    （如 "LA"），自动按针道断开 4 邻接，避免不同针道被错误连通
  - bipolar / shaft_car：需要加载通道名列表（.txt，每行一个，如 LA1..LA12）
  - white_matter：需要填写白质触点索引
"""
import json
import os
import sys
import threading
import tkinter as tk
from collections import deque
from tkinter import filedialog, messagebox, ttk

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing import (  # noqa: E402
    PreprocessConfig, run_preprocessing, evaluate_config, batch_preprocess,
    detect_bad_vote, parse_seeg_shafts, chain_bipolar_pairs,
    render_protocol_report, plot_protocol_qc,
)

DC_LABELS = {
    "none (不处理)": "none",
    "mean - Subtract Mean (减均值)": "mean",
    "median - Subtract Median (减中位数, 稳健)": "median",
}
BAD_METHOD_LABELS = {
    "mad - Median Absolute Deviation (双侧对数标准差)": "mad",
    "ptp - Peak-to-Peak (峰峰幅值)": "ptp",
    "flatness - Spectral Flatness (谱平坦度)": "flatness",
    "corr - Cross-channel Correlation (跨通道相关性)": "corr",
    "ica - ICA 1/VAF (重构残差判据)": "ica",
}
STIM_LABELS = {
    "template_subtract - Template Subtraction (模板相减)": "template_subtract",
    "interpolate - Boundary Interpolation (边界插值)": "interpolate",
    "blank - Blanking (置零, 不推荐)": "blank",
}
LINE_LABELS = {
    "iir_notch - IIR Notch (IIR 陷波, 零相位)": "iir_notch",
    "fir_notch - FIR Bandstop (FIR 带阻陷波)": "fir_notch",
    "cleanline - Spectral Regression (滑窗谱回归)": "cleanline",
    "fft_interp - FFT Interpolation Repair (频谱插值修复)": "fft_interp",
}
REF_LABELS = {
    "none (不重参考)": "none",
    "car - Common Average Reference (共同平均)": "car",
    "median - Median Reference (中位数参考)": "median",
    "group_car - Group Common Average (分组共同平均)": "group_car",
    "bipolar - Bipolar Reference (针道内双极)": "bipolar",
    "shaft_car - Shaft-local CAR (针道内均值)": "shaft_car",
    "white_matter - White Matter Reference (白质参考)": "white_matter",
    "laplacian - Laplacian Reference (拉普拉斯参考)": "laplacian",
}
IED_LABELS = {
    "none (不检测)": "none",
    "report - 只报告不剔除": "report",
    "exclude - 报告并剔除重叠试次": "exclude",
}

QC_GUIDE_TEXT = """QC = Quality Control（质量控制）。九张图是"处理前 vs 处理后"的对照体检，
用来回答两个问题：
  ① 这个组合有没有把该做的做干净（去工频、去共模、标坏道、标伪迹）；
  ② 有没有把不该动的神经信号改坏（保留度、频段、波形）。

比较方法：同一段数据分别用组合 A、B 各跑一次，逐张对照；
更好的一侧是"该掉的掉了、该留的留了"。以下逐张说明：

1) PSD 前后叠加（逐通道功率谱的中位数，即"典型通道"）
   raw 在 50/100/150 Hz（及自定义频点）有尖峰；after 应没有孤立尖峰、
   也没有异常深谷（IIR/FIR 的窄谷属正常，fft_interp 应完全平滑）。
   中高频 30-80 Hz 的 after 曲线不应整体坍塌。
   组合对比：工频峰去得干净、且 gamma 段基本不动者更优。

2) 波形前后叠加（通道 0 前 2 秒）
   after 应与 raw 形状一致，只少掉工频毛刺/伪迹；出现阶跃、振铃、
   突然漂移=某一步处理有误。
   组合对比：形状保留越好、伪迹消失越干净者更优。这是"没做坏"的底线检查。

3) 工频峰比（50/100/150 Hz，目标虚线=1）
   raw 柱很高（有孤立谱峰）；after 应贴 1，1=没有孤立峰。
   组合对比：after 柱越接近 1，工频去得越干净；附加频点也要贴 1。

4) 信号保留度直方图（输出行 r(参考信号, after)，按 out_ch_map 对齐）
   r 越集中在高值越好；大量低值通道=信号被误伤。drop/双极模式下行号已重映射，
   不会拿错位的通道算相关；坏通道行记 NaN、不参与直方图。
   注意：拉普拉斯/双极整体偏低是空间高通的固有属性，不是错误。
   组合对比：同一参考方式下 r 更高者更优；跨参考方式比较前先看 6/7 张。

5) 各频段功率保留比（after/before，虚线=1，按逐通道 PSD 中位数计算）
   delta/theta 在 CAR/高通/拉普拉斯下下降是预期；gamma/high-gamma 越接近 1 越好。
   组合对比：gamma 保留更高者更优；theta 下降幅度应与参考方式匹配
   （拉普拉斯 > 分组CAR ≈ CAR > 中位数 > none）。

6) 残余共模（秩1方差解释率 %）
   所有通道共享同一波形的占比；重参考前高、后低。
   组合对比：after 越低共模去得越干净，但要和第 5 张一起看——
   gamma 也被压没了就是过度处理。

7) 空间相干性（通道间平均相关系数，带符号）
   原始数据常有较强正值=容积传导/共模；重参考后应明显下降、接近 0 或转负。
   负值不代表"信号变坏"：CAR 会给已去相关的数据注入很小的负相关(-1/(n-1))，
   拉普拉斯/双极会给邻道注入更强的负相关(-0.5 量级)——这是空间高通的数学
   副作用，是"共模被拿掉"的另一种表现。
   组合对比：after 越接近 0 或越负，空间高通越强；下降幅度要匹配用途：
   做单通道解码/电极级特征 → 适度下降（分组CAR/CAR）；
   做行波、源定位、局灶分析 → 更强的空间高通（双极/拉普拉斯）。

8) 逐通道幅值（红=坏通道，行号按 out_ch_map 对齐）
   raw 与 after 的逐输出行中位幅值。坏道应红色标出且被处置
   （mask 后=0、插值后贴近邻道）；非坏通道的 after 曲线应整体平滑。
   非坏通道被整体压平=参考过强或步骤出错。
   组合对比：红色集中且少、非坏通道幅值平滑者更优。

9) 运动/EMG/IED 伪迹覆盖 / 坏通道摘要
   红色阴影=运动掩码段（后续分段质检会剔除这些时间）；
   灰色提示=该组合未检出运动伪迹。标题给出运动覆盖%、EMG/IED 覆盖%与坏通道列表。
   组合对比：覆盖%应与真实伪迹时长相当——整片标红说明阈值过低，
   完全没标说明阈值过高。

综合判断顺序：先看 3 与 2（工频干净、波形没坏）→ 再看 4/5（信号保留）→
用 6/7 确认参考方式的空间高通强度符合下游分析 → 最后用 8/9 复查坏道与伪迹。
"""


def parse_layout_xlsx(path: str, sheet: str) -> dict:
    """解析电极排布表，返回 {'groups','adjacency','shafts','n_ch','n_blocks'}。

    规则：
      - 若首列存在非数字单元格（针道标签，如 "LA"），每个标签 = 一根针道：
        4 邻接只在同一标签内构建，不同针道即使紧挨着也不会被横向连通，
        group_car 的组也按针道划分；
      - 无标签时，表内每个数字块（上下左右连通）视为一块电极（同旧行为）。
    通道号必须恰好覆盖 1..n_ch。
    """
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[sheet]
    grid = {}
    labels = {}
    first_col_labels = set()
    for row in ws.iter_rows():
        for cell in row:
            if cell.value is None:
                continue
            try:
                ch = int(cell.value)
            except (TypeError, ValueError):
                if cell.column == 1 and isinstance(cell.value, str):
                    first_col_labels.add(str(cell.value).strip())
                continue
            grid[(cell.row, cell.column)] = ch
    if first_col_labels:
        rows = sorted(set(r for r, _c in grid.keys()))
        for r in rows:
            lab = None
            for lr in sorted(set(cell.row for row in ws.iter_rows() for cell in row
                                 if cell.column == 1 and cell.value is not None)):
                if lr <= r:
                    lab = str(ws.cell(row=lr, column=1).value).strip()
                else:
                    break
            if lab is not None:
                for (rr, cc) in list(grid.keys()):
                    if rr == r:
                        labels[(rr, cc)] = lab
    n_ch = len(grid)
    expected = set(range(1, n_ch + 1))
    if set(grid.values()) != expected:
        raise ValueError(f"排布表通道号应恰好覆盖 1..{n_ch}，当前: {sorted(grid.values())}")

    adj = np.zeros((n_ch, n_ch))
    groups = []
    shafts = []
    if labels:
        for lab in sorted(set(labels.values())):
            members = [grid[c] - 1 for c in sorted(labels) if labels[c] == lab]
            groups.append(np.asarray(members, dtype=int))
            shafts.append(np.asarray(members, dtype=int))
            for (r, c) in sorted(labels):
                if labels[(r, c)] != lab:
                    continue
                a = grid[(r, c)] - 1
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nb = (r + dr, c + dc)
                    if nb in grid and labels.get(nb) == lab:
                        adj[a, grid[nb] - 1] = 1.0
    else:
        cells = sorted(grid.keys())
        blocks, seen = [], set()
        for start in cells:
            if start in seen:
                continue
            q = deque([start])
            seen.add(start)
            block = []
            while q:
                r, c = q.popleft()
                block.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nb = (r + dr, c + dc)
                    if nb in grid and nb not in seen:
                        seen.add(nb)
                        q.append(nb)
            blocks.append(block)
        for block in blocks:
            members = [grid[c] - 1 for c in block]
            groups.append(np.asarray(members, dtype=int))
            for (r, c) in block:
                a = grid[(r, c)] - 1
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nb = (r + dr, c + dc)
                    if nb in grid and nb in block:
                        adj[a, grid[nb] - 1] = 1.0
    return {"groups": groups, "adjacency": adj,
            "shafts": shafts if labels else None,
            "n_ch": n_ch, "n_blocks": len(groups)}


class PreprocessUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("iEEG 预处理协议 UI")
        self.geometry("1280x780")
        self.data = None
        self.fs = 500.0
        self.confirmed_bad = None
        self.layout = None
        self.layout_label_text = "未加载排布表"
        self.ch_names = None
        self.wm_channels = []
        self.last_result = None
        self.last_batch = None
        self._build()

    # ------------------------------------------------------------ 构件
    def _row(self, parent, label, values, default, width=30):
        f = ttk.Frame(parent); f.pack(fill="x", pady=2)
        ttk.Label(f, text=label, width=24).pack(side="left")
        var = tk.StringVar(value=default)
        cb = ttk.Combobox(f, textvariable=var, values=values, state="readonly", width=width)
        cb.pack(side="left")
        return var

    def _entry_row(self, parent, label, default, note):
        f = ttk.Frame(parent); f.pack(fill="x", pady=2)
        ttk.Label(f, text=label, width=24).pack(side="left")
        var = tk.StringVar(value=default)
        ttk.Entry(f, textvariable=var, width=10).pack(side="left")
        ttk.Label(f, text=note, foreground="gray").pack(side="left", padx=4)
        return var

    def _scrolled_column(self, parent, width):
        frame = ttk.Frame(parent, width=width)
        frame.pack(side="left", fill="both", expand=True)
        frame.pack_propagate(False)
        canvas = tk.Canvas(frame, width=width)
        vsb = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        return inner

    def _build(self):
        outer = ttk.Frame(self); outer.pack(fill="both", expand=True)
        left = self._scrolled_column(outer, 400)
        mid = self._scrolled_column(outer, 400)
        right = ttk.Frame(outer, width=440)
        right.pack(side="left", fill="y")
        right.pack_propagate(False)

        # ============ 左栏：步骤 0-3 ============
        ttk.Label(left, text="步骤 0 去直流 + 滤波").pack(anchor="w")
        self.dc = self._row(left, "去直流方法", list(DC_LABELS.keys()),
                            "median - Subtract Median (减中位数, 稳健)")
        self.do_f0 = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="频率滤波（在刺激伪迹之后自动执行）",
                        variable=self.do_f0).pack(anchor="w")
        self.f0 = self._row(left, "滤波方法", ["none", "highpass", "lowpass",
                                                 "bandpass", "detrend"], "highpass")
        self.hp = self._entry_row(left, "高通截止 (Hz)", "0.5", "建议 0.1-1.0")
        self.lp = self._entry_row(left, "低通截止 (Hz)", "250", "建议 150-250")
        self.ds = self._entry_row(left, "去趋势窗长 (s)", "10", "汉宁窗50%重叠, 建议 5-30")

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(left, text="步骤 1 坏通道（确认门控）").pack(anchor="w")
        self.do_bad = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, variable=self.do_bad).pack(anchor="w")
        self.bad_detect = self._row(left, "检测方法", list(BAD_METHOD_LABELS.keys())
                                    + ["vote:mad+ptp+corr (三取二)",
                                       "vote:mad+ptp+corr+ica (四取三)"],
                                    "vote:mad+ptp+corr (三取二)")
        self.bad_handle = self._row(left, "处置方式", ["mask (置零)", "interpolate (邻居插值)",
                                                        "drop (剔除并重映射)", "none"],
                                    "mask (置零)")
        self.manual_bad = self._entry_row(left, "手动建议坏通道", "", "与自动判据取并集")
        ttk.Button(left, text="预览坏通道检测", command=self.preview_bad).pack(anchor="w", pady=2)
        self.bad_box = tk.Text(left, height=3, width=46)
        self.bad_box.pack(fill="x")
        cf = ttk.Frame(left); cf.pack(fill="x", pady=2)
        ttk.Label(cf, text="最终坏通道:").pack(side="left")
        self.confirm_entry = ttk.Entry(cf, width=16)
        self.confirm_entry.pack(side="left", padx=4)
        self.confirm_btn = ttk.Button(cf, text="确认", command=self.confirm_bad)
        self.confirm_btn.pack(side="left", padx=4)
        self.confirm_label = ttk.Label(cf, text="未确认", foreground="red")
        self.confirm_label.pack(side="left")

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(left, text="步骤 2 刺激伪迹").pack(anchor="w")
        self.do_stim = tk.BooleanVar(value=False)
        self.stim_check = ttk.Checkbutton(left, variable=self.do_stim, state="disabled")
        self.stim_check.pack(anchor="w")
        self.stim = self._row(left, "方法", list(STIM_LABELS.keys()),
                              "interpolate - Boundary Interpolation (边界插值)")
        self.stim_pre = self._entry_row(left, "事件前边界 (s)", "0.05", "建议 0.02-0.10")
        self.stim_post = self._entry_row(left, "伪迹窗口 (s)", "0.15", "建议 0.05-0.50")

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(left, text="步骤 3 工频去除").pack(anchor="w")
        self.do_line = tk.BooleanVar(value=True)
        self.line_check = ttk.Checkbutton(left, variable=self.do_line, state="disabled")
        self.line_check.pack(anchor="w")
        self.line = self._row(left, "方法", list(LINE_LABELS.keys()),
                              "iir_notch - IIR Notch (IIR 陷波, 零相位)")
        self.line_f0 = self._entry_row(left, "基频 (Hz)", "50", "中国 50")
        self.line_q = self._entry_row(left, "IIR 品质因数 Q", "30", "建议 10-50")
        self.line_width = self._entry_row(left, "FIR 带宽 (Hz)", "2", "建议 1-5")
        self.line_win = self._entry_row(left, "CleanLine 滑窗 (s)", "4", "建议 2-8")
        self.line_harm = self._entry_row(left, "谐波数", "1", "1=只去基频")
        self.line_freqs = self._entry_row(left, "附加频率列表(Hz)", "", "逗号分隔, 如 60,65,78,140")

        # ============ 中栏：步骤 4-7 ============
        ttk.Label(mid, text="步骤 4 运动伪迹 (MAD 稳健 z)").pack(anchor="w")
        self.do_motion = tk.BooleanVar(value=False)
        self.motion_check = ttk.Checkbutton(mid, variable=self.do_motion, state="disabled")
        self.motion_check.pack(anchor="w")
        self.motion = self._row(mid, "方法",
                                ["mask - 掩码传播 (不修改波形)",
                                 "interpolate - 短段插值修复"],
                                "mask - 掩码传播 (不修改波形)")
        self.motion_z = self._entry_row(mid, "检测 z 阈值", "4", "建议 3-6")
        self.motion_win = self._entry_row(mid, "检测窗长 (s)", "1", "建议 0.5-2")
        self.motion_lf = self._entry_row(mid, "慢漂移低通 (Hz)", "2", "建议 1-3")
        self.motion_gap = self._entry_row(mid, "最长修复段 (s)", "0.5", "建议 0.2-1")

        ttk.Separator(mid, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(mid, text="步骤 4b 高频肌电 EMG (60-180Hz)").pack(anchor="w")
        self.do_emg = tk.BooleanVar(value=False)
        self.emg_check = ttk.Checkbutton(mid, variable=self.do_emg, state="disabled")
        self.emg_check.pack(anchor="w")
        self.emg_z = self._entry_row(mid, "EMG z 阈值", "5", "建议 4-8")
        self.emg_win = self._entry_row(mid, "EMG 窗长 (s)", "1", "建议 0.5-2")
        self.emg_lo = self._entry_row(mid, "EMG 频带下限 (Hz)", "60", "咀嚼/颞肌污染")
        self.emg_hi = self._entry_row(mid, "EMG 频带上限 (Hz)", "180", "上限 < 奈奎斯特")

        ttk.Separator(mid, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(mid, text="步骤 4c 发作间期放电 IED").pack(anchor="w")
        self.ied = self._row(mid, "方式", list(IED_LABELS.keys()), "none (不检测)")
        self.ied_z = self._entry_row(mid, "IED z 阈值", "5", "包络+幅值+差分三判据")

        ttk.Separator(mid, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(mid, text="步骤 5 重参考").pack(anchor="w")
        self.do_ref = tk.BooleanVar(value=True)
        self.ref_check = ttk.Checkbutton(mid, variable=self.do_ref, state="disabled")
        self.ref_check.pack(anchor="w")
        self.ref = self._row(mid, "方法", list(REF_LABELS.keys()),
                             "car - Common Average Reference (共同平均)")
        lf = ttk.Frame(mid); lf.pack(fill="x", pady=2)
        ttk.Button(lf, text="加载电极排布表 (.xlsx)", command=self.load_layout).pack(side="left", padx=4)
        self.layout_label = ttk.Label(lf, text=self.layout_label_text, foreground="gray")
        self.layout_label.pack(side="left")
        lf2 = ttk.Frame(mid); lf2.pack(fill="x", pady=2)
        ttk.Button(lf2, text="加载通道名 (.txt)", command=self.load_ch_names).pack(side="left", padx=4)
        self.ch_label = ttk.Label(lf2, text="未加载通道名", foreground="gray")
        self.ch_label.pack(side="left")
        self.wm_entry = self._entry_row(mid, "白质触点索引", "", "逗号分隔, 如 3,7,11")
        ttk.Label(mid, text="group_car/laplacian 需排布表（首列针道名会自动断开跨针道邻接）；"
                             "bipolar/shaft_car 需通道名；white_matter 需白质触点。").pack(anchor="w", pady=2)

        ttk.Separator(mid, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(mid, text="步骤 6 重采样").pack(anchor="w")
        self.do_resample = tk.BooleanVar(value=False)
        self.resample_check = ttk.Checkbutton(mid, variable=self.do_resample, state="disabled")
        self.resample_check.pack(anchor="w")
        self.resample_to = self._row(mid, "目标采样率(Hz)", ["256", "500", "1000"], "256")

        ttk.Separator(mid, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(mid, text="步骤 7 事件分段(演示固定间隔事件)").pack(anchor="w")
        self.do_epoch = tk.BooleanVar(value=False)
        self.epoch_check = ttk.Checkbutton(mid, variable=self.do_epoch, state="disabled")
        self.epoch_check.pack(anchor="w")

        # ============ 右栏：加载 / 运行 / 报告 / 导出 ============
        ttk.Button(right, text="加载数据 (.npy / .npz / demo)", command=self.load).pack(pady=6)
        self.file_label = ttk.Label(right, text="未加载", foreground="gray", wraplength=400)
        self.file_label.pack()
        ttk.Button(right, text="运行协议", command=self.run).pack(pady=6)
        ttk.Button(right, text="QC 图解读（1-9 逐张说明）",
                   command=self.show_qc_guide).pack(pady=2)
        self.status = ttk.Label(right, text="", foreground="blue")
        self.status.pack()
        exp = ttk.Frame(right); exp.pack(pady=4)
        self.export_btn = ttk.Button(exp, text="导出清洗结果 (.npz)", command=self.export_result,
                                     state="disabled")
        self.export_btn.pack(side="left", padx=3)
        self.batch_btn = ttk.Button(exp, text="全量批处理",
                                    command=self.run_batch, state="disabled")
        self.batch_btn.pack(side="left", padx=3)
        ttk.Label(right, text="协议报告：").pack(anchor="w", padx=8)
        self.report_box = tk.Text(right, height=28, width=50)
        self.report_box.pack(padx=8, fill="both", expand=True)

        # 门控：取消勾选步骤 1 时自动解锁后续步骤（修复死锁）
        self.do_bad.trace_add("write", self._on_bad_toggle)

    def _downstream_checks(self):
        return (self.stim_check, self.line_check, self.motion_check,
                self.emg_check, self.ref_check, self.resample_check,
                self.epoch_check)

    def _on_bad_toggle(self, *_):
        if not self.do_bad.get():
            for w in self._downstream_checks():
                w.config(state="normal")

    # ------------------------------------------------------------ 动作
    def load(self):
        path = filedialog.askopenfilename(filetypes=[("numpy/demo", "*.npz *.npy"), ("all", "*.*")])
        if not path:
            return
        if path.endswith(".npz"):
            z = np.load(path, allow_pickle=True)
            key = "X" if "X" in z.files else z.files[0]
            self.data = z[key].astype(np.float64)
            self.fs = float(z["fs"]) if "fs" in z.files else 500.0
        else:
            self.data = np.load(path).astype(np.float64)
        self.confirmed_bad = None
        self.last_result = None
        self.last_batch = None
        self.confirm_label.config(text="未确认", foreground="red")
        self.export_btn.config(state="disabled")
        self.batch_btn.config(state="disabled" if self.data.ndim != 3 else "normal")
        for w in self._downstream_checks():
            w.config(state="normal" if not self.do_bad.get() else "disabled")
        self.file_label.config(text=f"{os.path.basename(path)}\n{self.data.shape}  fs={self.fs}")

    def load_ch_names(self):
        path = filedialog.askopenfilename(filetypes=[("text", "*.txt"), ("all", "*.*")])
        if not path:
            return
        with open(path, encoding="utf-8", errors="ignore") as fh:
            self.ch_names = [ln.strip() for ln in fh if ln.strip()]
        self.ch_label.config(text=f"{os.path.basename(path)} ({len(self.ch_names)} 通道)",
                             foreground="green")

    def load_layout(self):
        path = filedialog.askopenfilename(filetypes=[("excel", "*.xlsx")])
        if not path:
            return
        import openpyxl
        wb = openpyxl.load_workbook(path, data_only=True)
        sheets = wb.sheetnames
        if not sheets:
            messagebox.showwarning("提示", "表格没有工作表")
            return
        sheet = sheets[0] if len(sheets) == 1 else self._ask_sheet(sheets)
        try:
            self.layout = parse_layout_xlsx(path, sheet)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("排布表解析失败", str(exc))
            return
        extra = (f", 针道模式 {len(self.layout['shafts'])} 根"
                 if self.layout.get("shafts") else "")
        self.layout_label_text = (f"{os.path.basename(path)} [{sheet}] "
                                  f"{self.layout['n_ch']} 通道, "
                                  f"{self.layout['n_blocks']} 组{extra}")
        self.layout_label.config(text=self.layout_label_text, foreground="green")

    def show_qc_guide(self):
        win = tk.Toplevel(self)
        win.title("QC 图解读（1-9 逐张说明）")
        win.geometry("860x680")
        txt = tk.Text(win, wrap="word", font=("Microsoft YaHei", 10))
        txt.insert("1.0", QC_GUIDE_TEXT)
        txt.config(state="disabled")
        sb = ttk.Scrollbar(win, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        txt.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    def _ask_sheet(self, sheets):
        win = tk.Toplevel(self)
        win.title("选择工作表")
        var = tk.StringVar(value=sheets[0])
        ttk.Combobox(win, textvariable=var, values=sheets, state="readonly").pack(padx=12, pady=8)
        ttk.Button(win, text="确定", command=win.destroy).pack(pady=6)
        self.wait_window(win)
        return var.get()

    def _selected_bad_methods(self):
        sel = self.bad_detect.get()
        if sel.startswith("vote:"):
            inner = sel[5:].split(" (")[0].split("+")
            return inner, (3 if "四取三" in sel else 2)
        return [BAD_METHOD_LABELS[sel]], 1

    def preview_bad(self):
        if self.data is None:
            messagebox.showwarning("提示", "请先加载数据")
            return
        methods, min_votes = self._selected_bad_methods()
        x = self.data[0].astype(np.float64) if self.data.ndim == 3 else self.data
        final, detail = detect_bad_vote(x, self.fs, methods, min_votes=min_votes)
        lines = [f"{m}: {len(res)} 个 -> {res.tolist()}" for m, res in detail.items()]
        lines.append(f"投票结果 (>= {min_votes} 票): {final.tolist()}")
        self.bad_box.delete("1.0", "end")
        self.bad_box.insert("1.0", "\n".join(lines))
        self.confirm_entry.delete(0, "end")
        self.confirm_entry.insert(0, ",".join(str(c) for c in final))

    def confirm_bad(self):
        text = self.confirm_entry.get().replace("，", ",").strip()
        self.confirmed_bad = [] if not text else [int(v) for v in text.split(",") if v.strip()]
        self.confirm_label.config(text=f"已确认: {self.confirmed_bad}", foreground="green")
        for w in self._downstream_checks():
            w.config(state="normal")

    # ------------------------------------------------------------ 配置组装
    def _build_cfg_and_kwargs(self):
        ref = REF_LABELS[self.ref.get()] if self.do_ref.get() else "none"
        run_kwargs = {}
        shafts = None
        n_ch_now = self.data.shape[-2]
        if self.ch_names is not None and len(self.ch_names) == n_ch_now:
            shafts = parse_seeg_shafts(self.ch_names)
        elif self.layout is not None and self.layout.get("shafts"):
            shafts = self.layout["shafts"]
        if ref == "group_car":
            if self.layout is None:
                messagebox.showwarning("提示", "group_car 需要先加载电极排布表")
                return None, None
            run_kwargs["groups"] = self.layout["groups"]
        elif ref == "laplacian":
            if self.layout is None:
                messagebox.showwarning("提示", "laplacian 需要先加载电极排布表")
                return None, None
            run_kwargs["adjacency"] = self.layout["adjacency"]
        elif ref == "bipolar":
            if shafts is None:
                messagebox.showwarning(
                    "提示", "bipolar 需要通道名列表或带针道标签的排布表；"
                            "否则将按数组顺序配对（解剖上不可靠）")
            run_kwargs["bipolar_pairs"] = chain_bipolar_pairs(n_ch_now, shafts)
        elif ref == "shaft_car":
            if shafts is None:
                messagebox.showwarning("提示", "shaft_car 需要通道名列表或带针道标签的排布表")
                return None, None
            run_kwargs["shafts"] = shafts
        elif ref == "white_matter":
            wm = [int(v) for v in self.wm_entry.get().replace("，", ",").split(",") if v.strip()]
            if not wm:
                messagebox.showwarning("提示", "white_matter 需要填写白质触点索引")
                return None, None
            self.wm_channels = wm
            run_kwargs["wm_channels"] = wm
        if self.ch_names is not None:
            run_kwargs["ch_names"] = self.ch_names

        methods, min_votes = self._selected_bad_methods()
        line_freqs = [float(v) for v in
                      self.line_freqs.get().replace("，", ",").replace(" ", "").split(",")
                      if v.strip()]
        cfg = PreprocessConfig(
            dc_method=DC_LABELS[self.dc.get()],
            filter_method=self.f0.get() if self.do_f0.get() else "none",
            hp_cutoff=float(self.hp.get()),
            lp_cutoff=float(self.lp.get()),
            detrend_seg=float(self.ds.get()),
            # 勾选步骤 1 = 使用确认名单（全会话统一）；取消 = 完全不做坏通道处理
            bad_detect="none",
            manual_bad=(self.confirmed_bad or []) if self.do_bad.get() else [],
            bad_handle=(self.bad_handle.get().split(" ")[0]
                        if self.do_bad.get() else "none"),
            stim=STIM_LABELS[self.stim.get()] if self.do_stim.get() else "none",
            stim_pre=float(self.stim_pre.get()),
            stim_post=float(self.stim_post.get()),
            line_method=LINE_LABELS[self.line.get()] if self.do_line.get() else "none",
            line_f0=float(self.line_f0.get()),
            line_q=float(self.line_q.get()),
            line_width=float(self.line_width.get()),
            line_win=float(self.line_win.get()),
            line_harmonics=int(self.line_harm.get()),
            line_freqs=line_freqs,
            motion=("interpolate" if "interpolate" in self.motion.get() else "mask")
                   if self.do_motion.get() else "none",
            motion_z=float(self.motion_z.get()),
            motion_win=float(self.motion_win.get()),
            motion_lf=float(self.motion_lf.get()),
            motion_max_gap=float(self.motion_gap.get()),
            emg="mask" if self.do_emg.get() else "none",
            emg_z=float(self.emg_z.get()),
            emg_win=float(self.emg_win.get()),
            emg_lo=float(self.emg_lo.get()),
            emg_hi=float(self.emg_hi.get()),
            ied=IED_LABELS[self.ied.get()],
            ied_z=float(self.ied_z.get()),
            reference=ref,
            resample_to=float(self.resample_to.get()) if self.do_resample.get() else None,
        )
        cfg.detect_kwargs["min_votes"] = min_votes
        if self.do_epoch.get():
            cfg.events = np.arange(1, self.data.shape[-1] - 2048, int(self.fs * 5))
            cfg.tmin, cfg.tmax = -1.0, 1.0
        return cfg, run_kwargs

    # ------------------------------------------------------------ 运行
    def run(self):
        if self.data is None:
            messagebox.showwarning("提示", "请先加载数据")
            return
        if self.do_bad.get() and self.confirmed_bad is None:
            messagebox.showwarning("提示", "请先在步骤 1 点击\"确认\"坏通道名单"
                                           "（或取消勾选步骤 1）")
            return
        cfg, run_kwargs = self._build_cfg_and_kwargs()
        if cfg is None:
            return
        self.status.config(text="运行中……")

        def work():
            if self.data.ndim == 3:
                n_eval = min(30, self.data.shape[0])
                ev = evaluate_config(self.data, self.fs, cfg, max_trials=n_eval,
                                     verbose=True, run_kwargs=run_kwargs)
                single = run_preprocessing(self.data[0].astype(np.float64),
                                           self.fs, cfg, **run_kwargs)
                lines = [
                    render_protocol_report(single["report"]),
                    "",
                    f"---- multi-trial summary (first {n_eval} trials, global bad lock) ----",
                    f"locked bad channels    : {ev['first_report']['bad_channels']}",
                    f"line peak ratio after  : {np.median(ev['line_ratio_after']):.2f} (1 = clean)",
                    f"signal retention r     : {np.median(ev['retention']):.3f}",
                    f"gamma retention        : {np.median(ev['gamma_retention']):.3f}",
                    f"theta retention        : {np.median(ev['theta_retention']):.3f}",
                    f"common mode after      : {np.median(100 * ev['common_mode_after']):.1f}%",
                    f"runtime                : {ev['runtime_s']:.0f}s",
                ]
                self.after(0, lambda: self._finish_run(
                    raw=self.data[0], clean=single["data"]["continuous"],
                    report=single["report"], text="\n".join(lines)))
            else:
                res = run_preprocessing(self.data, self.fs, cfg, **run_kwargs)
                self.after(0, lambda: self._finish_run(
                    raw=self.data, clean=res["data"]["continuous"],
                    report=res["report"],
                    text=render_protocol_report(res["report"])))
        threading.Thread(target=work, daemon=True).start()

    def _finish_run(self, raw, clean, report, text):
        self.last_result = {"raw": raw, "clean": clean, "report": report}
        self.last_batch = None
        self._show(text)
        self.export_btn.config(state="normal")
        self._plot(raw, clean, report)

    def run_batch(self):
        if self.data is None or self.data.ndim != 3:
            messagebox.showwarning("提示", "全量批处理需要 3 维 trial 数据")
            return
        if self.do_bad.get() and self.confirmed_bad is None:
            messagebox.showwarning("提示", "请先确认坏通道名单（或取消勾选步骤 1）")
            return
        cfg, run_kwargs = self._build_cfg_and_kwargs()
        if cfg is None:
            return
        self.status.config(text="全量批处理中……")

        def work():
            out = batch_preprocess(self.data, self.fs, cfg, run_kwargs=run_kwargs)
            lines = [
                render_protocol_report(out["reports"][0]),
                "",
                f"batch: {out['shape_out']} @ {out['fs_out']} Hz",
                f"locked bad channels: {out['bad_channels']}",
            ]
            self.after(0, lambda: self._finish_batch(out, "\n".join(lines)))
        threading.Thread(target=work, daemon=True).start()

    def _finish_batch(self, out, text):
        self.last_batch = out
        self._show(text)
        self.export_btn.config(state="normal")

    # ------------------------------------------------------------ 导出
    def export_result(self):
        if self.last_batch is not None:
            X_clean, fs_out = self.last_batch["X_clean"], self.last_batch["fs_out"]
            report = self.last_batch["reports"][0]
            bad = self.last_batch["bad_channels"]
            ch_map = self.last_batch["out_ch_map"]
        elif self.last_result is not None:
            X_clean = self.last_result["clean"]
            fs_out = self.last_result["report"]["fs"]
            report = self.last_result["report"]
            bad = report["bad_channels"]
            ch_map = report["out_ch_map"]
        else:
            messagebox.showwarning("提示", "请先运行协议")
            return
        path = filedialog.asksaveasfilename(defaultextension=".npz",
                                            filetypes=[("numpy archive", "*.npz")])
        if not path:
            return
        payload = {
            "X_clean": np.asarray(X_clean),
            "fs": np.asarray(float(fs_out)),
            "bad_channels": np.asarray(bad, dtype=int),
            "protocol_report": np.asarray(render_protocol_report(report)),
            "report_json": np.asarray(json.dumps(report, ensure_ascii=False, default=str)),
        }
        if ch_map is not None:
            payload["out_ch_map"] = np.asarray(ch_map, dtype=object)
        np.savez(path, **payload)
        self.status.config(text=f"已导出: {path}")

    def _show(self, text):
        self.status.config(text="完成")
        self.report_box.delete("1.0", "end")
        self.report_box.insert("1.0", text)

    def _plot(self, raw, cleaned, report=None):
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        fig = plot_protocol_qc(np.asarray(raw, dtype=float),
                               np.asarray(cleaned, dtype=float),
                               self.fs, report=report)
        plt.show()


if __name__ == "__main__":
    PreprocessUI().mainloop()
