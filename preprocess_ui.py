# -*- coding: utf-8 -*-
"""iEEG preprocessing protocol UI (Tkinter, single file, three columns).

Run:
    conda activate bci_env
    python preprocess_ui.py

Execution order (matches preprocessing.py; ringing- and step-safe):
    DC removal -> bad channels -> stimulation artifacts -> frequency filtering
    -> line noise -> motion/EMG masks -> re-referencing -> IED -> resampling
    -> epoching -> epoch QC

Spatial parameters for step 5:
  - group_car / laplacian / laplacian8: load an electrode layout workbook
    (.xlsx); connected blocks become regions and within-block 4/8-neighbor
    adjacency is built. Shaft names in the first column split regions.
  - bipolar / shaft_car: load a channel-name list (.txt, one name per line).
  - white_matter: enter white-matter contact indices.
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
    "none": "none",
    "mean - Subtract Mean": "mean",
    "median - Subtract Median (robust)": "median",
}
BAD_METHOD_LABELS = {
    "mad - Median Absolute Deviation": "mad",
    "ptp - Peak-to-Peak": "ptp",
    "flatness - Spectral Flatness": "flatness",
    "corr - Cross-channel Correlation": "corr",
    "ica - ICA 1/VAF (residual criterion)": "ica",
}
STIM_LABELS = {
    "template_subtract - Template Subtraction": "template_subtract",
    "interpolate - Boundary Interpolation": "interpolate",
    "blank - Blanking (not recommended)": "blank",
}
LINE_LABELS = {
    "iir_notch - IIR Notch (zero-phase)": "iir_notch",
    "fir_notch - FIR Bandstop": "fir_notch",
    "cleanline - Spectral Regression": "cleanline",
    "fft_interp - FFT Interpolation Repair": "fft_interp",
}
REF_LABELS = {
    "none": "none",
    "car - Common Average Reference": "car",
    "median - Median Reference": "median",
    "group_car - Group Common Average": "group_car",
    "bipolar - Bipolar Reference (within shaft)": "bipolar",
    "shaft_car - Shaft-local CAR": "shaft_car",
    "white_matter - White Matter Reference": "white_matter",
    "laplacian - Laplacian Reference (4-neighbor)": "laplacian",
    "laplacian8 - Laplacian Reference (8-neighbor)": "laplacian8",
}
IED_LABELS = {
    "none": "none",
    "report - report only": "report",
    "exclude - report and exclude epochs": "exclude",
}

QC_GUIDE_TEXT = """QC = Quality Control. The nine panels compare before vs after and answer
two questions: did the pipeline remove what it should (line noise, common
mode, bad channels, artifacts), and did it damage what it should keep
(retention, spectral content, waveforms)?

Run combo A and combo B on the same data and compare panel by panel. The
better side is the one where "what should go, went, and what should stay,
stayed".

1) PSD overlay (median over channels)
   raw shows peaks at 50/100/150 Hz (and custom frequencies); after must have
   no isolated peaks and no abnormal notches (narrow IIR/FIR notches are
   normal, fft_interp should be smooth). 30-80 Hz should not collapse.

2) Waveform overlay (channel 0, first 2 s)
   after keeps the raw shape with line noise/artifacts removed; steps, ringing
   or sudden drifts mean a step is broken. Baseline sanity check.

3) Line-noise peak ratio (50/100/150 Hz, target = 1.0)
   raw bars are high; after bars should sit near 1.0.

4) Signal retention histogram (output-row r(raw, after), ch-map aligned)
   higher is better. Drop/bipolar outputs are remapped so rows never compare
   mismatched channels; bad rows are NaN. Laplacian/bipolar values are
   naturally lower (spatial high-pass) - not an error.

5) Band power retention (after / before, dashed line = 1.0)
   delta/theta drop under CAR/high-pass/Laplacian is expected; gamma/
   high-gamma should stay near 1.0.

6) Residual common mode (rank-1 variance explained %)
   all channels sharing one waveform = 100%; after re-referencing it should
   drop. If gamma dropped too, the reference is too aggressive.

7) Spatial coherence (signed mean pairwise r)
   raw is positive (volume conduction); after near 0 or negative means common
   mode was removed. CAR injects ~ -1/(n-1), Laplacian/bipolar ~ -0.5 - these
   are expected side effects, not damage.

8) Per-output-row amplitude (ch-map aligned; red = bad)
   bad rows are marked red and handled (zeroed / interpolated); other rows
   should stay smooth.

9) Motion mask coverage / bad channels
   red shading = motion-masked segments (excluded at epoch QC); gray text =
   none detected. Coverage should match the true artifact duration.

Reading order: panels 3 and 2 first (line clean, waveform intact), then 4/5
(retention), then 6/7 (spatial high-pass strength vs your analysis goal),
finally 8/9 (bad channels and artifacts).

Bad-channel vote rule: when criteria disagree, majority voting (2 of 3) is
the recommended default - it removes only cross-criterion agreements and
keeps ambiguous channels visible in the panels. Use the union for
exploratory artifact hunting. Panel 8 marks bad rows red and Laplacian
isolated-node fallback rows orange.
"""


def parse_layout_xlsx(path: str, sheet: str) -> dict:
    """Parse an electrode layout workbook.

    Connected blocks of channel numbers become regions; a shaft label in the
    first column splits adjacency and regions per shaft. Channel numbers must
    cover 1..n_ch exactly. Returns groups, 4- and 8-neighbor adjacency,
    shafts, channel count and block count.
    """
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb[sheet]
    grid, labels = {}, {}
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
        label_rows = sorted({cell.row for row in ws.iter_rows() for cell in row
                             if cell.column == 1 and cell.value is not None})
        for r in sorted({rr for rr, _ in grid}):
            lab = None
            for lr in label_rows:
                if lr <= r:
                    lab = str(ws.cell(row=lr, column=1).value).strip()
                else:
                    break
            if lab is not None:
                for (rr, cc) in grid:
                    if rr == r:
                        labels[(rr, cc)] = lab
    n_ch = len(grid)
    if set(grid.values()) != set(range(1, n_ch + 1)):
        raise ValueError(f"layout must cover channels 1..{n_ch} exactly")

    adj = np.zeros((n_ch, n_ch))
    groups = []
    if labels:
        for lab in sorted(set(labels.values())):
            members = [grid[c] - 1 for c in sorted(labels) if labels[c] == lab]
            groups.append(np.asarray(members, dtype=int))
            for (r, c) in sorted(labels):
                if labels[(r, c)] != lab:
                    continue
                a = grid[(r, c)] - 1
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nb = (r + dr, c + dc)
                    if nb in grid and labels.get(nb) == lab:
                        adj[a, grid[nb] - 1] = 1.0
    else:
        cells = sorted(grid)
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
            groups.append(np.asarray([grid[c] - 1 for c in block], dtype=int))
            for (r, c) in block:
                a = grid[(r, c)] - 1
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nb = (r + dr, c + dc)
                    if nb in grid and nb in block:
                        adj[a, grid[nb] - 1] = 1.0
    adj8 = np.zeros((n_ch, n_ch))
    owner = {cell: labels[cell] for cell in grid} if labels else {
        cell: bi for bi, block in enumerate(blocks) for cell in block}
    for (r, c) in sorted(grid):
        a = grid[(r, c)] - 1
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nb = (r + dr, c + dc)
            if nb in grid and owner.get(nb) == owner.get((r, c)):
                adj8[a, grid[nb] - 1] = 1.0
    return {"groups": groups, "adjacency": adj, "adjacency8": adj8,
            "shafts": groups if labels else None,
            "n_ch": n_ch, "n_blocks": len(groups)}


class PreprocessUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("iEEG Preprocessing Protocol UI")
        self.geometry("1280x780")
        self.data = None
        self.fs = 500.0
        self.confirmed_bad = None
        self.layout = None
        self.layout_label_text = "no layout loaded"
        self.ch_names = None
        self.wm_channels = []
        self.cond_labels = None
        self.last_result = None
        self.last_batch = None
        self._build()

    def _row(self, parent, label, values, default, width=30):
        f = ttk.Frame(parent)
        f.pack(fill="x", pady=2)
        ttk.Label(f, text=label, width=26).pack(side="left")
        var = tk.StringVar(value=default)
        ttk.Combobox(f, textvariable=var, values=values, state="readonly",
                     width=width).pack(side="left")
        return var

    def _entry_row(self, parent, label, default, note):
        f = ttk.Frame(parent)
        f.pack(fill="x", pady=2)
        ttk.Label(f, text=label, width=26).pack(side="left")
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
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        return inner

    def _build(self):
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True)
        left = self._scrolled_column(outer, 400)
        mid = self._scrolled_column(outer, 400)
        right = ttk.Frame(outer, width=440)
        right.pack(side="left", fill="y")
        right.pack_propagate(False)

        ttk.Label(left, text="Step 0 DC removal + filtering").pack(anchor="w")
        self.dc = self._row(left, "DC removal", list(DC_LABELS.keys()),
                            "median - Subtract Median (robust)")
        self.do_f0 = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="frequency filtering (runs after stim artifacts)",
                        variable=self.do_f0).pack(anchor="w")
        self.f0 = self._row(left, "Filter method",
                            ["none", "highpass", "lowpass", "bandpass", "detrend"],
                            "highpass")
        self.hp = self._entry_row(left, "High-pass cutoff (Hz)", "0.5", "0.1-1.0")
        self.lp = self._entry_row(left, "Low-pass cutoff (Hz)", "250", "150-250")
        self.ds = self._entry_row(left, "Detrend window (s)", "10",
                                  "Hann 50% overlap")

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(left, text="Step 1 Bad channels (confirm gate)").pack(anchor="w")
        self.do_bad = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, variable=self.do_bad).pack(anchor="w")
        self.bad_detect = self._row(
            left, "Detection", list(BAD_METHOD_LABELS.keys())
            + ["vote:mad+ptp+corr (2 of 3)", "vote:mad+ptp+corr+ica (3 of 4)"],
            "vote:mad+ptp+corr (2 of 3)")
        self.bad_handle = self._row(left, "Handling",
                                    ["mask (zero)", "interpolate (neighbors)",
                                     "drop (remove + remap)", "none"],
                                    "mask (zero)")
        self.manual_bad = self._entry_row(left, "Manual bad channels", "",
                                          "union with auto results")
        ttk.Button(left, text="Preview bad-channel detection",
                   command=self.preview_bad).pack(anchor="w", pady=2)
        self.bad_box = tk.Text(left, height=3, width=46)
        self.bad_box.pack(fill="x")
        cf = ttk.Frame(left)
        cf.pack(fill="x", pady=2)
        ttk.Label(cf, text="Final bad channels:").pack(side="left")
        self.confirm_entry = ttk.Entry(cf, width=16)
        self.confirm_entry.pack(side="left", padx=4)
        ttk.Button(cf, text="Confirm", command=self.confirm_bad).pack(side="left", padx=4)
        self.confirm_label = ttk.Label(cf, text="not confirmed", foreground="red")
        self.confirm_label.pack(side="left")

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(left, text="Step 2 Stimulation artifacts").pack(anchor="w")
        self.do_stim = tk.BooleanVar(value=False)
        self.stim_check = ttk.Checkbutton(left, variable=self.do_stim, state="disabled")
        self.stim_check.pack(anchor="w")
        self.stim = self._row(left, "Method", list(STIM_LABELS.keys()),
                              "interpolate - Boundary Interpolation")
        self.stim_pre = self._entry_row(left, "Pre boundary (s)", "0.05", "0.02-0.10")
        self.stim_post = self._entry_row(left, "Artifact window (s)", "0.15", "0.05-0.50")

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(left, text="Step 3 Line noise").pack(anchor="w")
        self.do_line = tk.BooleanVar(value=True)
        self.line_check = ttk.Checkbutton(left, variable=self.do_line, state="disabled")
        self.line_check.pack(anchor="w")
        self.line = self._row(left, "Method", list(LINE_LABELS.keys()),
                              "iir_notch - IIR Notch (zero-phase)")
        self.line_f0 = self._entry_row(left, "Base frequency (Hz)", "50", "50 in China")
        self.line_q = self._entry_row(left, "IIR quality factor Q", "30", "10-50")
        self.line_width = self._entry_row(left, "FIR width (Hz)", "2", "1-5")
        self.line_win = self._entry_row(left, "CleanLine window (s)", "4", "2-8")
        self.line_harm = self._entry_row(left, "Harmonics", "1", "1 = base only")
        self.line_freqs = self._entry_row(left, "Extra frequencies (Hz)", "",
                                          "comma-separated, e.g. 60,65,78,140")

        ttk.Label(mid, text="Step 4 Motion artifacts (MAD-robust z)").pack(anchor="w")
        self.do_motion = tk.BooleanVar(value=False)
        self.motion_check = ttk.Checkbutton(mid, variable=self.do_motion, state="disabled")
        self.motion_check.pack(anchor="w")
        self.motion = self._row(mid, "Method",
                                ["mask - propagate (no zeroing)",
                                 "interpolate - short-segment repair"],
                                "mask - propagate (no zeroing)")
        self.motion_z = self._entry_row(mid, "z threshold", "4", "3-6")
        self.motion_win = self._entry_row(mid, "Window (s)", "1", "0.5-2")
        self.motion_lf = self._entry_row(mid, "Slow-drift low-pass (Hz)", "2", "1-3")
        self.motion_gap = self._entry_row(mid, "Max repair length (s)", "0.5", "0.2-1")

        ttk.Separator(mid, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(mid, text="Step 4b High-frequency EMG (60-180 Hz)").pack(anchor="w")
        self.do_emg = tk.BooleanVar(value=False)
        self.emg_check = ttk.Checkbutton(mid, variable=self.do_emg, state="disabled")
        self.emg_check.pack(anchor="w")
        self.emg_z = self._entry_row(mid, "EMG z threshold", "5", "4-8")
        self.emg_win = self._entry_row(mid, "EMG window (s)", "1", "0.5-2")
        self.emg_lo = self._entry_row(mid, "EMG band low (Hz)", "60", "mastication")
        self.emg_hi = self._entry_row(mid, "EMG band high (Hz)", "180", "< Nyquist")

        ttk.Separator(mid, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(mid, text="Step 4c IED (interictal discharges)").pack(anchor="w")
        self.ied = self._row(mid, "Mode", list(IED_LABELS.keys()), "none")
        self.ied_z = self._entry_row(mid, "IED z threshold", "5",
                                     "envelope+amplitude+steepness")

        ttk.Separator(mid, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(mid, text="Step 5 Re-referencing").pack(anchor="w")
        self.do_ref = tk.BooleanVar(value=True)
        self.ref_check = ttk.Checkbutton(mid, variable=self.do_ref, state="disabled")
        self.ref_check.pack(anchor="w")
        self.ref = self._row(mid, "Method", list(REF_LABELS.keys()),
                             "car - Common Average Reference")
        lf = ttk.Frame(mid)
        lf.pack(fill="x", pady=2)
        ttk.Button(lf, text="Load layout (.xlsx)", command=self.load_layout).pack(
            side="left", padx=4)
        self.layout_label = ttk.Label(lf, text=self.layout_label_text, foreground="gray")
        self.layout_label.pack(side="left")
        lf2 = ttk.Frame(mid)
        lf2.pack(fill="x", pady=2)
        ttk.Button(lf2, text="Load channel names (.txt)",
                   command=self.load_ch_names).pack(side="left", padx=4)
        self.ch_label = ttk.Label(lf2, text="no channel names", foreground="gray")
        self.ch_label.pack(side="left")
        self.wm_entry = self._entry_row(mid, "White-matter contacts", "",
                                        "comma-separated, e.g. 3,7,11")
        ttk.Label(mid, text="group_car/laplacian/laplacian8 need a layout "
                            "(shaft names split regions); bipolar/shaft_car need "
                            "channel names; white_matter needs contacts.").pack(
            anchor="w", pady=2)

        ttk.Separator(mid, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(mid, text="Step 6 Resampling").pack(anchor="w")
        self.do_resample = tk.BooleanVar(value=False)
        self.resample_check = ttk.Checkbutton(mid, variable=self.do_resample,
                                              state="disabled")
        self.resample_check.pack(anchor="w")
        self.resample_to = self._row(mid, "Target rate (Hz)", ["256", "500", "1000"],
                                     "256")

        ttk.Separator(mid, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(mid, text="Step 7 Epoching (demo fixed-interval events)").pack(anchor="w")
        self.do_epoch = tk.BooleanVar(value=False)
        self.epoch_check = ttk.Checkbutton(mid, variable=self.do_epoch, state="disabled")
        self.epoch_check.pack(anchor="w")
        self.evoked_sub = self._row(mid, "Evoked subtraction",
                                    ["none", "global (grand average)",
                                     "per_condition (per condition)"],
                                    "none")
        lf7 = ttk.Frame(mid)
        lf7.pack(fill="x", pady=2)
        ttk.Button(lf7, text="Load condition labels (.npy)",
                   command=self.load_cond_labels).pack(side="left", padx=4)
        self.cond_label = ttk.Label(lf7, text="no condition labels", foreground="gray")
        self.cond_label.pack(side="left")

        ttk.Button(right, text="Load data (.npy / .npz / demo)", command=self.load).pack(pady=6)
        self.file_label = ttk.Label(right, text="not loaded", foreground="gray",
                                    wraplength=400)
        self.file_label.pack()
        ttk.Button(right, text="Run protocol", command=self.run).pack(pady=6)
        ttk.Button(right, text="QC panel guide (1-9)",
                   command=self.show_qc_guide).pack(pady=2)
        self.status = ttk.Label(right, text="", foreground="blue")
        self.status.pack()
        exp = ttk.Frame(right)
        exp.pack(pady=4)
        self.export_btn = ttk.Button(exp, text="Export cleaned result (.npz)",
                                     command=self.export_result, state="disabled")
        self.export_btn.pack(side="left", padx=3)
        self.batch_btn = ttk.Button(exp, text="Full batch",
                                    command=self.run_batch, state="disabled")
        self.batch_btn.pack(side="left", padx=3)
        ttk.Label(right, text="Protocol report:").pack(anchor="w", padx=8)
        self.report_box = tk.Text(right, height=28, width=50)
        self.report_box.pack(padx=8, fill="both", expand=True)

        self.do_bad.trace_add("write", self._on_bad_toggle)

    def _downstream_checks(self):
        return (self.stim_check, self.line_check, self.motion_check,
                self.emg_check, self.ref_check, self.resample_check,
                self.epoch_check)

    def _on_bad_toggle(self, *_):
        if not self.do_bad.get():
            for w in self._downstream_checks():
                w.config(state="normal")

    # ---------------------------------------------------------------- actions
    def load(self):
        path = filedialog.askopenfilename(filetypes=[("numpy/demo", "*.npz *.npy"),
                                                     ("all", "*.*")])
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
        self.confirm_label.config(text="not confirmed", foreground="red")
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
        self.ch_label.config(text=f"{os.path.basename(path)} ({len(self.ch_names)} ch)",
                             foreground="green")

    def load_cond_labels(self):
        path = filedialog.askopenfilename(filetypes=[("numpy", "*.npy"), ("all", "*.*")])
        if not path:
            return
        self.cond_labels = np.load(path)
        self.cond_label.config(text=f"{os.path.basename(path)} ({len(self.cond_labels)})",
                               foreground="green")

    def load_layout(self):
        path = filedialog.askopenfilename(filetypes=[("excel", "*.xlsx")])
        if not path:
            return
        import openpyxl
        wb = openpyxl.load_workbook(path, data_only=True)
        sheets = wb.sheetnames
        if not sheets:
            messagebox.showwarning("Warning", "The workbook has no sheets")
            return
        sheet = sheets[0] if len(sheets) == 1 else self._ask_sheet(sheets)
        try:
            self.layout = parse_layout_xlsx(path, sheet)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Layout parse failed", str(exc))
            return
        extra = (f", shaft mode {len(self.layout['shafts'])} shafts"
                 if self.layout.get("shafts") else "")
        self.layout_label_text = (f"{os.path.basename(path)} [{sheet}] "
                                  f"{self.layout['n_ch']} ch, "
                                  f"{self.layout['n_blocks']} regions{extra}")
        self.layout_label.config(text=self.layout_label_text, foreground="green")

    def show_qc_guide(self):
        win = tk.Toplevel(self)
        win.title("QC panel guide (1-9)")
        win.geometry("860x680")
        txt = tk.Text(win, wrap="word", font=("Consolas", 10))
        txt.insert("1.0", QC_GUIDE_TEXT)
        txt.config(state="disabled")
        sb = ttk.Scrollbar(win, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        txt.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    def _ask_sheet(self, sheets):
        win = tk.Toplevel(self)
        win.title("Choose a sheet")
        var = tk.StringVar(value=sheets[0])
        ttk.Combobox(win, textvariable=var, values=sheets, state="readonly").pack(
            padx=12, pady=8)
        ttk.Button(win, text="OK", command=win.destroy).pack(pady=6)
        self.wait_window(win)
        return var.get()

    def _selected_bad_methods(self):
        sel = self.bad_detect.get()
        if sel.startswith("vote:"):
            inner = sel[5:].split(" (")[0].split("+")
            return inner, (3 if "3 of 4" in sel else 2)
        return [BAD_METHOD_LABELS[sel]], 1

    def preview_bad(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Load data first")
            return
        methods, min_votes = self._selected_bad_methods()
        x = self.data[0].astype(np.float64) if self.data.ndim == 3 else self.data
        final, detail = detect_bad_vote(x, self.fs, methods, min_votes=min_votes)
        lines = [f"{m}: {len(res)} -> {res.tolist()}" for m, res in detail.items()]
        lines.append(f"vote result (>= {min_votes}): {final.tolist()}")
        self.bad_box.delete("1.0", "end")
        self.bad_box.insert("1.0", "\n".join(lines))
        self.confirm_entry.delete(0, "end")
        self.confirm_entry.insert(0, ",".join(str(c) for c in final))

    def confirm_bad(self):
        text = self.confirm_entry.get().replace("，", ",").strip()
        self.confirmed_bad = [] if not text else [int(v) for v in text.split(",") if v.strip()]
        self.confirm_label.config(text=f"confirmed: {self.confirmed_bad}", foreground="green")
        for w in self._downstream_checks():
            w.config(state="normal")

    # ------------------------------------------------------------- config
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
                messagebox.showwarning("Warning", "group_car requires a layout")
                return None, None
            run_kwargs["groups"] = self.layout["groups"]
        elif ref in ("laplacian", "laplacian8"):
            if self.layout is None:
                messagebox.showwarning("Warning", f"{ref} requires a layout")
                return None, None
            run_kwargs["adjacency"] = (self.layout["adjacency8"]
                                       if ref == "laplacian8"
                                       else self.layout["adjacency"])
        elif ref == "bipolar":
            if shafts is None:
                messagebox.showwarning(
                    "Warning", "bipolar needs channel names or a shaft-labeled "
                               "layout; falling back to array-order pairing")
            run_kwargs["bipolar_pairs"] = chain_bipolar_pairs(n_ch_now, shafts)
        elif ref == "shaft_car":
            if shafts is None:
                messagebox.showwarning("Warning", "shaft_car needs channel names "
                                                  "or a shaft-labeled layout")
                return None, None
            run_kwargs["shafts"] = shafts
        elif ref == "white_matter":
            wm = [int(v) for v in self.wm_entry.get().replace("，", ",").split(",")
                  if v.strip()]
            if not wm:
                messagebox.showwarning("Warning", "white_matter needs contacts")
                return None, None
            self.wm_channels = wm
            run_kwargs["wm_channels"] = wm
        if self.ch_names is not None:
            run_kwargs["ch_names"] = self.ch_names

        methods, min_votes = self._selected_bad_methods()
        line_freqs = [float(v) for v in
                      self.line_freqs.get().replace("，", ",").replace(" ", "").split(",")
                      if v.strip()]
        evoked_map = {"none": "none",
                      "global (grand average)": "global",
                      "per_condition (per condition)": "per_condition"}
        evoked_sub = evoked_map[self.evoked_sub.get()]
        cond_labels = None
        if self.do_epoch.get() and evoked_sub == "per_condition":
            n_events = len(np.arange(1, self.data.shape[-1] - 2048, int(self.fs * 5)))
            if self.cond_labels is None:
                messagebox.showwarning("Warning", "per_condition needs condition labels")
                return None, None
            if len(self.cond_labels) != n_events:
                messagebox.showwarning(
                    "Warning", f"labels {len(self.cond_labels)} != events {n_events}")
                return None, None
            cond_labels = self.cond_labels
        cfg = PreprocessConfig(
            dc_method=DC_LABELS[self.dc.get()],
            filter_method=self.f0.get() if self.do_f0.get() else "none",
            hp_cutoff=float(self.hp.get()),
            lp_cutoff=float(self.lp.get()),
            detrend_seg=float(self.ds.get()),
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
            evoked_sub=evoked_sub,
            cond_labels=cond_labels,
        )
        cfg.detect_kwargs["min_votes"] = min_votes
        if self.do_epoch.get():
            cfg.events = np.arange(1, self.data.shape[-1] - 2048, int(self.fs * 5))
            cfg.tmin, cfg.tmax = -1.0, 1.0
        return cfg, run_kwargs

    # ---------------------------------------------------------------- run
    def run(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Load data first")
            return
        if self.do_bad.get() and self.confirmed_bad is None:
            messagebox.showwarning("Warning", "Confirm the bad-channel list first "
                                              "(or uncheck step 1)")
            return
        cfg, run_kwargs = self._build_cfg_and_kwargs()
        if cfg is None:
            return
        self.status.config(text="Running ...")

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
        """Full batch with a globally locked bad-channel set (results stay in
        memory; export with the Export button)."""
        if self.data is None or self.data.ndim != 3:
            messagebox.showwarning("Warning", "Full batch needs 3D trial data")
            return
        if self.do_bad.get() and self.confirmed_bad is None:
            messagebox.showwarning("Warning", "Confirm the bad-channel list first")
            return
        cfg, run_kwargs = self._build_cfg_and_kwargs()
        if cfg is None:
            return
        self.status.config(text="Full batch running ...")

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

    # -------------------------------------------------------------- export
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
            messagebox.showwarning("Warning", "Run the protocol first")
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
        self.status.config(text=f"Exported: {path}")

    def _show(self, text):
        self.status.config(text="Done")
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
