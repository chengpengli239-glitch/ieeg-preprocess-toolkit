# -*- coding: utf-8 -*-
"""Headless demo: synthetic data -> protocol -> report + QC figure + cleaned npz.

Run:
    conda activate bci_env
    python example_run.py

Outputs (example_output/, git-ignored):
    protocol_report.txt   full parameter archive
    qc_single.png         before/after 9-panel QC dashboard
    cleaned_demo.npz      batch-cleaned trials (X_clean, fs, bad_channels, out_ch_map)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from preprocessing import (PreprocessConfig, generate_demo_lfp, run_preprocessing,
                           batch_preprocess, render_protocol_report,
                           plot_protocol_qc)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_output")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    demo = generate_demo_lfp(seed=0, n_ch=32, fs=512.0, duration=120.0)
    X, fs = demo["X"], demo["fs"]

    cfg = PreprocessConfig(
        dc_method="median",
        bad_detect="vote:mad+ptp+corr", manual_bad=[], bad_handle="mask",
        stim="interpolate", stim_pre=0.05, stim_post=0.15,
        filter_method="none",
        line_method="iir_notch", line_f0=50.0, line_harmonics=2, line_freqs=[],
        motion="mask", motion_z=4.0,
        reference="car",
    )

    # 1) single continuous recording
    res = run_preprocessing(X, fs, cfg, stim_onsets=demo["stim_onsets"])
    report_text = render_protocol_report(res["report"])
    with open(os.path.join(OUT_DIR, "protocol_report.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(report_text)
    plot_protocol_qc(X, res["data"]["continuous"], fs, report=res["report"],
                     save_path=os.path.join(OUT_DIR, "qc_single.png"))

    # 2) multi-trial batch with a globally locked bad-channel set and
    #    per-grid group CAR (two 4x4 grids)
    X3 = np.stack([X, X * 0.9, X * 1.1])
    groups = [np.arange(16), np.arange(16, 32)]
    cfg3 = PreprocessConfig(
        dc_method="median",
        bad_detect="vote:mad+ptp+corr", manual_bad=[], bad_handle="mask",
        stim="none", filter_method="none",
        line_method="iir_notch", line_harmonics=2, line_freqs=[],
        motion="mask", motion_z=4.0,
        reference="group_car",
    )
    out = batch_preprocess(X3, fs, cfg3, run_kwargs={"groups": groups})
    np.savez(os.path.join(OUT_DIR, "cleaned_demo.npz"),
             X_clean=out["X_clean"], fs=np.asarray(out["fs_out"]),
             bad_channels=np.asarray(out["bad_channels"], dtype=int),
             out_ch_map=np.asarray(out["out_ch_map"], dtype=object))

    print("single run:", res["report"]["shape_out"], "->", report_text.splitlines()[3])
    print("batch:", out["shape_out"], "| locked bad channels:", out["bad_channels"])
    print("outputs written to:", OUT_DIR)


if __name__ == "__main__":
    main()
