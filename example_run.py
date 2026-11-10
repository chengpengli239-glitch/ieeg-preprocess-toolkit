# -*- coding: utf-8 -*-
"""无 UI 命令行示例：生成 demo 数据 -> 跑协议 -> 输出报告/QC 图/清洗结果。

用法：
    conda activate bci_env
    python example_run.py

输出（example_output/ 目录，已加入 .gitignore）：
    protocol_report.txt   协议档案（全部参数与处理结果）
    qc_single.png         单次运行前后 3x3 QC 面板
    cleaned_demo.npz      清洗后的多试次数据（X_clean, fs, bad_channels, out_ch_map）
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

    # 1) 单段连续数据
    res = run_preprocessing(X, fs, cfg, stim_onsets=demo["stim_onsets"])
    report_text = render_protocol_report(res["report"])
    with open(os.path.join(OUT_DIR, "protocol_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(report_text)
    plot_protocol_qc(X, res["data"]["continuous"], fs, report=res["report"],
                     save_path=os.path.join(OUT_DIR, "qc_single.png"))

    # 2) 多试次批处理（全局坏通道锁定）：两块 4x4 栅格各自分组 CAR
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

    print("单次运行报告：", res["report"]["shape_out"], "->", report_text.splitlines()[3])
    print("批处理输出：", out["shape_out"], "| 锁定坏通道：", out["bad_channels"])
    print("全部输出已写入:", OUT_DIR)


if __name__ == "__main__":
    main()
