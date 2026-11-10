# iEEG Preprocess Toolkit

一套面向颅内脑电（iEEG / ECoG / sEEG）的**可复现预处理协议库 + 桌面 UI + 九宫格质量控制**工具。

核心思路：预处理不是"点几下就完事"的黑箱，而是一份**可记录、可对比、可判读的协议**。本工具把每一
步做过的处理、用过的全部参数写成可读的协议档案，并用"处理前 vs 处理后"的九张 QC 图回答两个问题：
该做的有没有做干净（工频、共模、坏道、伪迹），不该动的神经信号有没有被改坏（保留度、频段、波形）。

## 目录结构

```text
ieeg-preprocess-toolkit/
├── preprocessing.py        # 核心函数族：全部预处理步骤、协议编排、QC 指标与绘图
├── utils.py                # 轻量工具：数据加载、特征、相位、时间顺序切分评估、绘图
├── preprocess_ui.py        # Tkinter 桌面 UI（单文件，依赖同目录的 preprocessing.py）
├── example_run.py          # 无 UI 的命令行示例（生成 demo 数据 -> 跑协议 -> 出 QC 图与报告）
├── layouts/
│   └── demo_layout.xlsx    # 演示用电极排布表（32 通道，两块 4x4 栅格）
└── README.md
```

## 功能总览

`preprocessing.py` 提供从原始数据到特征提取前的完整预处理，固定顺序（防振铃 + 防阶跃设计）：

```text
0 去直流(dc_method) -> 1 坏通道 -> 2 刺激伪迹(先于一切时域滤波)
-> 3 频率滤波 -> 4 工频(基频+谐波+自定义频点) -> 5 运动/EMG 掩码
-> 6 重参考(8 种) -> 7 IED(可选) -> 8 重采样 -> 9 事件分段 -> 10 段级质检
```

每一步都提供多种可切换方法，见下表：

| 步骤 | 可选方法 | 关键参数 | 何时选择 |
| --- | --- | --- | --- |
| 0 去直流 `dc_method` | `none` / `mean` / `median` | — | 有直流偏置/极化漂移选 `median`（对刺激脉冲稳健） |
| 1 坏通道 `bad_detect` | `mad` / `ptp` / `flatness` / `corr` / `ica`，`+` 并集，`vote:` 多数投票 | `min_votes`（三取二 / 四取三） | 推荐投票；每种判据结果都输出供复核 |
| 1 坏通道处置 `bad_handle` | `none` / `mask` / `interpolate` / `drop` | — | 保守选 `mask`；`drop` 会剔除并自动重映射双极对/分组/邻接矩阵 |
| 2 刺激伪迹 `stim` | `none` / `template_subtract` / `interpolate` / `blank` | `stim_pre`, `stim_post` | 电刺激必须处理且先于一切时域滤波；短伪迹选 `interpolate` |
| 3 频率滤波 `filter_method` | `none` / `highpass` / `lowpass` / `bandpass` / `detrend` | `hp_cutoff`, `lp_cutoff`, `detrend_seg`, `detrend_order` | 高通去慢漂移；`detrend` 为汉宁窗 50% 重叠滑动拟合，无窗口边界阶跃 |
| 4 工频 `line_method` | `none` / `iir_notch` / `fir_notch` / `cleanline` / `fft_interp` | `line_f0`, `line_q`, `line_width`, `line_win`, `line_harmonics`, `line_freqs` | 短记录稳定选 `iir_notch`；长记录相位漂移选 `cleanline`；介意 PSD 深谷选 `fft_interp`；`line_freqs` 支持自定义多频点（如显示器 60Hz、设备 65/78/140Hz） |
| 5 运动伪迹 `motion` | `none` / `mask` / `interpolate` | `motion_win`, `motion_z`, `motion_lf`, `motion_max_gap` | 掩码传播到分段质检，不置零避免阶跃振铃；z 分数用中位数/MAD 稳健统计 |
| 5 EMG `emg` | `none` / `mask` | `emg_lo`, `emg_hi`, `emg_z`, `emg_win` | 颞叶底面/外侧电极的咀嚼肌污染（60-180Hz 宽带） |
| 6 重参考 `reference` | `none` / `car` / `median` / `group_car` / `shaft_car` / `white_matter` / `bipolar` / `laplacian` | 见下 | 单一植入物选 `car`；多栅格/多脑区必须 `group_car`；sEEG 针道选 `shaft_car` 或 `bipolar`；有白质触点选 `white_matter`；二维栅格局灶分析选 `laplacian` |
| 7 IED `ied` | `none` / `report` / `exclude` | `ied_z` | 发作间期放电检测（包络 z + 幅值 + 一阶差分陡峭度三判据）；`exclude` 会剔除重叠试次 |
| 8 重采样 `resample_to` | `None` 或目标采样率 | — | 多相重采样，自动同步事件坐标 |
| 9 分段 | `events` / `tmin` / `tmax` / `baseline` | — | 与伪迹掩码重叠的试次自动剔除 |
| 10 段级质检 | `ptp_max` / `qc_var_z` | — | 剔除峰峰/方差异常片段 |

### 重参考的选用要点

- **CAR（共同平均）**：减所有好通道均值，温和的空间高通，去共模、留局部梯度，单通道解码/电极级特征首选。
- **group_car（分组共同平均）**：多块电极各自减本组均值，避免把一块电极的共模灌进另一块。
- **median（中位数参考）**：与 CAR 同理但更稳健，适合共模本身被个别高幅通道污染的数据。
- **bipolar（双极）**：沿针道相邻差分，临床 sEEG/癫痫与 HFO 检测主力，改变通道数与幅值地形。
- **laplacian（拉普拉斯）**：减最近邻平均，强空间高通，突出局灶活动，适合高 gamma 定位/CSD 地形，不适合低频振荡与解码类任务（会给相邻通道注入负相关、幅值大幅下降，均属预期）。
- **shaft_car / white_matter**：sEEG 专属，针道内均值 / 白质触点参考。

## 质量控制（QC 九宫格）

`plot_protocol_qc` 输出 3x3 面板（9 张，英文标注）：PSD 前后叠加、波形前后叠加、工频峰比
（50/100/150Hz）、逐通道信号保留度（按 `out_ch_map` 对齐，drop/双极不串行）、各频段功率保留比、
残余共模（秩 1 方差解释率）、空间相干性（带符号平均 r）、逐通道幅值（坏道标红）、伪迹覆盖/坏通道摘要。

`plot_config_comparison` 对多套协议输出对比面板。判读原则：先看工频干净与波形没坏，再看信号保留，
再用共模/空间相干性确认参考方式符合下游分析，最后复查坏道与伪迹。UI 内置"QC 图解读"按钮，逐张说明。

## 快速开始

### 1. 环境

```bash
conda create -n bci_env python=3.10 -y
conda activate bci_env
pip install numpy scipy pandas scikit-learn matplotlib openpyxl
```

### 2. 桌面 UI

```bash
conda activate bci_env
python preprocess_ui.py
```

三栏布局：左栏步骤 0-3（去直流、坏通道确认门控、刺激伪迹、工频），中栏步骤 4-7（运动/EMG、
IED、重参考、重采样、分段），右栏加载数据 / 运行协议 / QC 图解读 / 报告 / 导出。

- **运行协议**：二维数据完整跑一次；三维数据用前 30 个 trial 做指标评估 + 第 1 个 trial 画 QC 图
  （快速试跑，判断参数是否合适）。
- **全量批处理**：对全部 trial 执行完整预处理，坏通道在全会话上全局锁定（同一套通道对所有 trial 生效，
  保证 drop/双极后各试次维度一致）。
- **导出清洗结果**：把结果保存为 `.npz`（`X_clean`、`fs`、`bad_channels`、`out_ch_map`、协议报告）。
- 重参考选 `group_car` / `laplacian` 时加载电极排布表 `.xlsx`（表内连通块自动成为分组、块内构建
  4 邻接；首列为针道名时自动按针道断开邻接）；`bipolar` / `shaft_car` 需加载通道名 `.txt`。

### 3. 命令行示例

```bash
conda activate bci_env
python example_run.py
```

生成 32 通道合成 LFP（含 50/100Hz 工频、刺激伪迹、运动漂移、坏通道），跑一套协议，输出协议报告、
QC 九宫格图和清洗后的 `.npz`，可用于无 GUI 环境或批处理脚本。

## 核心 API

```python
from preprocessing import (
    PreprocessConfig, run_preprocessing, batch_preprocess, evaluate_config,
    render_protocol_report, plot_protocol_qc, plot_config_comparison,
)

cfg = PreprocessConfig(
    dc_method="median",
    bad_detect="vote:mad+ptp+corr", manual_bad=[], bad_handle="mask",
    stim="interpolate", stim_pre=0.05, stim_post=0.15,
    filter_method="highpass", hp_cutoff=0.5,
    line_method="iir_notch", line_f0=50.0, line_harmonics=1, line_freqs=[],
    motion="mask", motion_z=4.0,
    reference="car",
)

# 单段连续数据 (n_ch, n_t)
res = run_preprocessing(X, fs, cfg, stim_onsets=onsets)
print(render_protocol_report(res["report"]))
plot_protocol_qc(X, res["data"]["continuous"], fs, report=res["report"])

# 多试次数据 (n_trials, n_ch, n_t)：全局坏通道锁定 + 全量批处理
out = batch_preprocess(X3, fs, cfg, run_kwargs={"groups": groups})
np.savez("cleaned.npz", X_clean=out["X_clean"], fs=out["fs_out"],
         bad_channels=out["bad_channels"], out_ch_map=out["out_ch_map"])

# 多协议对比评估（每套协议一个 evaluate_config，再画对比面板）
ev = evaluate_config(X3, fs, cfg, max_trials=30)
```

数据约定：二维 `(n_ch, n_t)` 或三维 `(n_trials, n_ch, n_t)` 的 numpy float 数组，时间沿最后一维。
`run_preprocessing` 返回的报告含 `config` 全参数快照、`bad_channels`、`out_ch_map`（输出行 ->
原始通道索引或双极对）、`motion_frac` / `emg_frac` / `ied_count` / `laplacian_isolated` 等，
`render_protocol_report` 把它渲染成可读文本，是后续特征提取的复现依据。

## 注意事项

- 带通上限必须低于奈奎斯特频率；500 Hz 采样可做 80-250 Hz ripple，256 Hz 不行。
- `drop` 模式会改变通道数，下游特征提取务必携带 `out_ch_map` 记账。
- 同一批数据建议跑 2-3 套协议对比后固化其一，不要默认一套参数打天下。
- 本仓库为独立发布版；个人实验目录中的完整项目（notebook、数据、更多布局）不在仓库内。
