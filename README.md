# iEEG Preprocess Toolkit

A reproducible preprocessing protocol for intracranial electrophysiology (iEEG / ECoG / sEEG),
with a desktop UI and a nine-panel quality-control (QC) suite.

The core idea: preprocessing should not be a black box. Every run records **what was done and with
which parameters** in a human-readable protocol report, and a before/after QC dashboard answers two
questions - did the pipeline remove what it should (line noise, common mode, bad channels,
artifacts), and did it damage what it should keep (signal retention, spectral content, waveforms)?

## Features

- **Full preprocessing chain** with swappable methods at every step and a fixed, artifact-safe
  ordering (DC removal before filtering, stimulation-artifact removal before any temporal filter,
  windowed detrending without boundary steps).
- **Bad-channel detection** with five criteria (`mad`, `ptp`, `flatness`, `corr`, `ica`), union or
  majority-voting modes, per-criterion reporting, and manual additions that never override
  automatic results.
- **Drop-safe index bookkeeping**: deleting channels remaps bipolar pairs, channel groups, white
  matter contacts, and the adjacency matrix; every output row carries an `out_ch_map` back to the
  original channels (or bipolar pairs), so QC metrics and downstream analysis never misalign.
- **Global bad-channel locking for multi-trial data**: one mask estimated over the whole session
  and applied to every trial, so outputs always stack into a consistent 3-D tensor.
- **Eight re-referencing schemes**: common average (CAR), median, grouped CAR, shaft-local CAR,
  white-matter, bipolar, Laplacian, and none, with Laplacian isolated-node fallback.
- **Line-noise removal** with four algorithms (IIR notch, FIR bandstop, sliding-window spectral
  regression, FFT interpolation repair) supporting a base frequency, harmonics, and arbitrary
  custom frequency lists (e.g., 60 Hz monitor, 65/78/140 Hz device noise).
- **Artifact masks** for slow motion drift and high-frequency EMG (60-180 Hz), using robust
  median/MAD z-scores, plus interictal epileptiform discharge (IED) detection with a three-part
  criterion (envelope z-score + amplitude + first-derivative steepness).
- **Reproducible protocol reports** (`render_protocol_report`) with the full parameter snapshot,
  bad-channel details, mask coverages, and output-to-input channel mapping.
- **Nine-panel QC dashboard** (`plot_protocol_qc`) and multi-protocol comparison
  (`plot_config_comparison`) to select the right method combination for each dataset.
- **Tkinter desktop UI** with three-column layout, bad-channel confirmation gating, batch
  processing, and `.npz` export.

## Installation

```bash
conda create -n bci_env python=3.10 -y
conda activate bci_env
pip install numpy scipy pandas scikit-learn matplotlib openpyxl
```

Python 3.9+ is required.

## Quick start

### Desktop UI

```bash
conda activate bci_env
python preprocess_ui.py
```

- **Run protocol**: for 2-D data, one full run; for 3-D (trials x channels x time) data, a fast
  evaluation over the first 30 trials plus the nine-panel QC figure of the first trial.
- **Full batch**: processes every trial with the globally locked bad-channel mask.
- **Export cleaned result**: saves `X_clean`, `fs`, `bad_channels`, `out_ch_map`, and the protocol
  report to a `.npz` archive.

Spatial parameters are loaded from an electrode-layout workbook (`.xlsx`): connected blocks become
groups for grouped CAR and 4-neighbor adjacency for the Laplacian; shaft names in the first column
automatically break adjacency between shafts. Bipolar and shaft CAR additionally need a channel
name list (`.txt`, one name per line).

### Command line

```bash
conda activate bci_env
python example_run.py
```

Generates a 32-channel synthetic LFP (50/100 Hz line noise, stimulation artifacts, motion drift,
bad channels), runs the protocol, and writes the protocol report, the QC dashboard, and the cleaned
`.npz` archive into `example_output/`.

## Pipeline order

```text
0 DC removal (dc_method)           1 bad channels
2 stimulation artifacts            3 frequency filtering
(before any temporal filter)       (after stimulation artifacts)
4 line noise                       5 motion / EMG masks
(base + harmonics + custom list)
6 re-referencing                   7 IED detection (optional)
8 resampling (optional)            9 epoch segmentation
10 epoch QC
```

| Step | Methods | Key parameters | When to choose |
| --- | --- | --- | --- |
| 0 DC removal `dc_method` | `none` / `mean` / `median` | - | `median` is robust to stimulation pulses |
| 1 Bad channels `bad_detect` | `mad` / `ptp` / `flatness` / `corr` / `ica`, `+` union, `vote:` majority | `min_votes` | voting recommended; all per-criterion results are reported |
| 1 Handling `bad_handle` | `none` / `mask` / `interpolate` / `drop` | - | `mask` is conservative; `drop` remaps pairs/groups/adjacency automatically |
| 2 Stimulation `stim` | `none` / `template_subtract` / `interpolate` / `blank` | `stim_pre`, `stim_post` | must precede all temporal filters; `interpolate` for short pulses |
| 3 Filtering `filter_method` | `none` / `highpass` / `lowpass` / `bandpass` / `detrend` | `hp_cutoff`, `lp_cutoff`, `detrend_seg`, `detrend_order` | `detrend` uses Hann-weighted 50% overlap, no window-boundary steps |
| 4 Line noise `line_method` | `none` / `iir_notch` / `fir_notch` / `cleanline` / `fft_interp` | `line_f0`, `line_q`, `line_width`, `line_win`, `line_harmonics`, `line_freqs` | `cleanline` for phase drift; `fft_interp` avoids PSD notches; `line_freqs` for arbitrary device noise |
| 5 Motion `motion` | `none` / `mask` / `interpolate` | `motion_win`, `motion_z`, `motion_lf`, `motion_max_gap` | masks propagate to epoch QC; median/MAD robust z-scores |
| 5 EMG `emg` | `none` / `mask` | `emg_lo`, `emg_hi`, `emg_z`, `emg_win` | 60-180 Hz broadband muscle artifacts on temporal/basal contacts |
| 6 Reference `reference` | `none` / `car` / `median` / `group_car` / `shaft_car` / `white_matter` / `bipolar` / `laplacian` | see below | choose by montage and analysis goal |
| 7 IED `ied` | `none` / `report` / `exclude` | `ied_z` | `exclude` drops overlapping epochs |
| 8 Resampling `resample_to` | `None` or target rate | - | polyphase resampling, event coordinates synced |
| 9 Epoching | `events` / `tmin` / `tmax` / `baseline` | - | epochs overlapping artifact masks are dropped |
| 10 Epoch QC | `ptp_max` / `qc_var_z` | - | removes peak-to-peak / variance outliers |

### Choosing a re-reference

| Method | What it subtracts | Typical use |
| --- | --- | --- |
| `car` | mean of all good channels | default for decoding and per-channel features |
| `group_car` | per-group mean | multiple grids/regions - avoids mixing common modes across distant arrays |
| `median` | median of all good channels | robust variant of CAR when the common mode itself is contaminated |
| `bipolar` | adjacent channel difference along a shaft | clinical sEEG, epilepsy, HFO detection |
| `laplacian` | 4-neighbor mean | focal high-gamma mapping / CSD topography; strong spatial high-pass |
| `shaft_car` | per-shaft mean | depth electrodes |
| `white_matter` | mean of white-matter contacts | sEEG with identified white-matter contacts |
| `none` | - | raw monopolar, usually only as a control |

Note that CAR/Laplacian effects are visible in the QC panels (amplitude drop, theta retention,
signed pairwise correlation), and Laplacian is generally unsuitable for low-frequency oscillation
or decoding tasks.

## Quality control panels

`plot_protocol_qc` draws a 3x3 English-labeled dashboard: (1) PSD overlay, (2) waveform overlay,
(3) line-noise peak ratios at 50/100/150 Hz, (4) per-output-row signal retention aligned by
`out_ch_map`, (5) band power retention, (6) residual common mode (rank-1 variance explained),
(7) spatial coherence (signed mean pairwise r), (8) per-output-row amplitude with bad channels in
red, (9) motion/EMG/IED coverage and bad-channel summary. `plot_config_comparison` compares
multiple protocol evaluations side by side.

Reading order: first line-noise cleanliness and waveform integrity, then signal retention, then
common-mode/spatial-coherence against the intended spatial high-pass strength, and finally artifact
and bad-channel summaries.

## Core API

```python
import numpy as np
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

# Single continuous recording (n_ch, n_t)
res = run_preprocessing(X, fs, cfg, stim_onsets=onsets)
print(render_protocol_report(res["report"]))
plot_protocol_qc(X, res["data"]["continuous"], fs, report=res["report"])

# Multi-trial data (n_trials, n_ch, n_t): global bad-channel lock + batch processing
out = batch_preprocess(X3, fs, cfg, run_kwargs={"groups": groups})
np.savez("cleaned.npz", X_clean=out["X_clean"], fs=out["fs_out"],
         bad_channels=out["bad_channels"], out_ch_map=out["out_ch_map"])

# Protocol comparison across methods
ev_a = evaluate_config(X3, fs, cfg_car, max_trials=30)
ev_e = evaluate_config(X3, fs, cfg_group_car, max_trials=30, run_kwargs={"groups": groups})
plot_config_comparison({"A: car": ev_a, "E: group_car": ev_e})
```

## Data conventions

- Input arrays: `(n_ch, n_t)` for continuous data or `(n_trials, n_ch, n_t)` for epoched data,
  as numpy float arrays with time along the last axis.
- `.npz` archives use keys `X` and `fs`; channel names and trial metadata are optional.
- Bandpass upper limits must stay below the Nyquist frequency (no 80-250 Hz ripple analysis on
  256 Hz data).
- `drop` changes the channel count: always carry `out_ch_map` into downstream feature extraction.
- Compare 2-3 protocol configurations per dataset before freezing one.

## Project structure

```text
ieeg-preprocess-toolkit/
├── preprocessing.py        # full pipeline, protocol orchestration, QC metrics and plotting
├── utils.py                # lightweight loaders, features, phase metrics, chronological evaluation
├── preprocess_ui.py        # Tkinter desktop UI (imports preprocessing from the same directory)
├── example_run.py          # headless demo: synthetic data -> report + QC + cleaned .npz
├── layouts/
│   └── demo_layout.xlsx    # example 32-channel layout (two 4x4 grids)
├── README.md
└── .gitignore
```

This repository is the standalone release of a personal experimental pipeline. The full lab
workspace (notebooks, data, additional layouts) is intentionally not included, and no raw or
cleaned data is tracked by git.
