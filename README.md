# EEG Seizure Detection Baseline (CHB-MIT)

Minimal end-to-end EEG pipeline for seizure detection using real clinical data from the CHB-MIT Scalp EEG Database. The project demonstrates preprocessing, event-based labeling, feature extraction, and baseline classification under a leakage-safe evaluation setup.

## Overview

This pipeline:

- loads EDF recordings using MNE
- parses seizure annotations from patient summary files
- applies light preprocessing (`0.5-40 Hz` bandpass filter)
- generates fixed-length windows with overlap
- labels windows based on seizure-interval overlap
- extracts interpretable per-channel features
- trains a baseline classifier with file-level splitting

## Dataset

- Source: CHB-MIT Scalp EEG Database (PhysioNet)
- Patient: `chb01`
- Subset used: `chb01_01`, `chb01_03`, `chb01_04`, `chb01_05`, `chb01_06`, `chb01_15`, `chb01_16`, `chb01_17`

## Method

### Preprocessing

- Bandpass filter: `0.5-40 Hz`
- No aggressive artifact removal, intentionally kept minimal

### Windowing and Labeling

- Fixed `2-second` windows with `50%` overlap
- Window labeled as seizure if it overlaps the annotated seizure interval

### Features (Per-Channel)

- Band power: delta, theta, alpha, beta
- Mean
- Variance
- Line length

### Model

- Logistic regression
- Class-weighted to address imbalance
- File-level split with no window-level leakage

## Results

Train files: `chb01_03`, `chb01_15`, `chb01_01`, `chb01_05`  
Test files: `chb01_04`, `chb01_16`, `chb01_06`, `chb01_17`

| Metric | Value |
| --- | ---: |
| Precision | `0.4815` |
| Recall | `0.6500` |
| F1 Score | `0.5532` |

A temporal post-processing stage using moving-average smoothing and event aggregation detected `2/2` held-out seizure events with `0.25` false alarms/hour on the current test split.

This is a simple baseline for portfolio-scale experimentation, not a clinical model.

## Project Structure

```text
src/eeg_pipeline/        # reusable modules (loading, windowing, features)
scripts/                 # pipeline stages (preprocessing, modeling)
data/processed/          # intermediate artifacts
results/
  figures/               # plots (confusion matrix, examples)
  tables/                # metrics, predictions
```

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

python scripts/day2_window_and_features.py
python scripts/day3_baseline_model.py
```

## Limitations

- Uses a small subset of CHB-MIT rather than the full dataset
- Window-level labels are derived from seizure intervals rather than expert-reviewed events
- Uses simple handcrafted features and a baseline logistic-regression model
- Not intended for clinical use
