# EEG Mini-Project

Minimal clinical EEG pipeline built around the CHB-MIT Scalp EEG Database.

## Current status

Day 1 is scaffolded:

- local Python environment with `mne`, `numpy`, `pandas`, `matplotlib`, and `scikit-learn`
- downloader for a tiny CHB-MIT subset
- summary parser for seizure timings
- EDF loading through MNE
- raw and filtered inspection plots for one seizure file and one non-seizure file

## Quick start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python scripts/day1_load_and_inspect.py
```

Outputs land in:

- `figures/`
- `data/processed/`
- `results/`

## Dataset subset used for Day 1

- Patient: `chb01`
- Non-seizure recording: `chb01_01.edf`
- Seizure recording: `chb01_03.edf`
- Source: CHB-MIT Scalp EEG Database on PhysioNet

## Notes

- The current script keeps preprocessing intentionally light: a `0.5-40 Hz` bandpass filter without aggressive artifact removal.
- The pipeline uses real CHB-MIT scalp EEG recordings from PhysioNet rather than simulated signals.
- Seizure-window labels are created by sliding fixed windows across each EDF and marking a window as seizure when it overlaps the annotated seizure interval from the patient summary file.
- The baseline classifier uses per-channel spectral and statistical features, including delta/theta/alpha/beta band power, mean, variance, and line length.
- Evaluation is constrained to a file-level split, so whole EDF files are assigned to train or test rather than randomly mixing windows across both sets.

## Results

The final subset uses eight real CHB-MIT `chb01` recordings: `chb01_01`, `03`, `04`, `05`, `06`, `15`, `16`, and `17`. The file-level split keeps whole EDFs intact, with train files `chb01_03`, `15`, `01`, `05` and test files `chb01_04`, `16`, `06`, `17`. On that meaningful split, the logistic-regression baseline reached precision `0.4815`, recall `0.6500`, and F1 `0.5532`. This is a simple baseline for portfolio-scale experimentation, not a clinical model.

## Limitations

This project uses a small subset of CHB-MIT rather than the full dataset. Window labels are derived from seizure time intervals, so each window inherits a coarse window-level label instead of event-level expert review. The feature set is limited to simple handcrafted spectral and statistical descriptors with logistic regression as the classifier. The pipeline is for educational and portfolio use only and is not intended for clinical deployment.
