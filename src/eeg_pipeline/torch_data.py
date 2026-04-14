from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_pipeline.chbmit import common_channels, ensure_subset, load_recording, parse_summary, seizure_intervals_for_file
from eeg_pipeline.windowing import create_labeled_windows


@dataclass(frozen=True)
class RawWindowDataset:
    windows: np.ndarray
    metadata: pd.DataFrame
    channel_names: list[str]
    sampling_rate_hz: float


def build_raw_window_dataset(
    data_dir: Path,
    patient_id: str,
    filenames: list[str],
    filter_low_hz: float,
    filter_high_hz: float,
    window_seconds: float,
    overlap_fraction: float,
) -> RawWindowDataset:
    files = ensure_subset(data_dir, patient_id, filenames)
    summary_path = files[f"{patient_id}-summary.txt"]
    sampling_rate_hz, _, summary_df = parse_summary(summary_path)

    raw_recordings = {filename: load_recording(files[filename]) for filename in filenames}
    shared_channels = common_channels(raw_recordings.values())

    windows_list: list[np.ndarray] = []
    metadata_list: list[pd.DataFrame] = []

    for filename in filenames:
        raw = raw_recordings[filename]
        raw.pick(shared_channels)
        filtered = raw.copy().filter(filter_low_hz, filter_high_hz, verbose="ERROR")
        seizure_intervals = seizure_intervals_for_file(summary_df, filename)

        windows, _, metadata = create_labeled_windows(
            data=filtered.get_data(),
            sfreq=filtered.info["sfreq"],
            channel_names=filtered.ch_names,
            patient_id=patient_id,
            filename=filename,
            seizure_intervals=seizure_intervals,
            window_seconds=window_seconds,
            overlap_fraction=overlap_fraction,
        )
        metadata["sampling_rate_hz"] = filtered.info["sfreq"]
        metadata["channel_names"] = ", ".join(filtered.ch_names)
        metadata["label_name"] = metadata["label"].map({0: "non_seizure", 1: "seizure"})

        windows_list.append(windows.astype(np.float32))
        metadata_list.append(metadata)

    all_windows = np.concatenate(windows_list, axis=0)
    all_metadata = pd.concat(metadata_list, ignore_index=True)
    all_metadata["window_id"] = range(len(all_metadata))

    # Per-channel z-score using training data later is ideal; for raw amplitudes this global
    # standardization keeps optimization stable while preserving waveform shape.
    channel_mean = all_windows.mean(axis=(0, 2), keepdims=True)
    channel_std = all_windows.std(axis=(0, 2), keepdims=True)
    all_windows = (all_windows - channel_mean) / np.clip(channel_std, a_min=1e-6, a_max=None)

    return RawWindowDataset(
        windows=all_windows,
        metadata=all_metadata,
        channel_names=shared_channels,
        sampling_rate_hz=sampling_rate_hz,
    )
