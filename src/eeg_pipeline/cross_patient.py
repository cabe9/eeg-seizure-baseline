from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_pipeline.chbmit import common_channels, ensure_subset, load_recording, parse_summary, seizure_intervals_for_file
from eeg_pipeline.features import extract_features
from eeg_pipeline.windowing import create_labeled_windows


@dataclass(frozen=True)
class ProcessedFeatureDataset:
    features: pd.DataFrame
    metadata: pd.DataFrame
    channel_names: list[str]
    summary_by_patient: dict[str, pd.DataFrame]


def shared_channels_for_patient_files(
    data_dir: Path,
    patient_files: dict[str, list[str]],
) -> list[str]:
    raw_recordings = []
    for patient_id, filenames in patient_files.items():
        files = ensure_subset(data_dir, patient_id, filenames)
        raw_recordings.extend(load_recording(files[filename]) for filename in filenames)
    return common_channels(raw_recordings)


def build_feature_dataset(
    data_dir: Path,
    patient_files: dict[str, list[str]],
    shared_channels: list[str],
    filter_low_hz: float,
    filter_high_hz: float,
    window_seconds: float,
    overlap_fraction: float,
) -> ProcessedFeatureDataset:
    windows_list: list[np.ndarray] = []
    metadata_list: list[pd.DataFrame] = []
    summary_by_patient: dict[str, pd.DataFrame] = {}

    for patient_id, filenames in patient_files.items():
        files = ensure_subset(data_dir, patient_id, filenames)
        summary_path = files[f"{patient_id}-summary.txt"]
        sampling_rate_hz, _, summary_df = parse_summary(summary_path)
        summary_by_patient[patient_id] = summary_df

        for filename in filenames:
            raw = load_recording(files[filename], pick_channels=shared_channels)
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

            if not np.isclose(filtered.info["sfreq"], sampling_rate_hz):
                raise ValueError(f"Inconsistent sampling rate for {filename}")

            windows_list.append(windows.astype(np.float32))
            metadata_list.append(metadata)

    all_windows = np.concatenate(windows_list, axis=0)
    all_metadata = pd.concat(metadata_list, ignore_index=True)
    all_metadata["window_id"] = range(len(all_metadata))

    features = extract_features(
        all_windows,
        sfreq=float(all_metadata["sampling_rate_hz"].iloc[0]),
        channel_names=shared_channels,
    )
    features = features.merge(
        all_metadata[
            [
                "window_id",
                "patient_id",
                "file_name",
                "label",
                "label_name",
                "window_start_seconds",
                "window_end_seconds",
            ]
        ],
        on="window_id",
        how="left",
    )

    return ProcessedFeatureDataset(
        features=features,
        metadata=all_metadata,
        channel_names=shared_channels,
        summary_by_patient=summary_by_patient,
    )
