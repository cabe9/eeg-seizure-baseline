from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_pipeline.chbmit import (
    common_channels,
    ensure_subset,
    load_recording,
    parse_summary,
    seizure_intervals_for_file,
)
from eeg_pipeline.features import extract_features
from eeg_pipeline.windowing import create_labeled_windows


PATIENT_ID = "chb01"
FILES = [
    "chb01_01.edf",
    "chb01_03.edf",
    "chb01_04.edf",
    "chb01_05.edf",
    "chb01_06.edf",
    "chb01_15.edf",
    "chb01_16.edf",
    "chb01_17.edf",
]
FILTER_LOW_HZ = 0.5
FILTER_HIGH_HZ = 40.0
WINDOW_SECONDS = 2.0
OVERLAP_FRACTION = 0.5
PLOT_CHANNELS = ["FP1-F7", "F7-T7", "T7-P7", "FP1-F3", "F3-C3", "C3-P3"]


def ensure_dirs() -> dict[str, Path]:
    dirs = {
        "data_raw": ROOT / "data" / "raw" / "chbmit",
        "data_processed": ROOT / "data" / "processed",
        "figures": ROOT / "figures",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def plot_window(
    raw,
    metadata_row: pd.Series,
    channels: list[str],
    title: str,
    output_path: Path,
    seizure_intervals: list[tuple[float, float]],
) -> None:
    start_seconds = float(metadata_row["window_start_seconds"])
    end_seconds = float(metadata_row["window_end_seconds"])
    start_sample = int(start_seconds * raw.info["sfreq"])
    stop_sample = int(end_seconds * raw.info["sfreq"])

    data, times = raw.get_data(picks=channels, start=start_sample, stop=stop_sample, return_times=True)
    data_uv = data * 1e6

    fig, axes = plt.subplots(len(channels), 1, figsize=(12, 9), sharex=True)
    for index, channel in enumerate(channels):
        axes[index].plot(times, data_uv[index], linewidth=0.9)
        for seizure_start, seizure_end in seizure_intervals:
            overlap_start = max(start_seconds, seizure_start)
            overlap_end = min(end_seconds, seizure_end)
            if overlap_end > overlap_start:
                axes[index].axvspan(overlap_start, overlap_end, color="crimson", alpha=0.18)
        axes[index].set_ylabel(f"{channel}\n(uV)")
        axes[index].grid(alpha=0.2)

    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    dirs = ensure_dirs()
    files = ensure_subset(dirs["data_raw"], PATIENT_ID, FILES)
    summary_path = files[f"{PATIENT_ID}-summary.txt"]
    sampling_rate, _, summary_df = parse_summary(summary_path)

    raw_recordings = {filename: load_recording(files[filename]) for filename in FILES}
    shared_channels = common_channels(raw_recordings.values())
    plot_channels = [channel for channel in PLOT_CHANNELS if channel in shared_channels]

    windows_list = []
    metadata_list = []

    for filename in FILES:
        raw = raw_recordings[filename]
        raw.pick(shared_channels)
        filtered = raw.copy().filter(FILTER_LOW_HZ, FILTER_HIGH_HZ, verbose="ERROR")
        seizure_intervals = seizure_intervals_for_file(summary_df, filename)

        windows, labels, metadata = create_labeled_windows(
            data=filtered.get_data(),
            sfreq=filtered.info["sfreq"],
            channel_names=filtered.ch_names,
            patient_id=PATIENT_ID,
            filename=filename,
            seizure_intervals=seizure_intervals,
            window_seconds=WINDOW_SECONDS,
            overlap_fraction=OVERLAP_FRACTION,
        )
        metadata["sampling_rate_hz"] = filtered.info["sfreq"]
        metadata["channel_names"] = ", ".join(filtered.ch_names)
        metadata["label_name"] = metadata["label"].map({0: "non_seizure", 1: "seizure"})

        windows_list.append(windows)
        metadata_list.append(metadata)

        print(
            f"{filename}: windows={len(labels)} "
            f"seizure={int((labels == 1).sum())} "
            f"non_seizure={int((labels == 0).sum())}"
        )

    concatenated_windows = np.concatenate(windows_list, axis=0)
    windows_metadata = pd.concat(metadata_list, ignore_index=True)
    windows_metadata["window_id"] = range(len(windows_metadata))

    features = extract_features(
        concatenated_windows,
        sfreq=sampling_rate,
        channel_names=shared_channels,
    )
    features = features.merge(
        windows_metadata[["window_id", "patient_id", "file_name", "label", "label_name"]],
        on="window_id",
        how="left",
    )

    windows_metadata.to_csv(dirs["data_processed"] / "windows_metadata.csv", index=False)
    features.to_csv(dirs["data_processed"] / "features.csv", index=False)

    class_counts = windows_metadata["label"].value_counts().sort_index()
    print("\nClass balance")
    print(f"  non-seizure windows: {int(class_counts.get(0, 0))}")
    print(f"  seizure windows: {int(class_counts.get(1, 0))}")

    seizure_example = windows_metadata.loc[windows_metadata["label"] == 1].iloc[0]
    non_seizure_example = windows_metadata.loc[windows_metadata["label"] == 0].iloc[0]

    seizure_file = seizure_example["file_name"]
    non_seizure_file = non_seizure_example["file_name"]

    plot_window(
        raw=raw_recordings[seizure_file],
        metadata_row=seizure_example,
        channels=plot_channels,
        title=(
            f"Seizure-labeled window | {seizure_file} | "
            f"{seizure_example['window_start_seconds']:.1f}-{seizure_example['window_end_seconds']:.1f}s"
        ),
        output_path=dirs["figures"] / "window_label_seizure_check.png",
        seizure_intervals=seizure_intervals_for_file(summary_df, seizure_file),
    )
    plot_window(
        raw=raw_recordings[non_seizure_file],
        metadata_row=non_seizure_example,
        channels=plot_channels,
        title=(
            f"Non-seizure window | {non_seizure_file} | "
            f"{non_seizure_example['window_start_seconds']:.1f}-{non_seizure_example['window_end_seconds']:.1f}s"
        ),
        output_path=dirs["figures"] / "window_label_non_seizure_check.png",
        seizure_intervals=seizure_intervals_for_file(summary_df, non_seizure_file),
    )

    print("\nSaved outputs:")
    print("  TABLE   data/processed/windows_metadata.csv")
    print("  TABLE   data/processed/features.csv")
    print("  FIGURE  figures/window_label_seizure_check.png")
    print("  FIGURE  figures/window_label_non_seizure_check.png")
    print("\nSplit note: file-level train/test splitting is intentionally not implemented yet.")


if __name__ == "__main__":
    main()
