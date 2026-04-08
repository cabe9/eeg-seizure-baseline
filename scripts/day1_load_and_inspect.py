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

from eeg_pipeline.chbmit import common_channels, ensure_subset, load_recording, parse_summary


PATIENT_ID = "chb01"
SEIZURE_FILE = "chb01_03.edf"
NON_SEIZURE_FILE = "chb01_01.edf"
PLOT_CHANNELS = ["FP1-F7", "F7-T7", "T7-P7", "FP1-F3", "F3-C3", "C3-P3"]
FILTER_LOW_HZ = 0.5
FILTER_HIGH_HZ = 40.0
SEGMENT_SECONDS = 20.0
NON_SEIZURE_START_SECONDS = 300.0


def ensure_dirs() -> dict[str, Path]:
    dirs = {
        "data_raw": ROOT / "data" / "raw" / "chbmit",
        "data_processed": ROOT / "data" / "processed",
        "figures": ROOT / "figures",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def segment_to_frame(raw, start_sec: float, duration_sec: float, channels: list[str]) -> pd.DataFrame:
    start_sample = int(start_sec * raw.info["sfreq"])
    stop_sample = int((start_sec + duration_sec) * raw.info["sfreq"])
    data, times = raw.get_data(
        picks=channels,
        start=start_sample,
        stop=stop_sample,
        return_times=True,
    )
    frame = pd.DataFrame(data.T * 1e6, columns=channels)
    frame.insert(0, "time_seconds", times)
    return frame


def plot_multichannel_segment(
    frame: pd.DataFrame,
    channels: list[str],
    title: str,
    output_path: Path,
    marker_time: float | None = None,
) -> None:
    fig, axes = plt.subplots(len(channels), 1, figsize=(14, 10), sharex=True)
    for index, channel in enumerate(channels):
        axes[index].plot(frame["time_seconds"], frame[channel], linewidth=0.8)
        if marker_time is not None:
            axes[index].axvline(marker_time, color="crimson", linestyle="--", linewidth=1.0)
        axes[index].set_ylabel(f"{channel}\n(uV)")
        axes[index].grid(alpha=0.2)
    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_filter_comparison(
    raw_frame: pd.DataFrame,
    filtered_frame: pd.DataFrame,
    channel: str,
    title: str,
    output_path: Path,
    marker_time: float | None = None,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(raw_frame["time_seconds"], raw_frame[channel], linewidth=0.8, color="tab:blue")
    if marker_time is not None:
        axes[0].axvline(marker_time, color="crimson", linestyle="--", linewidth=1.0)
    axes[0].set_title(f"{channel} raw")
    axes[0].set_ylabel("Amplitude (uV)")
    axes[0].grid(alpha=0.2)

    axes[1].plot(
        filtered_frame["time_seconds"],
        filtered_frame[channel],
        linewidth=0.8,
        color="tab:orange",
    )
    if marker_time is not None:
        axes[1].axvline(marker_time, color="crimson", linestyle="--", linewidth=1.0)
    axes[1].set_title(f"{channel} bandpass filtered ({FILTER_LOW_HZ}-{FILTER_HIGH_HZ} Hz)")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].set_ylabel("Amplitude (uV)")
    axes[1].grid(alpha=0.2)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_psd(raw, channels: list[str], output_path: Path) -> None:
    figure = raw.compute_psd(fmin=0.5, fmax=45.0, picks=channels).plot(
        average=True,
        show=False,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    dirs = ensure_dirs()

    files = ensure_subset(
        data_dir=dirs["data_raw"],
        patient_id=PATIENT_ID,
        filenames=[NON_SEIZURE_FILE, SEIZURE_FILE],
    )
    summary_path = files[f"{PATIENT_ID}-summary.txt"]

    sampling_rate, summary_channels, summary_df = parse_summary(summary_path)
    summary_df.to_csv(dirs["data_processed"] / "chb01_recording_summary.csv", index=False)

    metadata = summary_df.loc[
        summary_df["filename"].isin([NON_SEIZURE_FILE, SEIZURE_FILE])
    ].copy()
    metadata["sampling_rate_hz"] = sampling_rate
    metadata["channels_listed_in_summary"] = len(summary_channels)
    metadata.to_csv(dirs["data_processed"] / "day1_subset_metadata.csv", index=False)

    seizure_meta = summary_df.loc[summary_df["filename"] == SEIZURE_FILE].iloc[0]
    seizure_start = float(seizure_meta["seizure_start_seconds"])
    seizure_window_start = max(seizure_start - 10.0, 0.0)

    raw_non = load_recording(files[NON_SEIZURE_FILE])
    raw_seizure = load_recording(files[SEIZURE_FILE])

    shared_channels = common_channels([raw_non, raw_seizure])
    channels = [channel for channel in PLOT_CHANNELS if channel in shared_channels]

    raw_non.pick(channels)
    raw_seizure.pick(channels)

    filtered_non = raw_non.copy().filter(FILTER_LOW_HZ, FILTER_HIGH_HZ, verbose="ERROR")
    filtered_seizure = raw_seizure.copy().filter(FILTER_LOW_HZ, FILTER_HIGH_HZ, verbose="ERROR")

    non_frame = segment_to_frame(raw_non, NON_SEIZURE_START_SECONDS, SEGMENT_SECONDS, channels)
    seizure_frame = segment_to_frame(raw_seizure, seizure_window_start, SEGMENT_SECONDS, channels)
    seizure_filtered_frame = segment_to_frame(
        filtered_seizure,
        seizure_window_start,
        SEGMENT_SECONDS,
        channels,
    )

    plot_multichannel_segment(
        frame=non_frame,
        channels=channels,
        title=f"{PATIENT_ID} non-seizure segment ({NON_SEIZURE_FILE})",
        output_path=dirs["figures"] / "day1_non_seizure_raw.png",
    )
    plot_multichannel_segment(
        frame=seizure_frame,
        channels=channels,
        title=f"{PATIENT_ID} seizure-onset segment ({SEIZURE_FILE})",
        output_path=dirs["figures"] / "day1_seizure_raw.png",
        marker_time=seizure_start,
    )
    plot_filter_comparison(
        raw_frame=seizure_frame,
        filtered_frame=seizure_filtered_frame,
        channel=channels[0],
        title=f"{PATIENT_ID} seizure segment raw vs filtered",
        output_path=dirs["figures"] / "day1_filter_comparison.png",
        marker_time=seizure_start,
    )
    plot_psd(
        raw=filtered_seizure,
        channels=channels,
        output_path=dirs["figures"] / "day1_filtered_psd.png",
    )

    run_summary = pd.DataFrame(
        [
            {
                "patient_id": PATIENT_ID,
                "sampling_rate_hz": sampling_rate,
                "edf_files_downloaded": 2,
                "shared_channel_count": len(shared_channels),
                "channels_used_for_plots": ", ".join(channels),
                "non_seizure_file": NON_SEIZURE_FILE,
                "non_seizure_start_seconds": NON_SEIZURE_START_SECONDS,
                "seizure_file": SEIZURE_FILE,
                "seizure_start_seconds": seizure_start,
                "plot_window_seconds": SEGMENT_SECONDS,
                "filter_band_hz": f"{FILTER_LOW_HZ}-{FILTER_HIGH_HZ}",
            }
        ]
    )
    run_summary.to_csv(dirs["data_processed"] / "day1_run_summary.csv", index=False)

    print("Saved outputs:")
    for path in sorted(dirs["figures"].glob("day1_*")):
        print(f"  FIGURE  {path.relative_to(ROOT)}")
    for path in sorted(dirs["data_processed"].glob("day1_*")):
        print(f"  TABLE   {path.relative_to(ROOT)}")
    print(f"  TABLE   {(dirs['data_processed'] / 'chb01_recording_summary.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
