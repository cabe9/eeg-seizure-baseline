from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def compute_window_step(window_seconds: float, overlap_fraction: float) -> float:
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if not 0 <= overlap_fraction < 1:
        raise ValueError("overlap_fraction must be in [0, 1)")

    step_seconds = window_seconds * (1.0 - overlap_fraction)
    if step_seconds <= 0:
        raise ValueError("overlap_fraction results in a non-positive step size")
    return step_seconds


def fixed_window_array(
    data: np.ndarray,
    sfreq: float,
    window_seconds: float,
    overlap_fraction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if data.ndim != 2:
        raise ValueError("data must have shape (n_channels, n_samples)")

    window_samples = int(round(window_seconds * sfreq))
    step_samples = int(round(compute_window_step(window_seconds, overlap_fraction) * sfreq))
    if window_samples <= 0 or step_samples <= 0:
        raise ValueError("window and step sizes must be positive")
    if data.shape[1] < window_samples:
        raise ValueError("recording is shorter than one window")

    starts = np.arange(0, data.shape[1] - window_samples + 1, step_samples, dtype=int)
    stops = starts + window_samples
    windows = np.stack([data[:, start:stop] for start, stop in zip(starts, stops)], axis=0)

    start_seconds = starts / sfreq
    end_seconds = stops / sfreq
    return windows, start_seconds, end_seconds


def label_windows_by_intervals(
    start_seconds: np.ndarray,
    end_seconds: np.ndarray,
    seizure_intervals: Iterable[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    intervals = list(seizure_intervals)
    labels = np.zeros(len(start_seconds), dtype=int)
    overlap_seconds = np.zeros(len(start_seconds), dtype=float)

    for index, (window_start, window_end) in enumerate(zip(start_seconds, end_seconds)):
        total_overlap = 0.0
        for seizure_start, seizure_end in intervals:
            overlap = max(0.0, min(window_end, seizure_end) - max(window_start, seizure_start))
            total_overlap += overlap

        overlap_seconds[index] = total_overlap
        labels[index] = int(total_overlap > 0.0)

    return labels, overlap_seconds


def create_labeled_windows(
    data: np.ndarray,
    sfreq: float,
    channel_names: list[str],
    patient_id: str,
    filename: str,
    seizure_intervals: Iterable[tuple[float, float]],
    window_seconds: float,
    overlap_fraction: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    windows, start_seconds, end_seconds = fixed_window_array(
        data=data,
        sfreq=sfreq,
        window_seconds=window_seconds,
        overlap_fraction=overlap_fraction,
    )
    labels, overlap_seconds = label_windows_by_intervals(
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        seizure_intervals=seizure_intervals,
    )

    metadata = pd.DataFrame(
        {
            "window_id": np.arange(len(windows), dtype=int),
            "patient_id": patient_id,
            "file_name": filename,
            "window_start_seconds": start_seconds,
            "window_end_seconds": end_seconds,
            "window_duration_seconds": window_seconds,
            "overlap_fraction": overlap_fraction,
            "n_channels": len(channel_names),
            "n_samples_per_window": windows.shape[2],
            "label": labels,
            "seizure_overlap_seconds": overlap_seconds,
        }
    )
    return windows, labels, metadata
