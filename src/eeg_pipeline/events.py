from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def smooth_probabilities(
    predictions: pd.DataFrame,
    probability_col: str,
    group_col: str = "file_name",
    sort_col: str = "window_start_seconds",
    smoothing_windows: int = 5,
) -> pd.DataFrame:
    if smoothing_windows <= 0:
        raise ValueError("smoothing_windows must be positive")

    smoothed_groups: list[pd.DataFrame] = []
    for _, group in predictions.groupby(group_col, sort=False):
        ordered = group.sort_values(sort_col).copy()
        ordered["smoothed_probability_seizure"] = (
            ordered[probability_col]
            .rolling(window=smoothing_windows, min_periods=1)
            .mean()
        )
        smoothed_groups.append(ordered)

    return pd.concat(smoothed_groups, ignore_index=True)


def apply_consecutive_window_threshold(
    labels: np.ndarray,
    min_consecutive_positive: int,
) -> np.ndarray:
    if min_consecutive_positive <= 1:
        return labels.astype(int)

    filtered = np.zeros_like(labels, dtype=int)
    run_start: int | None = None

    for index, value in enumerate(labels):
        if value and run_start is None:
            run_start = index
        elif not value and run_start is not None:
            run_length = index - run_start
            if run_length >= min_consecutive_positive:
                filtered[run_start:index] = 1
            run_start = None

    if run_start is not None:
        run_length = len(labels) - run_start
        if run_length >= min_consecutive_positive:
            filtered[run_start:] = 1

    return filtered


def classify_smoothed_windows(
    predictions: pd.DataFrame,
    threshold: float = 0.5,
    min_consecutive_positive: int = 3,
    group_col: str = "file_name",
    sort_col: str = "window_start_seconds",
) -> pd.DataFrame:
    classified_groups: list[pd.DataFrame] = []
    for _, group in predictions.groupby(group_col, sort=False):
        ordered = group.sort_values(sort_col).copy()
        ordered["smoothed_positive_raw"] = (
            ordered["smoothed_probability_seizure"] >= threshold
        ).astype(int)
        ordered["smoothed_positive_label"] = apply_consecutive_window_threshold(
            ordered["smoothed_positive_raw"].to_numpy(),
            min_consecutive_positive=min_consecutive_positive,
        )
        classified_groups.append(ordered)

    return pd.concat(classified_groups, ignore_index=True)


def windows_to_events(
    predictions: pd.DataFrame,
    label_col: str = "smoothed_positive_label",
    group_col: str = "file_name",
) -> pd.DataFrame:
    events: list[dict[str, float | int | str]] = []

    for file_name, group in predictions.groupby(group_col, sort=False):
        positives = group.loc[group[label_col] == 1].sort_values("window_start_seconds")
        if positives.empty:
            continue

        step_seconds = positives["window_start_seconds"].diff().dropna().median()
        if pd.isna(step_seconds):
            step_seconds = float(
                positives["window_end_seconds"].iloc[0] - positives["window_start_seconds"].iloc[0]
            )
        gap_tolerance_seconds = float(step_seconds)

        current_start = float(positives.iloc[0]["window_start_seconds"])
        current_end = float(positives.iloc[0]["window_end_seconds"])
        current_rows = [positives.iloc[0]]
        event_index = 0

        for _, row in positives.iloc[1:].iterrows():
            row_start = float(row["window_start_seconds"])
            row_end = float(row["window_end_seconds"])
            if row_start <= current_end + gap_tolerance_seconds:
                current_end = max(current_end, row_end)
                current_rows.append(row)
                continue

            event_frame = pd.DataFrame(current_rows)
            events.append(
                {
                    "file_name": file_name,
                    "predicted_event_id": event_index,
                    "event_start_seconds": current_start,
                    "event_end_seconds": current_end,
                    "duration_seconds": current_end - current_start,
                    "n_windows": len(event_frame),
                    "max_smoothed_probability": float(event_frame["smoothed_probability_seizure"].max()),
                    "mean_smoothed_probability": float(event_frame["smoothed_probability_seizure"].mean()),
                }
            )
            event_index += 1
            current_start = row_start
            current_end = row_end
            current_rows = [row]

        event_frame = pd.DataFrame(current_rows)
        events.append(
            {
                "file_name": file_name,
                "predicted_event_id": event_index,
                "event_start_seconds": current_start,
                "event_end_seconds": current_end,
                "duration_seconds": current_end - current_start,
                "n_windows": len(event_frame),
                "max_smoothed_probability": float(event_frame["smoothed_probability_seizure"].max()),
                "mean_smoothed_probability": float(event_frame["smoothed_probability_seizure"].mean()),
            }
        )

    return pd.DataFrame(events)


def true_events_from_intervals(
    seizure_intervals_by_file: dict[str, Iterable[tuple[float, float]]],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for file_name, intervals in seizure_intervals_by_file.items():
        for index, (start_seconds, end_seconds) in enumerate(intervals):
            rows.append(
                {
                    "file_name": file_name,
                    "true_event_id": index,
                    "event_start_seconds": float(start_seconds),
                    "event_end_seconds": float(end_seconds),
                    "duration_seconds": float(end_seconds) - float(start_seconds),
                }
            )
    return pd.DataFrame(rows)


def intervals_overlap(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> bool:
    return min(end_a, end_b) > max(start_a, start_b)


def evaluate_event_predictions(
    predicted_events: pd.DataFrame,
    true_events: pd.DataFrame,
    recording_durations_seconds: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    predicted = predicted_events.copy()
    true = true_events.copy()

    predicted["overlaps_true_event"] = False
    true["detected"] = False

    for true_index, true_row in true.iterrows():
        overlapping_predictions = predicted.loc[
            predicted["file_name"].eq(true_row["file_name"])
            & predicted.apply(
                lambda row: intervals_overlap(
                    float(row["event_start_seconds"]),
                    float(row["event_end_seconds"]),
                    float(true_row["event_start_seconds"]),
                    float(true_row["event_end_seconds"]),
                ),
                axis=1,
            )
        ]
        if not overlapping_predictions.empty:
            true.at[true_index, "detected"] = True
            predicted.loc[overlapping_predictions.index, "overlaps_true_event"] = True

    total_true_events = len(true)
    detected_true_events = int(true["detected"].sum())
    false_alarm_count = int((~predicted["overlaps_true_event"]).sum())
    total_duration_hours = sum(recording_durations_seconds.values()) / 3600.0

    metrics = {
        "total_true_events": float(total_true_events),
        "detected_true_events": float(detected_true_events),
        "detection_rate": (
            float(detected_true_events / total_true_events) if total_true_events > 0 else 0.0
        ),
        "false_alarm_count": float(false_alarm_count),
        "false_alarms_per_hour": (
            float(false_alarm_count / total_duration_hours) if total_duration_hours > 0 else 0.0
        ),
        "total_test_duration_hours": float(total_duration_hours),
    }

    return predicted, true, metrics
