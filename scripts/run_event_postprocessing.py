from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_pipeline.chbmit import parse_summary, seizure_intervals_for_file
from eeg_pipeline.events import (
    classify_smoothed_windows,
    evaluate_event_predictions,
    smooth_probabilities,
    true_events_from_intervals,
    windows_to_events,
)


SMOOTHING_WINDOWS = 5
SMOOTHING_THRESHOLD = 0.5
MIN_CONSECUTIVE_POSITIVE_WINDOWS = 3
PATIENT_ID = "chb01"


def ensure_results_dirs() -> tuple[Path, Path]:
    figures_dir = ROOT / "results" / "figures"
    tables_dir = ROOT / "results" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, tables_dir


def main() -> None:
    _figures_dir, tables_dir = ensure_results_dirs()

    predictions = pd.read_csv(ROOT / "results" / "tables" / "logistic_baseline_predictions.csv")
    summary_path = ROOT / "data" / "raw" / "chbmit" / PATIENT_ID / f"{PATIENT_ID}-summary.txt"
    _, _, summary_df = parse_summary(summary_path)

    smoothed = smooth_probabilities(
        predictions=predictions,
        probability_col="predicted_probability_seizure",
        smoothing_windows=SMOOTHING_WINDOWS,
    )
    smoothed = classify_smoothed_windows(
        smoothed,
        threshold=SMOOTHING_THRESHOLD,
        min_consecutive_positive=MIN_CONSECUTIVE_POSITIVE_WINDOWS,
    )

    predicted_events = windows_to_events(smoothed)
    test_files = sorted(smoothed["file_name"].unique())
    seizure_intervals_by_file = {
        file_name: seizure_intervals_for_file(summary_df, file_name)
        for file_name in test_files
    }
    true_events = true_events_from_intervals(seizure_intervals_by_file)
    recording_durations_seconds = {
        file_name: float(group["window_end_seconds"].max())
        for file_name, group in smoothed.groupby("file_name")
    }

    scored_predicted_events, scored_true_events, metrics = evaluate_event_predictions(
        predicted_events=predicted_events,
        true_events=true_events,
        recording_durations_seconds=recording_durations_seconds,
    )

    smoothed.to_csv(tables_dir / "event_postprocessing_smoothed_predictions.csv", index=False)
    scored_predicted_events.to_csv(tables_dir / "event_postprocessing_event_predictions.csv", index=False)
    scored_true_events.to_csv(tables_dir / "event_postprocessing_true_event_evaluation.csv", index=False)

    metrics_lines = [
        "Temporal smoothing and event-level evaluation",
        f"Smoothing windows: {SMOOTHING_WINDOWS}",
        f"Smoothing threshold: {SMOOTHING_THRESHOLD:.2f}",
        f"Minimum consecutive positive windows: {MIN_CONSECUTIVE_POSITIVE_WINDOWS}",
        f"Test files: {', '.join(test_files)}",
        f"Total true seizure events: {int(metrics['total_true_events'])}",
        f"Detected true seizure events: {int(metrics['detected_true_events'])}",
        f"Detection rate: {metrics['detection_rate']:.4f}",
        f"False alarm count: {int(metrics['false_alarm_count'])}",
        f"False alarms per hour: {metrics['false_alarms_per_hour']:.4f}",
        f"Total test duration (hours): {metrics['total_test_duration_hours']:.4f}",
    ]
    (tables_dir / "event_postprocessing_metrics.txt").write_text("\n".join(metrics_lines) + "\n")

    print("Saved outputs:")
    print("  FILE    results/tables/event_postprocessing_smoothed_predictions.csv")
    print("  FILE    results/tables/event_postprocessing_event_predictions.csv")
    print("  FILE    results/tables/event_postprocessing_true_event_evaluation.csv")
    print("  FILE    results/tables/event_postprocessing_metrics.txt")
    print(f"  DETECT  {int(metrics['detected_true_events'])}/{int(metrics['total_true_events'])}")
    print(f"  DRATE   {metrics['detection_rate']:.4f}")
    print(f"  FA/H    {metrics['false_alarms_per_hour']:.4f}")


if __name__ == "__main__":
    main()
