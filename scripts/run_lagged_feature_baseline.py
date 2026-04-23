from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_pipeline.chbmit import load_recording, parse_summary, seizure_intervals_for_file
from eeg_pipeline.events import (
    classify_smoothed_windows,
    evaluate_event_predictions,
    smooth_probabilities,
    true_events_from_intervals,
    windows_to_events,
)
from eeg_pipeline.temporal import add_lagged_features


TRAIN_FILES = [
    "chb01_03.edf",
    "chb01_15.edf",
    "chb01_01.edf",
    "chb01_05.edf",
]

TEST_FILES = [
    "chb01_04.edf",
    "chb01_16.edf",
    "chb01_06.edf",
    "chb01_17.edf",
]

N_LAGS = 2
SMOOTHING_WINDOWS = 5
SMOOTHING_THRESHOLD = 0.5
MIN_CONSECUTIVE_POSITIVE_WINDOWS = 3
PATIENT_ID = "chb01"
EXAMPLE_CHANNEL = "FP1-F7"
EXAMPLE_PADDING_SECONDS = 20.0


def ensure_results_dir() -> tuple[Path, Path]:
    results_dir = ROOT / "results"
    figures_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, tables_dir


def validate_file_split(windows_metadata: pd.DataFrame) -> tuple[list[str], list[str]]:
    available_files = set(windows_metadata["file_name"].unique())
    train_files = [file_name for file_name in TRAIN_FILES if file_name in available_files]
    test_files = [file_name for file_name in TEST_FILES if file_name in available_files]

    if len(train_files) < 2 or len(test_files) < 2:
        raise ValueError("Meaningful split requires at least two train files and two test files")

    if set(train_files) & set(test_files):
        raise ValueError("Train and test file lists must not overlap")

    train_positive = int(windows_metadata.loc[windows_metadata["file_name"].isin(train_files), "label"].sum())
    test_positive = int(windows_metadata.loc[windows_metadata["file_name"].isin(test_files), "label"].sum())
    if train_positive == 0 or test_positive == 0:
        raise ValueError("Both train and test splits must contain seizure windows")

    return train_files, test_files


def plot_confusion_matrix(confusion: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(confusion, cmap="Greens")
    fig.colorbar(image, ax=ax)
    class_labels = ["non-seizure", "seizure"]
    ax.set_xticks([0, 1], labels=class_labels)
    ax.set_yticks([0, 1], labels=class_labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Lagged-Feature Baseline Confusion Matrix")
    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            ax.text(col, row, str(confusion[row, col]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_temporal_example(
    predictions: pd.DataFrame,
    true_events: pd.DataFrame,
    raw_data_dir: Path,
    output_path: Path,
) -> None:
    if true_events.empty:
        return

    example_event = true_events.sort_values("duration_seconds", ascending=False).iloc[0]
    file_name = str(example_event["file_name"])
    event_start = float(example_event["event_start_seconds"])
    event_end = float(example_event["event_end_seconds"])
    window_start = max(0.0, event_start - EXAMPLE_PADDING_SECONDS)
    window_end = event_end + EXAMPLE_PADDING_SECONDS

    raw = load_recording(raw_data_dir / file_name)
    channel = EXAMPLE_CHANNEL if EXAMPLE_CHANNEL in raw.ch_names else raw.ch_names[0]
    start_sample = int(window_start * raw.info["sfreq"])
    stop_sample = int(window_end * raw.info["sfreq"])
    eeg_signal, eeg_times = raw.get_data(
        picks=[channel],
        start=start_sample,
        stop=stop_sample,
        return_times=True,
    )
    eeg_signal_uv = eeg_signal[0] * 1e6

    segment_predictions = predictions.loc[
        predictions["file_name"].eq(file_name)
        & predictions["window_end_seconds"].ge(window_start)
        & predictions["window_start_seconds"].le(window_end)
    ].copy()
    segment_predictions["window_center_seconds"] = (
        segment_predictions["window_start_seconds"] + segment_predictions["window_end_seconds"]
    ) / 2.0

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    axes[0].plot(eeg_times, eeg_signal_uv, linewidth=0.8, color="tab:blue")
    axes[0].axvspan(event_start, event_end, color="crimson", alpha=0.18)
    axes[0].set_ylabel(f"{channel}\n(uV)")
    axes[0].set_title(f"Representative EEG segment with temporal predictions ({file_name})")
    axes[0].grid(alpha=0.2)

    axes[1].plot(
        segment_predictions["window_center_seconds"],
        segment_predictions["predicted_probability_seizure"],
        label="Raw probability",
        linewidth=1.0,
        alpha=0.65,
        color="tab:gray",
    )
    axes[1].plot(
        segment_predictions["window_center_seconds"],
        segment_predictions["smoothed_probability_seizure"],
        label="Smoothed probability",
        linewidth=1.5,
        color="tab:green",
    )
    axes[1].axhline(SMOOTHING_THRESHOLD, linestyle="--", linewidth=1.0, color="black", alpha=0.6)
    axes[1].axvspan(event_start, event_end, color="crimson", alpha=0.18)
    axes[1].set_ylabel("Probability")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.2)

    axes[2].step(
        segment_predictions["window_center_seconds"],
        segment_predictions["predicted_label"],
        where="mid",
        label="Raw predicted label",
        linewidth=1.0,
        color="tab:gray",
        alpha=0.7,
    )
    axes[2].step(
        segment_predictions["window_center_seconds"],
        segment_predictions["smoothed_positive_label"],
        where="mid",
        label="Smoothed event label",
        linewidth=1.5,
        color="tab:green",
    )
    axes[2].axvspan(event_start, event_end, color="crimson", alpha=0.18)
    axes[2].set_ylabel("Label")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_yticks([0, 1])
    axes[2].legend(loc="upper right")
    axes[2].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    processed_dir = ROOT / "data" / "processed"
    figures_dir, tables_dir = ensure_results_dir()

    features = pd.read_csv(processed_dir / "features.csv")
    windows_metadata = pd.read_csv(processed_dir / "windows_metadata.csv")

    data = features.merge(
        windows_metadata[["window_id", "file_name", "label", "window_start_seconds", "window_end_seconds"]],
        on=["window_id", "file_name", "label"],
        how="inner",
    )

    base_feature_columns = sorted([column for column in data.columns if "__" in column])
    if not base_feature_columns:
        raise ValueError(
            "No per-channel feature columns found. Re-run build_window_feature_dataset.py first."
        )

    temporal_data = add_lagged_features(
        data=data,
        feature_columns=base_feature_columns,
        n_lags=N_LAGS,
    )

    temporal_feature_columns = sorted(
        [column for column in temporal_data.columns if "__" in column]
    )

    train_files, test_files = validate_file_split(windows_metadata)
    train_mask = temporal_data["file_name"].isin(train_files)
    test_mask = temporal_data["file_name"].isin(test_files)

    x_train = temporal_data.loc[train_mask, temporal_feature_columns]
    y_train = temporal_data.loc[train_mask, "label"]
    x_test = temporal_data.loc[test_mask, temporal_feature_columns]
    y_test = temporal_data.loc[test_mask, "label"]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_score = model.predict_proba(x_test)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    confusion = confusion_matrix(y_test, y_pred, labels=[0, 1])

    predictions = temporal_data.loc[
        test_mask,
        ["window_id", "file_name", "window_start_seconds", "window_end_seconds", "label"],
    ].copy()
    predictions["predicted_label"] = y_pred
    predictions["predicted_probability_seizure"] = y_score
    predictions.to_csv(tables_dir / "lagged_feature_baseline_predictions.csv", index=False)

    plot_confusion_matrix(confusion, figures_dir / "lagged_feature_baseline_confusion_matrix.png")

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
    seizure_intervals_by_file = {
        file_name: seizure_intervals_for_file(summary_df, file_name)
        for file_name in sorted(predictions["file_name"].unique())
    }
    true_events = true_events_from_intervals(seizure_intervals_by_file)
    recording_durations_seconds = {
        file_name: float(group["window_end_seconds"].max())
        for file_name, group in smoothed.groupby("file_name")
    }
    scored_predicted_events, scored_true_events, event_metrics = evaluate_event_predictions(
        predicted_events=predicted_events,
        true_events=true_events,
        recording_durations_seconds=recording_durations_seconds,
    )
    smoothed.to_csv(tables_dir / "lagged_feature_baseline_smoothed_predictions.csv", index=False)
    scored_predicted_events.to_csv(tables_dir / "lagged_feature_baseline_event_predictions.csv", index=False)
    scored_true_events.to_csv(tables_dir / "lagged_feature_baseline_true_event_evaluation.csv", index=False)

    plot_temporal_example(
        predictions=smoothed,
        true_events=scored_true_events,
        raw_data_dir=ROOT / "data" / "raw" / "chbmit" / PATIENT_ID,
        output_path=figures_dir / "lagged_feature_temporal_example.png",
    )

    metrics_lines = [
        "Lagged-feature temporal baseline",
        f"Train files: {', '.join(train_files)}",
        f"Test files: {', '.join(test_files)}",
        f"Number of lags: {N_LAGS}",
        f"Base per-channel feature count: {len(base_feature_columns)}",
        f"Temporal feature count: {len(temporal_feature_columns)}",
        f"Training windows: {len(x_train)}",
        f"Test windows: {len(x_test)}",
        f"Precision: {precision:.4f}",
        f"Recall: {recall:.4f}",
        f"F1: {f1:.4f}",
        "Confusion matrix [[tn, fp], [fn, tp]]:",
        str(confusion.tolist()),
        f"Event detection rate: {event_metrics['detection_rate']:.4f}",
        f"False alarms per hour: {event_metrics['false_alarms_per_hour']:.4f}",
    ]
    (tables_dir / "lagged_feature_baseline_metrics.txt").write_text("\n".join(metrics_lines) + "\n")

    print("Saved outputs:")
    print("  FILE    results/tables/lagged_feature_baseline_predictions.csv")
    print("  FILE    results/tables/lagged_feature_baseline_smoothed_predictions.csv")
    print("  FILE    results/tables/lagged_feature_baseline_event_predictions.csv")
    print("  FILE    results/tables/lagged_feature_baseline_true_event_evaluation.csv")
    print("  FILE    results/tables/lagged_feature_baseline_metrics.txt")
    print("  FIGURE  results/figures/lagged_feature_baseline_confusion_matrix.png")
    print("  FIGURE  results/figures/lagged_feature_temporal_example.png")
    print(f"  PREC    {precision:.4f}")
    print(f"  REC     {recall:.4f}")
    print(f"  F1      {f1:.4f}")
    print(f"  E-DRATE {event_metrics['detection_rate']:.4f}")
    print(f"  E-FA/H  {event_metrics['false_alarms_per_hour']:.4f}")


if __name__ == "__main__":
    main()
