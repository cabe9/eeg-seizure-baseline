from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_pipeline.cross_patient import build_feature_dataset, shared_channels_for_patient_files
from eeg_pipeline.events import (
    classify_smoothed_windows,
    evaluate_event_predictions,
    smooth_probabilities,
    true_events_from_intervals,
    windows_to_events,
)
from eeg_pipeline.torch_data import (
    apply_channel_standardizer,
    build_raw_window_dataset,
    fit_channel_standardizer,
)
from eeg_pipeline.torch_models import EEGWindowCNN
from eeg_pipeline.chbmit import seizure_intervals_for_file


TRAIN_PATIENT_ID = "chb01"
TRAIN_FILES = [
    "chb01_01.edf",
    "chb01_03.edf",
    "chb01_04.edf",
    "chb01_05.edf",
    "chb01_06.edf",
    "chb01_15.edf",
    "chb01_16.edf",
    "chb01_17.edf",
]
TEST_PATIENT_ID = "chb02"
TEST_FILES = [
    "chb02_16.edf",
    "chb02_16+.edf",
    "chb02_17.edf",
    "chb02_19.edf",
]
PATIENT_FILES = {
    TRAIN_PATIENT_ID: TRAIN_FILES,
    TEST_PATIENT_ID: TEST_FILES,
}
FILTER_LOW_HZ = 0.5
FILTER_HIGH_HZ = 40.0
WINDOW_SECONDS = 2.0
OVERLAP_FRACTION = 0.5
SMOOTHING_WINDOWS = 5
SMOOTHING_THRESHOLD = 0.5
MIN_CONSECUTIVE_POSITIVE_WINDOWS = 3
SEED = 7
EPOCHS = 12
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_results_dir() -> tuple[Path, Path]:
    results_dir = ROOT / "results"
    figures_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, tables_dir


def plot_confusion_matrix(confusion: np.ndarray, output_path: Path, title: str, cmap: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(confusion, cmap=cmap)
    fig.colorbar(image, ax=ax)
    class_labels = ["non-seizure", "seizure"]
    ax.set_xticks([0, 1], labels=class_labels)
    ax.set_yticks([0, 1], labels=class_labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            ax.text(col, row, str(confusion[row, col]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate_temporal_predictions(
    predictions: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
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
    return smoothed, scored_predicted_events, scored_true_events, event_metrics


def save_outputs(
    prefix: str,
    model_label: str,
    predictions: pd.DataFrame,
    confusion: np.ndarray,
    precision: float,
    recall: float,
    f1: float,
    event_metrics: dict[str, float],
    smoothed: pd.DataFrame,
    predicted_events: pd.DataFrame,
    true_event_scores: pd.DataFrame,
    figures_dir: Path,
    tables_dir: Path,
    confusion_cmap: str,
) -> None:
    predictions.to_csv(tables_dir / f"{prefix}_predictions.csv", index=False)
    smoothed.to_csv(tables_dir / f"{prefix}_smoothed_predictions.csv", index=False)
    predicted_events.to_csv(tables_dir / f"{prefix}_event_predictions.csv", index=False)
    true_event_scores.to_csv(tables_dir / f"{prefix}_true_event_evaluation.csv", index=False)
    plot_confusion_matrix(
        confusion,
        figures_dir / f"{prefix}_confusion_matrix.png",
        title=f"{model_label} Cross-Patient Confusion Matrix",
        cmap=confusion_cmap,
    )

    metrics_lines = [
        model_label,
        f"Train patient: {TRAIN_PATIENT_ID}",
        f"Test patient: {TEST_PATIENT_ID}",
        f"Train files: {', '.join(TRAIN_FILES)}",
        f"Test files: {', '.join(TEST_FILES)}",
        f"Smoothing windows: {SMOOTHING_WINDOWS}",
        f"Smoothing threshold: {SMOOTHING_THRESHOLD}",
        f"Minimum consecutive positive windows: {MIN_CONSECUTIVE_POSITIVE_WINDOWS}",
        f"Precision: {precision:.4f}",
        f"Recall: {recall:.4f}",
        f"F1: {f1:.4f}",
        "Confusion matrix [[tn, fp], [fn, tp]]:",
        str(confusion.tolist()),
        f"True held-out seizure events: {int(event_metrics['total_true_events'])}",
        f"Detected held-out seizure events: {int(event_metrics['detected_true_events'])}",
        f"Event detection rate: {event_metrics['detection_rate']:.4f}",
        f"False alarm count: {int(event_metrics['false_alarm_count'])}",
        f"False alarms per hour: {event_metrics['false_alarms_per_hour']:.4f}",
        "Leakage note: feature scaling and raw-window standardization are fit on chb01 only; smoothing is causal and applied after prediction.",
    ]
    (tables_dir / f"{prefix}_metrics.txt").write_text("\n".join(metrics_lines) + "\n")


def run_logistic_baseline(
    feature_data,
    figures_dir: Path,
    tables_dir: Path,
) -> None:
    data = feature_data.features.copy()
    per_channel_columns = sorted([column for column in data.columns if "__" in column])
    if not per_channel_columns:
        raise ValueError("No per-channel feature columns found for logistic baseline")

    train_mask = data["patient_id"].eq(TRAIN_PATIENT_ID)
    test_mask = data["patient_id"].eq(TEST_PATIENT_ID)

    x_train = data.loc[train_mask, per_channel_columns]
    y_train = data.loc[train_mask, "label"]
    x_test = data.loc[test_mask, per_channel_columns]
    y_test = data.loc[test_mask, "label"].to_numpy()

    if y_train.nunique() < 2:
        raise ValueError("Cross-patient training split contains fewer than two classes")
    if y_test.sum() == 0:
        raise ValueError("Cross-patient test split contains no seizure windows")

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

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions_binary = model.predict(x_test)
    confusion = confusion_matrix(y_test, predictions_binary, labels=[0, 1])
    precision = precision_score(y_test, predictions_binary, zero_division=0)
    recall = recall_score(y_test, predictions_binary, zero_division=0)
    f1 = f1_score(y_test, predictions_binary, zero_division=0)

    predictions = data.loc[
        test_mask,
        ["window_id", "patient_id", "file_name", "window_start_seconds", "window_end_seconds", "label"],
    ].copy()
    predictions["predicted_label"] = predictions_binary
    predictions["predicted_probability_seizure"] = probabilities

    smoothed, predicted_events, true_event_scores, event_metrics = evaluate_temporal_predictions(
        predictions=predictions,
        summary_df=feature_data.summary_by_patient[TEST_PATIENT_ID],
    )
    save_outputs(
        prefix="cross_patient_logistic",
        model_label="Cross-patient logistic baseline",
        predictions=predictions,
        confusion=confusion,
        precision=precision,
        recall=recall,
        f1=f1,
        event_metrics=event_metrics,
        smoothed=smoothed,
        predicted_events=predicted_events,
        true_event_scores=true_event_scores,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        confusion_cmap="Blues",
    )

    print(
        "Logistic baseline complete: "
        f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} "
        f"event_detection={event_metrics['detected_true_events']:.0f}/{event_metrics['total_true_events']:.0f} "
        f"false_alarms_per_hour={event_metrics['false_alarms_per_hour']:.4f}"
    )


def run_pytorch_cnn(
    shared_channels: list[str],
    figures_dir: Path,
    tables_dir: Path,
    summary_df: pd.DataFrame,
) -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    train_dataset = build_raw_window_dataset(
        data_dir=ROOT / "data" / "raw" / "chbmit",
        patient_id=TRAIN_PATIENT_ID,
        filenames=TRAIN_FILES,
        filter_low_hz=FILTER_LOW_HZ,
        filter_high_hz=FILTER_HIGH_HZ,
        window_seconds=WINDOW_SECONDS,
        overlap_fraction=OVERLAP_FRACTION,
        pick_channels=shared_channels,
        normalize=False,
    )
    test_dataset = build_raw_window_dataset(
        data_dir=ROOT / "data" / "raw" / "chbmit",
        patient_id=TEST_PATIENT_ID,
        filenames=TEST_FILES,
        filter_low_hz=FILTER_LOW_HZ,
        filter_high_hz=FILTER_HIGH_HZ,
        window_seconds=WINDOW_SECONDS,
        overlap_fraction=OVERLAP_FRACTION,
        pick_channels=shared_channels,
        normalize=False,
    )

    standardizer = fit_channel_standardizer(train_dataset.windows)
    x_train_np = apply_channel_standardizer(train_dataset.windows, standardizer)
    x_test_np = apply_channel_standardizer(test_dataset.windows, standardizer)

    x_train = torch.tensor(x_train_np, dtype=torch.float32)
    y_train = torch.tensor(train_dataset.metadata["label"].to_numpy(), dtype=torch.float32)
    x_test = torch.tensor(x_test_np, dtype=torch.float32)
    y_test = test_dataset.metadata["label"].to_numpy()

    if y_train.sum() == 0 or y_test.sum() == 0:
        raise ValueError("Cross-patient CNN split requires seizure windows in both train and test")

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = EEGWindowCNN(
        n_channels=x_train.shape[1],
        n_samples=x_train.shape[2],
    ).to(device)

    positive_count = float(y_train.sum().item())
    negative_count = float(len(y_train) - positive_count)
    pos_weight = torch.tensor([negative_count / max(positive_count, 1.0)], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for _ in range(EPOCHS):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(x_test.to(device)).cpu().numpy()
        probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()
        predictions_binary = (probabilities >= 0.5).astype(int)

    confusion = confusion_matrix(y_test, predictions_binary, labels=[0, 1])
    precision = precision_score(y_test, predictions_binary, zero_division=0)
    recall = recall_score(y_test, predictions_binary, zero_division=0)
    f1 = f1_score(y_test, predictions_binary, zero_division=0)

    predictions = test_dataset.metadata[
        ["window_id", "patient_id", "file_name", "window_start_seconds", "window_end_seconds", "label"]
    ].copy()
    predictions["predicted_label"] = predictions_binary
    predictions["predicted_probability_seizure"] = probabilities

    smoothed, predicted_events, true_event_scores, event_metrics = evaluate_temporal_predictions(
        predictions=predictions,
        summary_df=summary_df,
    )
    save_outputs(
        prefix="cross_patient_pytorch_cnn",
        model_label="Cross-patient PyTorch CNN baseline",
        predictions=predictions,
        confusion=confusion,
        precision=precision,
        recall=recall,
        f1=f1,
        event_metrics=event_metrics,
        smoothed=smoothed,
        predicted_events=predicted_events,
        true_event_scores=true_event_scores,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        confusion_cmap="Oranges",
    )

    print(
        "PyTorch CNN complete: "
        f"precision={precision:.4f} recall={recall:.4f} f1={f1:.4f} "
        f"event_detection={event_metrics['detected_true_events']:.0f}/{event_metrics['total_true_events']:.0f} "
        f"false_alarms_per_hour={event_metrics['false_alarms_per_hour']:.4f}"
    )


def main() -> None:
    set_seed(SEED)
    figures_dir, tables_dir = ensure_results_dir()
    data_dir = ROOT / "data" / "raw" / "chbmit"

    shared_channels = shared_channels_for_patient_files(
        data_dir=data_dir,
        patient_files=PATIENT_FILES,
    )
    feature_data = build_feature_dataset(
        data_dir=data_dir,
        patient_files=PATIENT_FILES,
        shared_channels=shared_channels,
        filter_low_hz=FILTER_LOW_HZ,
        filter_high_hz=FILTER_HIGH_HZ,
        window_seconds=WINDOW_SECONDS,
        overlap_fraction=OVERLAP_FRACTION,
    )

    run_logistic_baseline(feature_data=feature_data, figures_dir=figures_dir, tables_dir=tables_dir)
    run_pytorch_cnn(
        shared_channels=shared_channels,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        summary_df=feature_data.summary_by_patient[TEST_PATIENT_ID],
    )

    print("Leakage guardrails:")
    print("  - train/test separation is by patient and therefore by file")
    print("  - logistic scaling is fit on chb01 only")
    print("  - CNN raw-window standardization is fit on chb01 only and applied unchanged to chb02")
    print("  - temporal smoothing is trailing-only and applied after model prediction")


if __name__ == "__main__":
    main()
