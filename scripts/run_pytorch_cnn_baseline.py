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
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
from eeg_pipeline.torch_data import build_raw_window_dataset
from eeg_pipeline.torch_models import EEGWindowCNN


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


def plot_confusion_matrix(confusion: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(confusion, cmap="Oranges")
    fig.colorbar(image, ax=ax)
    class_labels = ["non-seizure", "seizure"]
    ax.set_xticks([0, 1], labels=class_labels)
    ax.set_yticks([0, 1], labels=class_labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("PyTorch CNN Baseline Confusion Matrix")
    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            ax.text(col, row, str(confusion[row, col]), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    figures_dir, tables_dir = ensure_results_dir()

    dataset = build_raw_window_dataset(
        data_dir=ROOT / "data" / "raw" / "chbmit",
        patient_id=PATIENT_ID,
        filenames=FILES,
        filter_low_hz=FILTER_LOW_HZ,
        filter_high_hz=FILTER_HIGH_HZ,
        window_seconds=WINDOW_SECONDS,
        overlap_fraction=OVERLAP_FRACTION,
    )

    metadata = dataset.metadata.copy()
    train_mask = metadata["file_name"].isin(TRAIN_FILES)
    test_mask = metadata["file_name"].isin(TEST_FILES)

    x_train = torch.tensor(dataset.windows[train_mask.to_numpy()], dtype=torch.float32)
    y_train = torch.tensor(metadata.loc[train_mask, "label"].to_numpy(), dtype=torch.float32)
    x_test = torch.tensor(dataset.windows[test_mask.to_numpy()], dtype=torch.float32)
    y_test = metadata.loc[test_mask, "label"].to_numpy()

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
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        predictions_binary = (probabilities >= 0.5).astype(int)

    precision = precision_score(y_test, predictions_binary, zero_division=0)
    recall = recall_score(y_test, predictions_binary, zero_division=0)
    f1 = f1_score(y_test, predictions_binary, zero_division=0)
    confusion = confusion_matrix(y_test, predictions_binary, labels=[0, 1])

    predictions = metadata.loc[
        test_mask,
        ["window_id", "file_name", "window_start_seconds", "window_end_seconds", "label"],
    ].copy()
    predictions["predicted_label"] = predictions_binary
    predictions["predicted_probability_seizure"] = probabilities
    predictions.to_csv(tables_dir / "pytorch_cnn_baseline_predictions.csv", index=False)

    plot_confusion_matrix(confusion, figures_dir / "pytorch_cnn_baseline_confusion_matrix.png")

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
    smoothed.to_csv(tables_dir / "pytorch_cnn_baseline_smoothed_predictions.csv", index=False)
    scored_predicted_events.to_csv(tables_dir / "pytorch_cnn_baseline_event_predictions.csv", index=False)
    scored_true_events.to_csv(tables_dir / "pytorch_cnn_baseline_true_event_evaluation.csv", index=False)

    metrics_lines = [
        "PyTorch CNN baseline",
        f"Device: {device}",
        f"Train files: {', '.join(TRAIN_FILES)}",
        f"Test files: {', '.join(TEST_FILES)}",
        f"Epochs: {EPOCHS}",
        f"Batch size: {BATCH_SIZE}",
        f"Learning rate: {LEARNING_RATE}",
        f"Precision: {precision:.4f}",
        f"Recall: {recall:.4f}",
        f"F1: {f1:.4f}",
        "Confusion matrix [[tn, fp], [fn, tp]]:",
        str(confusion.tolist()),
        f"Event detection rate: {event_metrics['detection_rate']:.4f}",
        f"False alarms per hour: {event_metrics['false_alarms_per_hour']:.4f}",
    ]
    (tables_dir / "pytorch_cnn_baseline_metrics.txt").write_text("\n".join(metrics_lines) + "\n")

    print("Saved outputs:")
    print("  FILE    results/tables/pytorch_cnn_baseline_predictions.csv")
    print("  FILE    results/tables/pytorch_cnn_baseline_smoothed_predictions.csv")
    print("  FILE    results/tables/pytorch_cnn_baseline_event_predictions.csv")
    print("  FILE    results/tables/pytorch_cnn_baseline_true_event_evaluation.csv")
    print("  FILE    results/tables/pytorch_cnn_baseline_metrics.txt")
    print("  FIGURE  results/figures/pytorch_cnn_baseline_confusion_matrix.png")
    print(f"  PREC    {precision:.4f}")
    print(f"  REC     {recall:.4f}")
    print(f"  F1      {f1:.4f}")
    print(f"  E-DRATE {event_metrics['detection_rate']:.4f}")
    print(f"  E-FA/H  {event_metrics['false_alarms_per_hour']:.4f}")


if __name__ == "__main__":
    main()
