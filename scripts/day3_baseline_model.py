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


def ensure_results_dir() -> tuple[Path, Path]:
    results_dir = ROOT / "results"
    figures_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, tables_dir


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
    image = ax.imshow(confusion, cmap="Blues")
    fig.colorbar(image, ax=ax)

    class_labels = ["non-seizure", "seizure"]
    ax.set_xticks([0, 1], labels=class_labels)
    ax.set_yticks([0, 1], labels=class_labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Day 3 Confusion Matrix")

    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            ax.text(col, row, str(confusion[row, col]), ha="center", va="center", color="black")

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

    per_channel_columns = sorted([column for column in data.columns if "__" in column])
    if not per_channel_columns:
        raise ValueError("No per-channel feature columns found. Re-run the Day 2 feature extraction step first.")

    train_files, test_files = validate_file_split(windows_metadata)
    train_mask = data["file_name"].isin(train_files)
    test_mask = data["file_name"].isin(test_files)

    x_train = data.loc[train_mask, per_channel_columns]
    y_train = data.loc[train_mask, "label"]
    x_test = data.loc[test_mask, per_channel_columns]
    y_test = data.loc[test_mask, "label"]

    if y_train.nunique() < 2:
        raise ValueError("Training split contains fewer than two classes")

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

    predictions = data.loc[
        test_mask,
        ["window_id", "file_name", "window_start_seconds", "window_end_seconds", "label"],
    ].copy()
    predictions["predicted_label"] = y_pred
    predictions["predicted_probability_seizure"] = y_score
    predictions.to_csv(tables_dir / "day3_predictions.csv", index=False)

    plot_confusion_matrix(confusion, figures_dir / "day3_confusion_matrix.png")

    metrics_lines = [
        "Day 3 baseline logistic regression",
        f"Train files: {', '.join(train_files)}",
        f"Test files: {', '.join(test_files)}",
        f"Training windows: {len(x_train)}",
        f"Test windows: {len(x_test)}",
        f"Per-channel feature count: {len(per_channel_columns)}",
        f"Precision: {precision:.4f}",
        f"Recall: {recall:.4f}",
        f"F1: {f1:.4f}",
        "Confusion matrix [[tn, fp], [fn, tp]]:",
        str(confusion.tolist()),
    ]

    (tables_dir / "day3_metrics.txt").write_text("\n".join(metrics_lines) + "\n")

    print("Saved outputs:")
    print("  FILE    results/tables/day3_metrics.txt")
    print("  FILE    results/tables/day3_predictions.csv")
    print("  FIGURE  results/figures/day3_confusion_matrix.png")
    print(f"  TRAIN   {', '.join(train_files)}")
    print(f"  TEST    {', '.join(test_files)}")
    print(f"  PREC    {precision:.4f}")
    print(f"  REC     {recall:.4f}")
    print(f"  F1      {f1:.4f}")


if __name__ == "__main__":
    main()
