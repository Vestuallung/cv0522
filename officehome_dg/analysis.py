from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .constants import DOMAINS
from .data import ImageRecord


METHODS = ("erm", "randaugment", "mixup", "coral")


def class_names_from_records(records: Sequence[ImageRecord], num_classes: int) -> list[str]:
    names = [f"class_{index}" for index in range(num_classes)]
    for record in records:
        names[record.class_id] = record.class_name
    return names


def read_history(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_matrix_csv(matrix: np.ndarray, class_names: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", *class_names])
        for class_name, row in zip(class_names, matrix.tolist()):
            writer.writerow([class_name, *row])


def write_per_class_metrics(matrix: np.ndarray, class_names: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    true_counts = matrix.sum(axis=1)
    predicted_counts = matrix.sum(axis=0)
    correct = np.diag(matrix)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("class_id", "class_name", "support", "predicted", "correct", "recall", "precision"),
        )
        writer.writeheader()
        for class_id, class_name in enumerate(class_names):
            support = int(true_counts[class_id])
            predicted = int(predicted_counts[class_id])
            hit = int(correct[class_id])
            writer.writerow(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "support": support,
                    "predicted": predicted,
                    "correct": hit,
                    "recall": 0.0 if support == 0 else hit / support,
                    "precision": 0.0 if predicted == 0 else hit / predicted,
                }
            )


def plot_confusion_matrix(matrix: np.ndarray, class_names: Sequence[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_sums = np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    normalized = matrix / row_sums
    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(normalized, cmap="viridis", vmin=0.0, vmax=1.0)
    step = max(1, len(class_names) // 16)
    ticks = list(range(0, len(class_names), step))
    axis.set_xticks(ticks, [class_names[index] for index in ticks], rotation=75, ha="right", fontsize=7)
    axis.set_yticks(ticks, [class_names[index] for index in ticks], fontsize=7)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Normalized test confusion matrix")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_training_history(metrics_path: Path, path: Path) -> None:
    history = read_history(metrics_path)
    if not history:
        return
    epochs = [int(row["epoch"]) + 1 for row in history]
    train_loss = [float(row["train"]["loss"]) for row in history]
    val_loss = [float(row["val"]["loss"]) for row in history]
    train_acc = [float(row["train"]["acc"]) for row in history]
    val_acc = [float(row["val"]["acc"]) for row in history]
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, train_loss, marker="o", label="train")
    axes[0].plot(epochs, val_loss, marker="o", label="source val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(epochs, train_acc, marker="o", label="train")
    axes[1].plot(epochs, val_acc, marker="o", label="source val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Top-1 accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Accuracy")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_run_analysis(run_dir: Path, matrix: np.ndarray, class_names: Sequence[str]) -> None:
    analysis_dir = run_dir / "analysis"
    write_matrix_csv(matrix, class_names, analysis_dir / "test_confusion_matrix.csv")
    write_per_class_metrics(matrix, class_names, analysis_dir / "test_per_class_metrics.csv")
    plot_confusion_matrix(matrix, class_names, analysis_dir / "test_confusion_matrix.png")
    plot_training_history(run_dir / "metrics.jsonl", analysis_dir / "training_curves.png")


def plot_manifest_overview(manifest_dir: Path, output_dir: Path) -> None:
    summary_path = manifest_dir / "summary.json"
    if not summary_path.is_file():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = list(DOMAINS)
    split_colors = {"train": "#287D8E", "val": "#E2A93B", "test": "#B24745"}
    figure, axes = plt.subplots(1, len(folds), figsize=(15, 4), sharey=True)
    for axis, fold in zip(axes, folds):
        domains = list(DOMAINS)
        bottom = np.zeros(len(domains))
        for split in ("train", "val", "test"):
            values = np.array(
                [summary["folds"][fold].get(domain, {}).get(split, 0) for domain in domains],
                dtype=float,
            )
            axis.bar(domains, values, bottom=bottom, color=split_colors[split], label=split)
            bottom += values
        axis.set_title(f"Held out: {fold}")
        axis.tick_params(axis="x", labelrotation=55, labelsize=8)
    axes[0].set_ylabel("Images")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3)
    figure.suptitle("Office-Home LODO split counts", y=1.03)
    figure.tight_layout()
    figure.savefig(output_dir / "data_split_counts.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def collect_run_rows(runs_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(runs_root.glob("fold_*/*/*/test_metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        run_dir = metrics_path.parent
        rows.append(
            {
                "fold": payload["fold"],
                "method": payload["method"],
                "run_name": run_dir.name,
                "test_acc": float(payload["test"]["acc"]),
                "test_loss": float(payload["test"]["loss"]),
                "run_dir": str(run_dir),
            }
        )
    return rows


def write_rows_csv(rows: Iterable[dict[str, object]], path: Path, fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def best_rows_by_method_fold(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    chosen: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        key = (str(row["method"]), str(row["fold"]))
        if key not in chosen or float(row["test_acc"]) > float(chosen[key]["test_acc"]):
            chosen[key] = row
    return [chosen[key] for key in sorted(chosen)]


def summarize_methods(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(float(row["test_acc"]))
    return [
        {
            "method": method,
            "fold_count": len(grouped[method]),
            "mean_test_acc": float(np.mean(grouped[method])),
            "std_test_acc": float(np.std(grouped[method])),
        }
        for method in METHODS
        if method in grouped
    ]


def plot_ablation(rows: Sequence[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    value_map = {(str(row["method"]), str(row["fold"])): float(row["test_acc"]) for row in rows}
    available_methods = [method for method in METHODS if any((method, fold) in value_map for fold in DOMAINS)]
    x = np.arange(len(DOMAINS))
    width = 0.82 / max(len(available_methods), 1)
    figure, axis = plt.subplots(figsize=(10, 5))
    for index, method in enumerate(available_methods):
        values = [value_map.get((method, fold), np.nan) for fold in DOMAINS]
        offset = (index - (len(available_methods) - 1) / 2) * width
        axis.bar(x + offset, values, width=width, label=method)
    axis.set_xticks(x, DOMAINS)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Held-out test Top-1 accuracy")
    axis.set_title("Ablation across methods and held-out domains")
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def summarize_results(runs_root: Path, manifest_dir: Path, output_dir: Path) -> list[dict[str, object]]:
    rows = collect_run_rows(runs_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows_csv(rows, output_dir / "all_runs.csv", ("fold", "method", "run_name", "test_acc", "test_loss", "run_dir"))
    chosen = best_rows_by_method_fold(rows)
    write_rows_csv(
        chosen,
        output_dir / "ablation_by_fold.csv",
        ("fold", "method", "run_name", "test_acc", "test_loss", "run_dir"),
    )
    write_rows_csv(
        summarize_methods(chosen),
        output_dir / "ablation_mean.csv",
        ("method", "fold_count", "mean_test_acc", "std_test_acc"),
    )
    plot_ablation(chosen, output_dir / "ablation_accuracy.png")
    plot_manifest_overview(manifest_dir, output_dir)
    return chosen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Office-Home result runs and draw ablation figures.")
    parser.add_argument("--runs-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("artifacts/manifests"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = summarize_results(args.runs_root, args.manifest_dir, args.output_dir)
    print(json.dumps({"selected_runs": len(rows), "output_dir": str(args.output_dir)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
