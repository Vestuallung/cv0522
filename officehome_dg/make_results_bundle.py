from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


RESULT_PREFIX = Path("officehome_dg_results")
RUN_FILES = {
    "config.json",
    "metrics.jsonl",
    "test_metrics.json",
    "analysis/test_confusion_matrix.csv",
    "analysis/test_confusion_matrix.png",
    "analysis/test_per_class_metrics.csv",
    "analysis/training_curves.png",
    "checkpoints/best.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive final checkpoints and analysis figures from result runs.")
    parser.add_argument("--runs-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("outputs/analysis"))
    parser.add_argument("--archive", type=Path, default=Path("outputs/bundles/officehome_dg_results.tar.gz"))
    parser.add_argument("--include-last", action="store_true")
    return parser.parse_args()


def run_file_allowed(run_root: Path, path: Path, include_last: bool) -> bool:
    relative = path.relative_to(run_root).as_posix()
    return relative in RUN_FILES or (include_last and relative == "checkpoints/last.pt")


def archive_results(
    runs_root: Path,
    analysis_dir: Path,
    archive: Path,
    include_last: bool = False,
) -> Path:
    runs_root = runs_root.expanduser().resolve()
    analysis_dir = analysis_dir.expanduser().resolve()
    archive = archive.expanduser().resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)
    best_checkpoints = list(runs_root.glob("fold_*/*/*/checkpoints/best.pt"))
    confusion_figures = list(runs_root.glob("fold_*/*/*/analysis/test_confusion_matrix.png"))
    if not best_checkpoints or not confusion_figures:
        raise ValueError("Result runs must contain best checkpoints and confusion matrix figures")
    with tarfile.open(archive, "w:gz") as tar:
        for run_root in sorted({path.parents[1] for path in best_checkpoints}):
            for path in sorted(run_root.rglob("*")):
                if path.is_file() and run_file_allowed(run_root, path, include_last):
                    arcname = RESULT_PREFIX / "runs" / run_root.relative_to(runs_root) / path.relative_to(run_root)
                    tar.add(path, arcname=arcname)
        if analysis_dir.is_dir():
            tar.add(analysis_dir, arcname=RESULT_PREFIX / "analysis")
    validate_results_bundle(archive)
    return archive


def validate_results_bundle(archive: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()
    if not any(name.endswith("/checkpoints/best.pt") for name in names):
        raise ValueError("Result archive has no best checkpoint")
    if not any(name.endswith("/analysis/test_confusion_matrix.png") for name in names):
        raise ValueError("Result archive has no confusion matrix figure")
    if not any(name.endswith("/analysis/ablation_accuracy.png") for name in names):
        raise ValueError("Result archive has no ablation figure")


def main() -> None:
    args = parse_args()
    archive = archive_results(args.runs_root, args.analysis_dir, args.archive, args.include_last)
    print(archive)


if __name__ == "__main__":
    main()
