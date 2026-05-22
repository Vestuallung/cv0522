from __future__ import annotations

import argparse
import json
from pathlib import Path

from .constants import DOMAINS, EXPECTED_CLASS_COUNT
from .data import manifest_summary, scan_images, split_fold, write_class_mapping, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Office-Home LODO manifests from the image directories.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/manifests"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--expected-classes", type=int, default=EXPECTED_CLASS_COUNT)
    return parser.parse_args()


def prepare_manifests(
    data_root: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
    expected_classes: int = EXPECTED_CLASS_COUNT,
) -> dict[str, object]:
    records = scan_images(data_root, expected_classes)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_class_mapping(records, output_dir / "class_to_id.json")
    summaries: dict[str, object] = {
        "data_root": str(data_root.expanduser().resolve()),
        "image_count": len(records),
        "val_ratio": val_ratio,
        "seed": seed,
        "folds": {},
    }
    for fold in DOMAINS:
        fold_records = split_fold(records, fold, val_ratio, seed)
        write_manifest(fold_records, output_dir / f"{fold.replace(' ', '_')}.csv")
        summaries["folds"][fold] = manifest_summary(fold_records)
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return summaries


def main() -> None:
    args = parse_args()
    summary = prepare_manifests(args.data_root, args.output_dir, args.val_ratio, args.seed, args.expected_classes)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
