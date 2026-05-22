from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .analysis import class_names_from_records, write_run_analysis
from .config import TrainConfig, load_config, override_config
from .constants import DOMAINS
from .distributed import cleanup, init_runtime
from .models import build_model
from .train import evaluate, evaluate_confusion_matrix, load_checkpoint, make_loaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved Office-Home checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/vitb_officehome.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fold", choices=DOMAINS, required=True)
    parser.add_argument("--method", choices=("erm", "randaugment", "mixup", "coral"), default="erm")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--write-run-analysis", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--data-root")
    parser.add_argument("--manifest-dir")
    parser.add_argument("--weights-dir")
    parser.add_argument("--model-name")
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-eval-steps", type=int)
    return parser.parse_args()


def load_eval_config(args: argparse.Namespace) -> TrainConfig:
    config, base_dir = load_config(args.config)
    names = (
        "data_root",
        "manifest_dir",
        "weights_dir",
        "model_name",
        "eval_batch_size",
        "num_workers",
        "max_eval_steps",
    )
    return override_config(config, {name: getattr(args, name) for name in names}).resolve_paths(base_dir)


def main() -> None:
    args = parse_args()
    config = load_eval_config(args)
    runtime = init_runtime(args.device)
    try:
        _train_loader, val_loader, test_loader = make_loaders(config, args.fold, args.method, runtime)
        model = build_model(config.model_name, config.num_classes, pretrained=False, weights_dir=config.weights_dir)
        model.to(runtime.device)
        load_checkpoint(args.checkpoint, model, map_location=runtime.device)
        selected_loader = val_loader if args.split == "val" else test_loader
        metrics = evaluate(model, selected_loader, runtime, config.max_eval_steps)
        confusion = None
        if args.write_run_analysis is not None:
            confusion = evaluate_confusion_matrix(
                model,
                selected_loader,
                runtime,
                config.num_classes,
                config.max_eval_steps,
            )
        if runtime.is_main:
            if confusion is not None:
                class_names = class_names_from_records(selected_loader.dataset.records, config.num_classes)
                write_run_analysis(args.write_run_analysis, confusion.detach().cpu().numpy(), class_names)
            print(json.dumps({"fold": args.fold, "method": args.method, "split": args.split, "metrics": metrics}))
    finally:
        cleanup(runtime)


if __name__ == "__main__":
    main()
