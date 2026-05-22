from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .analysis import class_names_from_records, write_run_analysis
from .config import TrainConfig, load_config, override_config
from .constants import DOMAINS
from .data import DomainBalancedBatchSampler, OfficeHomeDataset, build_transforms
from .distributed import Runtime, barrier, cleanup, init_runtime, reduce_sums, shared_text
from .methods import coral_loss, mixup_batch, soft_cross_entropy
from .models import build_model


def seed_everything(seed: int, rank: int = 0) -> None:
    seeded = seed + rank
    random.seed(seeded)
    np.random.seed(seeded)
    torch.manual_seed(seeded)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seeded)


def fold_manifest(manifest_dir: str | Path, fold: str) -> Path:
    manifest_path = Path(manifest_dir) / f"{fold.replace(' ', '_')}.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest for {fold}: {manifest_path}")
    return manifest_path


def make_loaders(
    config: TrainConfig,
    fold: str,
    method: str,
    runtime: Runtime,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    manifest = fold_manifest(config.manifest_dir, fold)
    train_dataset = OfficeHomeDataset(
        config.data_root,
        manifest,
        "train",
        build_transforms(config.image_size, method, training=True),
    )
    eval_transform = build_transforms(config.image_size, method, training=False)
    val_dataset = OfficeHomeDataset(config.data_root, manifest, "val", eval_transform)
    test_dataset = OfficeHomeDataset(config.data_root, manifest, "test", eval_transform)
    persistent = config.num_workers > 0
    if method == "coral":
        train_batch_sampler = DomainBalancedBatchSampler(
            train_dataset.records,
            batch_size=config.batch_size,
            seed=config.seed,
            num_replicas=runtime.world_size,
            rank=runtime.rank,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=config.num_workers,
            pin_memory=runtime.device.type == "cuda",
            persistent_workers=persistent,
        )
    else:
        train_sampler = (
            DistributedSampler(train_dataset, shuffle=True, seed=config.seed)
            if runtime.distributed
            else None
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=config.num_workers,
            pin_memory=runtime.device.type == "cuda",
            persistent_workers=persistent,
        )
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if runtime.distributed else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if runtime.distributed else None
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.num_workers,
        pin_memory=runtime.device.type == "cuda",
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=config.num_workers,
        pin_memory=runtime.device.type == "cuda",
        persistent_workers=persistent,
    )
    return train_loader, val_loader, test_loader


def set_loader_epoch(loader: DataLoader, epoch: int) -> None:
    sampler = loader.batch_sampler if isinstance(loader.batch_sampler, DomainBalancedBatchSampler) else loader.sampler
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    epoch: int,
    best_val_acc: float,
    config: TrainConfig,
    fold: str,
    method: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "config": asdict(config),
            "fold": fold,
            "method": method,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: AdamW | None = None,
    scheduler: CosineAnnealingLR | None = None,
    map_location: torch.device | str = "cpu",
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    unwrap_model(model).load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint.get("epoch", -1)) + 1, float(checkpoint.get("best_val_acc", 0.0))


def move_batch(batch, device: torch.device):
    return tuple(tensor.to(device, non_blocking=device.type == "cuda") for tensor in batch)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    config: TrainConfig,
    method: str,
    runtime: Runtime,
) -> dict[str, float]:
    model.train()
    totals = torch.zeros(3, device=runtime.device)
    use_amp = config.amp and runtime.device.type == "cuda"
    for step, batch in enumerate(loader):
        images, labels, domains = move_batch(batch, runtime.device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=runtime.device.type, dtype=torch.float16, enabled=use_amp):
            if method == "mixup":
                images, soft_targets = mixup_batch(images, labels, config.mixup_alpha, config.num_classes)
                logits = model(images)
                loss = soft_cross_entropy(logits, soft_targets)
            elif method == "coral":
                logits, features = model(images, return_features=True)
                loss = nn.functional.cross_entropy(logits, labels)
                loss = loss + config.coral_weight * coral_loss(features, domains)
            else:
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.shape[0]
        totals += torch.stack(
            [
                loss.detach().float() * batch_size,
                (logits.argmax(dim=1) == labels).sum().float(),
                torch.tensor(float(batch_size), device=runtime.device),
            ]
        )
        if config.max_train_steps and step + 1 >= config.max_train_steps:
            break
    totals = reduce_sums(totals, runtime)
    return {"loss": (totals[0] / totals[2]).item(), "acc": (totals[1] / totals[2]).item()}


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    runtime: Runtime,
    max_steps: int = 0,
) -> dict[str, float]:
    model.eval()
    totals = torch.zeros(3, device=runtime.device)
    for step, batch in enumerate(loader):
        images, labels, _domains = move_batch(batch, runtime.device)
        logits = model(images)
        loss = nn.functional.cross_entropy(logits, labels)
        batch_size = labels.shape[0]
        totals += torch.stack(
            [
                loss.detach().float() * batch_size,
                (logits.argmax(dim=1) == labels).sum().float(),
                torch.tensor(float(batch_size), device=runtime.device),
            ]
        )
        if max_steps and step + 1 >= max_steps:
            break
    totals = reduce_sums(totals, runtime)
    return {"loss": (totals[0] / totals[2]).item(), "acc": (totals[1] / totals[2]).item()}


@torch.inference_mode()
def evaluate_confusion_matrix(
    model: nn.Module,
    loader: DataLoader,
    runtime: Runtime,
    num_classes: int,
    max_steps: int = 0,
) -> torch.Tensor:
    model.eval()
    matrix = torch.zeros((num_classes, num_classes), device=runtime.device, dtype=torch.float32)
    for step, batch in enumerate(loader):
        images, labels, _domains = move_batch(batch, runtime.device)
        predicted = model(images).argmax(dim=1)
        flat_indices = (labels * num_classes + predicted).detach().cpu()
        counts = torch.bincount(flat_indices, minlength=num_classes * num_classes)
        matrix += counts.reshape(num_classes, num_classes).to(runtime.device, dtype=matrix.dtype)
        if max_steps and step + 1 >= max_steps:
            break
    return reduce_sums(matrix, runtime)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def append_metrics(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def run_training(
    config: TrainConfig,
    fold: str,
    method: str,
    device: str,
    run_name: str | None = None,
    resume: str | None = None,
) -> Path | None:
    if fold not in DOMAINS:
        raise ValueError(f"Unknown fold {fold}; choose one of {DOMAINS}")
    if method not in {"erm", "randaugment", "mixup", "coral"}:
        raise ValueError(f"Unknown method {method}")
    runtime = init_runtime(device)
    try:
        seed_everything(config.seed, runtime.rank)
        local_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        actual_run_name = shared_text(local_name, runtime)
        run_dir = Path(config.output_dir) / f"fold_{fold.replace(' ', '_')}" / method / actual_run_name
        if runtime.is_main:
            run_dir.mkdir(parents=True, exist_ok=True)
            write_json(run_dir / "config.json", asdict(config) | {"fold": fold, "method": method})
        train_loader, val_loader, test_loader = make_loaders(config, fold, method, runtime)
        model = build_model(config.model_name, config.num_classes, config.pretrained, config.weights_dir)
        model.to(runtime.device)
        if runtime.distributed:
            ddp_device_ids = [runtime.local_rank] if runtime.device.type == "cuda" else None
            model = DistributedDataParallel(model, device_ids=ddp_device_ids)
        optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))
        start_epoch = 0
        best_val_acc = -1.0
        if resume:
            start_epoch, best_val_acc = load_checkpoint(resume, model, optimizer, scheduler, runtime.device)
        for epoch in range(start_epoch, config.epochs):
            set_loader_epoch(train_loader, epoch)
            train_metrics = train_one_epoch(model, train_loader, optimizer, config, method, runtime)
            val_metrics = evaluate(model, val_loader, runtime, config.max_eval_steps)
            scheduler.step()
            row = {"epoch": epoch, "lr": scheduler.get_last_lr()[0], "train": train_metrics, "val": val_metrics}
            if runtime.is_main:
                append_metrics(run_dir / "metrics.jsonl", row)
                save_checkpoint(
                    run_dir / "checkpoints" / "last.pt",
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    max(best_val_acc, val_metrics["acc"]),
                    config,
                    fold,
                    method,
                )
                if val_metrics["acc"] > best_val_acc:
                    save_checkpoint(
                        run_dir / "checkpoints" / "best.pt",
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        val_metrics["acc"],
                        config,
                        fold,
                        method,
                    )
            best_val_acc = max(best_val_acc, val_metrics["acc"])
        barrier(runtime)
        best_checkpoint = run_dir / "checkpoints" / "best.pt"
        if best_checkpoint.is_file():
            load_checkpoint(best_checkpoint, model, map_location=runtime.device)
        test_metrics = evaluate(model, test_loader, runtime, config.max_eval_steps)
        confusion = evaluate_confusion_matrix(
            model,
            test_loader,
            runtime,
            config.num_classes,
            config.max_eval_steps,
        )
        if runtime.is_main:
            write_json(run_dir / "test_metrics.json", {"fold": fold, "method": method, "test": test_metrics})
            class_names = class_names_from_records(test_loader.dataset.records, config.num_classes)
            write_run_analysis(run_dir, confusion.detach().cpu().numpy(), class_names)
            return run_dir
        return None
    finally:
        cleanup(runtime)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Office-Home domain generalization models.")
    parser.add_argument("--config", type=Path, default=Path("configs/vitb_officehome.yaml"))
    parser.add_argument("--fold", choices=DOMAINS, required=True)
    parser.add_argument("--method", choices=("erm", "randaugment", "mixup", "coral"), default="erm")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--run-name")
    parser.add_argument("--resume")
    parser.add_argument("--data-root")
    parser.add_argument("--manifest-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--weights-dir")
    parser.add_argument("--model-name")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-train-steps", type=int)
    parser.add_argument("--max-eval-steps", type=int)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, base_dir = load_config(args.config)
    override_names = (
        "data_root",
        "manifest_dir",
        "output_dir",
        "weights_dir",
        "model_name",
        "epochs",
        "batch_size",
        "eval_batch_size",
        "num_workers",
        "lr",
        "seed",
        "max_train_steps",
        "max_eval_steps",
        "pretrained",
        "amp",
    )
    overrides = {name: getattr(args, name) for name in override_names}
    config = override_config(config, overrides).resolve_paths(base_dir)
    run_dir = run_training(config, args.fold, args.method, args.device, args.run_name, args.resume)
    if run_dir is not None:
        print(run_dir)


if __name__ == "__main__":
    main()
