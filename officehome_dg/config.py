from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .constants import DEFAULT_MODEL


@dataclass
class TrainConfig:
    data_root: str = "../OfficeHomeDataset_10072016"
    manifest_dir: str = "artifacts/manifests"
    output_dir: str = "outputs/runs"
    weights_dir: str = "artifacts/weights"
    model_name: str = DEFAULT_MODEL
    num_classes: int = 65
    image_size: int = 224
    epochs: int = 30
    batch_size: int = 32
    eval_batch_size: int = 64
    num_workers: int = 4
    lr: float = 3e-5
    weight_decay: float = 0.05
    seed: int = 20260522
    mixup_alpha: float = 0.2
    coral_weight: float = 0.1
    pretrained: bool = True
    max_train_steps: int = 0
    max_eval_steps: int = 0
    amp: bool = True

    def resolve_paths(self, base_dir: Path) -> "TrainConfig":
        resolved = asdict(self)
        for key in ("data_root", "manifest_dir", "output_dir", "weights_dir"):
            path = Path(str(resolved[key])).expanduser()
            resolved[key] = str(path if path.is_absolute() else (base_dir / path).resolve())
        return TrainConfig(**resolved)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=True, indent=2, sort_keys=True)


def load_config(path: str | Path | None) -> tuple[TrainConfig, Path]:
    if path is None:
        return TrainConfig(), Path.cwd()
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    unknown = set(raw) - set(TrainConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return TrainConfig(**raw), config_path.parent


def override_config(config: TrainConfig, values: dict[str, Any]) -> TrainConfig:
    merged = asdict(config)
    for key, value in values.items():
        if value is not None:
            merged[key] = value
    return TrainConfig(**merged)
