from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

from .constants import DOMAIN_TO_ID, DOMAINS, EXPECTED_CLASS_COUNT, IMAGE_EXTENSIONS

MANIFEST_FIELDS = ("path", "domain", "class_name", "class_id", "fold", "split")


@dataclass(frozen=True)
class ImageRecord:
    path: str
    domain: str
    class_name: str
    class_id: int
    fold: str = ""
    split: str = ""

    def to_row(self) -> dict[str, str | int]:
        return {
            "path": self.path,
            "domain": self.domain,
            "class_name": self.class_name,
            "class_id": self.class_id,
            "fold": self.fold,
            "split": self.split,
        }


def list_classes(data_root: Path, expected_classes: int = EXPECTED_CLASS_COUNT) -> list[str]:
    classes_by_domain: dict[str, set[str]] = {}
    for domain in DOMAINS:
        domain_root = data_root / domain
        if not domain_root.is_dir():
            raise FileNotFoundError(f"Missing Office-Home domain directory: {domain_root}")
        classes_by_domain[domain] = {path.name for path in domain_root.iterdir() if path.is_dir()}
        if len(classes_by_domain[domain]) != expected_classes:
            raise ValueError(
                f"{domain} has {len(classes_by_domain[domain])} class directories, "
                f"expected {expected_classes}"
            )
    reference = classes_by_domain[DOMAINS[0]]
    for domain, class_names in classes_by_domain.items():
        if class_names != reference:
            raise ValueError(f"{domain} class names differ from {DOMAINS[0]}")
    return sorted(reference)


def scan_images(data_root: Path, expected_classes: int = EXPECTED_CLASS_COUNT) -> list[ImageRecord]:
    data_root = data_root.expanduser().resolve()
    class_names = list_classes(data_root, expected_classes)
    class_to_id = {name: index for index, name in enumerate(class_names)}
    records: list[ImageRecord] = []
    for domain in DOMAINS:
        for class_name in class_names:
            class_root = data_root / domain / class_name
            image_paths = sorted(
                path for path in class_root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            for image_path in image_paths:
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                records.append(
                    ImageRecord(
                        path=image_path.relative_to(data_root).as_posix(),
                        domain=domain,
                        class_name=class_name,
                        class_id=class_to_id[class_name],
                    )
                )
    if not records:
        raise ValueError(f"No images found under {data_root}")
    return records


def _val_size(sample_count: int, val_ratio: float) -> int:
    if sample_count <= 1:
        return 0
    return min(sample_count - 1, max(1, int(round(sample_count * val_ratio))))


def split_fold(
    records: Sequence[ImageRecord],
    held_out_domain: str,
    val_ratio: float,
    seed: int,
) -> list[ImageRecord]:
    if held_out_domain not in DOMAINS:
        raise ValueError(f"Unknown Office-Home domain: {held_out_domain}")
    grouped: dict[tuple[str, int], list[ImageRecord]] = defaultdict(list)
    fold_records: list[ImageRecord] = []
    for record in records:
        if record.domain == held_out_domain:
            fold_records.append(
                ImageRecord(**{**record.__dict__, "fold": held_out_domain, "split": "test"})
            )
        else:
            grouped[(record.domain, record.class_id)].append(record)
    for group_key in sorted(grouped):
        group = sorted(grouped[group_key], key=lambda item: item.path)
        chooser = random.Random(f"{seed}:{held_out_domain}:{group_key[0]}:{group_key[1]}")
        chooser.shuffle(group)
        validation = {record.path for record in group[: _val_size(len(group), val_ratio)]}
        for record in group:
            split = "val" if record.path in validation else "train"
            fold_records.append(ImageRecord(**{**record.__dict__, "fold": held_out_domain, "split": split}))
    validate_fold(fold_records, held_out_domain)
    return sorted(fold_records, key=lambda item: (item.split, item.domain, item.class_id, item.path))


def validate_fold(records: Sequence[ImageRecord], held_out_domain: str) -> None:
    targets = [record for record in records if record.domain == held_out_domain]
    if not targets or {record.split for record in targets} != {"test"}:
        raise ValueError(f"{held_out_domain} must only appear in the test split")
    leaked = [
        record.path
        for record in records
        if record.domain == held_out_domain and record.split in {"train", "val"}
    ]
    if leaked:
        raise ValueError(f"Target-domain leakage in {held_out_domain}: {leaked[:3]}")
    train_domains = {record.domain for record in records if record.split == "train"}
    val_domains = {record.domain for record in records if record.split == "val"}
    expected_sources = set(DOMAINS) - {held_out_domain}
    if train_domains != expected_sources or val_domains != expected_sources:
        raise ValueError(f"{held_out_domain} source splits do not cover all source domains")


def write_manifest(records: Sequence[ImageRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())


def read_manifest(path: str | Path, split: str | None = None) -> list[ImageRecord]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    records = [
        ImageRecord(
            path=row["path"],
            domain=row["domain"],
            class_name=row["class_name"],
            class_id=int(row["class_id"]),
            fold=row["fold"],
            split=row["split"],
        )
        for row in rows
    ]
    return [record for record in records if split is None or record.split == split]


def manifest_summary(records: Iterable[ImageRecord]) -> dict[str, dict[str, int]]:
    counts = Counter((record.domain, record.split) for record in records)
    return {
        domain: {split: counts[(domain, split)] for split in ("train", "val", "test") if counts[(domain, split)]}
        for domain in DOMAINS
    }


def write_class_mapping(records: Sequence[ImageRecord], path: Path) -> None:
    mapping = {record.class_name: record.class_id for record in records}
    ordered = dict(sorted(mapping.items(), key=lambda item: item[1]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ordered, ensure_ascii=True, indent=2), encoding="utf-8")


def build_transforms(image_size: int, method: str, training: bool) -> transforms.Compose:
    if training:
        operations: list[object] = [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
        if method == "randaugment":
            operations.append(transforms.RandAugment())
    else:
        resize_size = int(round(image_size / 0.875))
        operations = [transforms.Resize(resize_size), transforms.CenterCrop(image_size)]
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(operations)


class OfficeHomeDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, data_root: str | Path, manifest: str | Path, split: str, transform) -> None:
        self.data_root = Path(data_root)
        self.records = read_manifest(manifest, split)
        if not self.records:
            raise ValueError(f"Manifest {manifest} has no {split} records")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index]
        image_path = self.data_root / Path(record.path)
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        return (
            tensor,
            torch.tensor(record.class_id, dtype=torch.long),
            torch.tensor(DOMAIN_TO_ID[record.domain], dtype=torch.long),
        )


class DomainBalancedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        records: Sequence[ImageRecord],
        batch_size: int,
        seed: int,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        if batch_size < len({record.domain for record in records}):
            raise ValueError("CORAL batch size must cover every source domain")
        self.batch_size = batch_size
        self.seed = seed
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.groups: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            self.groups[record.domain].append(index)
        self.domains = sorted(self.groups)
        if len(self.domains) < 2:
            raise ValueError("CORAL requires at least two source domains")
        self.total_local = sum(math.ceil(len(indices) / num_replicas) for indices in self.groups.values())

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return math.ceil(self.total_local / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        local: dict[str, list[int]] = {}
        for domain in self.domains:
            indices = list(self.groups[domain])
            rng.shuffle(indices)
            shard = indices[self.rank :: self.num_replicas]
            local[domain] = shard or indices
        positions = {domain: 0 for domain in self.domains}
        base = self.batch_size // len(self.domains)
        remainder = self.batch_size % len(self.domains)
        for _ in range(len(self)):
            batch: list[int] = []
            for domain_index, domain in enumerate(self.domains):
                quota = base + int(domain_index < remainder)
                for _ in range(quota):
                    choices = local[domain]
                    batch.append(choices[positions[domain] % len(choices)])
                    positions[domain] += 1
            rng.shuffle(batch)
            yield batch
