from __future__ import annotations

from pathlib import Path

from PIL import Image

from officehome_dg.constants import DOMAINS


def make_tiny_officehome(root: Path, classes: int = 3, images_per_class: int = 3) -> list[str]:
    class_names = [f"Class_{index}" for index in range(classes)]
    for domain_index, domain in enumerate(DOMAINS):
        for class_index, class_name in enumerate(class_names):
            class_root = root / domain / class_name
            class_root.mkdir(parents=True, exist_ok=True)
            for image_index in range(images_per_class):
                color = (30 * domain_index, 20 * class_index, 10 * image_index)
                Image.new("RGB", (40, 40), color=color).save(class_root / f"{image_index:05d}.jpg")
    return class_names
