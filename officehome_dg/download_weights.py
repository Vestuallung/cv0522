from __future__ import annotations

import argparse
from pathlib import Path

from .constants import DEFAULT_MODEL
from .models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the timm ViT weights into a bundleable cache.")
    parser.add_argument("--weights-dir", type=Path, default=Path("artifacts/weights"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--num-classes", type=int, default=65)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    build_model(args.model_name, args.num_classes, pretrained=True, weights_dir=args.weights_dir)
    print(args.weights_dir.expanduser().resolve())


if __name__ == "__main__":
    main()
