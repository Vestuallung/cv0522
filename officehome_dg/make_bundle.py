from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


PROJECT_PREFIX = Path("officehome_dg_bundle")
EXCLUDED_PARTS = {
    ".pytest_cache",
    "__pycache__",
    ".DS_Store",
}
EXCLUDED_ROOTS = {
    Path("outputs") / "runs",
    Path("outputs") / "analysis",
    Path("outputs") / "bundles",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a server upload archive with code, data, manifests and weights.")
    parser.add_argument("--data-root", type=Path, default=Path("../OfficeHomeDataset_10072016"))
    parser.add_argument("--archive", type=Path, default=Path("outputs/bundles/officehome_dg_bundle.tar.gz"))
    return parser.parse_args()


def should_skip(project_root: Path, path: Path) -> bool:
    relative = path.relative_to(project_root)
    if any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in relative.parts):
        return True
    return any(relative == root or root in relative.parents for root in EXCLUDED_ROOTS)


def add_project(tar: tarfile.TarFile, project_root: Path) -> None:
    for path in sorted(project_root.rglob("*")):
        if should_skip(project_root, path):
            continue
        tar.add(path, arcname=PROJECT_PREFIX / path.relative_to(project_root), recursive=False)


def make_bundle(project_root: Path, data_root: Path, archive: Path) -> Path:
    data_root = data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Office-Home data root not found: {data_root}")
    archive = archive.expanduser().resolve()
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tar:
        add_project(tar, project_root)
        tar.add(data_root, arcname=PROJECT_PREFIX / data_root.name)
    validate_bundle(archive, data_root.name)
    return archive


def validate_bundle(archive: Path, data_dir_name: str) -> None:
    required = {
        str(PROJECT_PREFIX / "README.md"),
        str(PROJECT_PREFIX / "requirements" / "common.txt"),
        str(PROJECT_PREFIX / "scripts" / "train_cuda_8x3090.sh"),
        str(PROJECT_PREFIX / "scripts" / "analyze_results.sh"),
        str(PROJECT_PREFIX / "scripts" / "make_results_bundle.sh"),
        str(PROJECT_PREFIX / "artifacts" / "manifests" / "class_to_id.json"),
        str(PROJECT_PREFIX / data_dir_name),
    }
    with tarfile.open(archive, "r:gz") as tar:
        names = set(tar.getnames())
    missing = sorted(item for item in required if item not in names)
    if missing:
        raise ValueError(f"Bundle validation failed; missing: {missing}")
    weight_prefix = str(PROJECT_PREFIX / "artifacts" / "weights")
    if not any(name.startswith(weight_prefix + "/") for name in names):
        raise ValueError("Bundle validation failed; artifacts/weights has no cached files")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    archive = make_bundle(project_root, args.data_root, args.archive)
    print(archive)


if __name__ == "__main__":
    main()
