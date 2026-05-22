from __future__ import annotations

import json

from officehome_dg.data import read_manifest
from officehome_dg.prepare_data import prepare_manifests

from .helpers import make_tiny_officehome


def test_prepare_manifests_keeps_target_domain_out_of_source_splits(tmp_path):
    data_root = tmp_path / "OfficeHome"
    make_tiny_officehome(data_root, classes=3, images_per_class=3)
    output_dir = tmp_path / "manifests"

    summary = prepare_manifests(data_root, output_dir, val_ratio=0.1, seed=7, expected_classes=3)

    art_records = read_manifest(output_dir / "Art.csv")
    assert summary["image_count"] == 36
    assert {record.split for record in art_records if record.domain == "Art"} == {"test"}
    assert not [record for record in art_records if record.domain == "Art" and record.split != "test"]
    assert {record.domain for record in art_records if record.split == "val"} == {
        "Clipart",
        "Product",
        "Real World",
    }
    assert any("Real World" in record.path for record in art_records)
    assert json.loads((output_dir / "class_to_id.json").read_text(encoding="utf-8")) == {
        "Class_0": 0,
        "Class_1": 1,
        "Class_2": 2,
    }
