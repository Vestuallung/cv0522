from __future__ import annotations

from dataclasses import replace

from officehome_dg.analysis import summarize_results
from officehome_dg.config import TrainConfig
from officehome_dg.make_results_bundle import archive_results
from officehome_dg.prepare_data import prepare_manifests
from officehome_dg.train import run_training

from .helpers import make_tiny_officehome


def test_cpu_smoke_train_and_resume(tmp_path):
    data_root = tmp_path / "OfficeHome"
    make_tiny_officehome(data_root, classes=3, images_per_class=3)
    manifest_dir = tmp_path / "manifests"
    prepare_manifests(data_root, manifest_dir, val_ratio=0.1, seed=11, expected_classes=3)
    config = TrainConfig(
        data_root=str(data_root),
        manifest_dir=str(manifest_dir),
        output_dir=str(tmp_path / "runs"),
        weights_dir=str(tmp_path / "weights"),
        model_name="tiny_cnn",
        num_classes=3,
        image_size=32,
        epochs=1,
        batch_size=4,
        eval_batch_size=4,
        num_workers=0,
        pretrained=False,
        amp=False,
    )

    run_dir = run_training(config, fold="Art", method="erm", device="cpu", run_name="smoke")

    assert run_dir is not None
    assert (run_dir / "checkpoints" / "best.pt").is_file()
    assert (run_dir / "test_metrics.json").is_file()
    assert (run_dir / "analysis" / "test_confusion_matrix.csv").is_file()
    assert (run_dir / "analysis" / "test_confusion_matrix.png").is_file()
    assert (run_dir / "analysis" / "training_curves.png").is_file()
    resumed = run_training(
        replace(config, epochs=2),
        fold="Art",
        method="erm",
        device="cpu",
        run_name="resume",
        resume=str(run_dir / "checkpoints" / "last.pt"),
    )
    assert resumed is not None
    assert (resumed / "checkpoints" / "last.pt").is_file()
    analysis_dir = tmp_path / "analysis"
    rows = summarize_results(tmp_path / "runs", manifest_dir, analysis_dir)
    assert rows
    assert (analysis_dir / "ablation_accuracy.png").is_file()
    assert (analysis_dir / "data_split_counts.png").is_file()
    archive = archive_results(tmp_path / "runs", analysis_dir, tmp_path / "results.tar.gz")
    assert archive.is_file()
