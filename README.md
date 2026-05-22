# Office-Home 域泛化服务器训练包

## 训练目的

本包用于在 Office-Home 上训练跨视觉风格域泛化分类模型。协议固定为 Leave-One-Domain-Out：Art、Clipart、Product、Real World 每次留一个域做最终测试域，另外三个源域内部按类别切出 `10%` 验证集选择 checkpoint。主干是 `timm` 的 `vit_base_patch16_224.augreg_in21k`，首版实验比较 ERM、RandAugment、Mixup、CORAL 四种训练策略。服务器配置默认每个实验训练 `30` 个 epoch，配置文件在 `configs/vitb_officehome.yaml`。

## 如何部署环境

把代码包上传到 Linux 服务器并解压，进入解压后的工程根目录。代码包已经带 Office-Home 数据、LODO manifest 和 ViT 预训练权重缓存，训练脚本默认离线读取这些文件。

```bash
tar -xzf officehome_dg_bundle.tar.gz
cd officehome_dg_bundle
python -m pip install uv
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements/cuda.txt
```

确认当前环境能看到 CUDA 与 8 张 GPU：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

如果服务器需要指定与驱动匹配的 CUDA PyTorch wheel，先安装对应版本的 `torch` 与 `torchvision`，再执行 `uv pip install -r requirements/common.txt` 安装本项目公共依赖。

## 如何开始训练

完整实验入口如下。它会依次训练四个 held-out domain 与四种方法，单次实验由 `torchrun` 在单机 8 卡上启动；全部训练结束后会自动生成消融汇总图表并压缩结果包。

```bash
scripts/train_cuda_grid.sh
```

只跑一个实验时，命令格式是 `scripts/train_cuda_8x3090.sh <fold> <method>`。例如：

```bash
scripts/train_cuda_8x3090.sh Art erm
scripts/train_cuda_8x3090.sh "Real World" coral
```

`fold` 可选 `Art`、`Clipart`、`Product`、`Real World`，`method` 可选 `erm`、`randaugment`、`mixup`、`coral`。单个实验结束后已经有该 run 的 checkpoint、混淆矩阵和训练曲线；如果只跑了部分实验，再执行下面两条命令生成当前结果汇总与结果压缩包：

```bash
scripts/analyze_results.sh
scripts/make_results_bundle.sh
```

## 如何找到返回结果

每个 run 在 `outputs/runs/fold_<domain>/<method>/<run_name>/` 下保存结果。`checkpoints/best.pt` 是按源域验证集选择的模型，`checkpoints/last.pt` 用于续训，`test_metrics.json` 是 held-out test 指标，`metrics.jsonl` 是逐 epoch 日志。`analysis/` 下保存 `test_confusion_matrix.csv`、`test_confusion_matrix.png`、`test_per_class_metrics.csv` 和 `training_curves.png`。

全局图表在 `outputs/analysis/`：`data_split_counts.png` 是数据划分图，`ablation_by_fold.csv` 与 `ablation_mean.csv` 是消融表，`ablation_accuracy.png` 是四域四方法结果图。训练网格结束后，返回 `outputs/bundles/officehome_dg_results.tar.gz`；这个结果包保留 `best.pt`、混淆矩阵、逐类指标、训练曲线、测试指标和全局分析图，不包含每个 run 的 `last.pt`。
