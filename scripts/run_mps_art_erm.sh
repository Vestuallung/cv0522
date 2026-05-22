#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

python -m officehome_dg.train \
  --config configs/vitb_officehome.yaml \
  --fold Art \
  --method erm \
  --device mps \
  --epochs 1 \
  --batch-size "${BATCH_SIZE:-8}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-16}" \
  --num-workers "${NUM_WORKERS:-2}" \
  --run-name mps_art_erm_epoch1
