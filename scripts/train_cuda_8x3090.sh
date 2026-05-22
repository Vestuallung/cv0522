#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

FOLD="${1:-Art}"
METHOD="${2:-erm}"
RUN_NAME="${RUN_NAME:-cuda_${FOLD// /_}_${METHOD}}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/OfficeHomeDataset_10072016}"
if [[ ! -d "$DATA_ROOT" ]]; then
  DATA_ROOT="$PROJECT_ROOT/../OfficeHomeDataset_10072016"
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

torchrun --standalone --nproc-per-node="${NPROC_PER_NODE:-8}" \
  -m officehome_dg.train \
  --config configs/vitb_officehome.yaml \
  --fold "$FOLD" \
  --method "$METHOD" \
  --device cuda \
  --data-root "$DATA_ROOT" \
  --run-name "$RUN_NAME"
