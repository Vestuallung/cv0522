#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python -m officehome_dg.prepare_data \
  --data-root ../OfficeHomeDataset_10072016 \
  --output-dir artifacts/manifests \
  --val-ratio 0.1 \
  --seed 20260522
