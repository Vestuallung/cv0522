#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

for fold in Art Clipart Product "Real World"; do
  for method in erm randaugment mixup coral; do
    RUN_NAME="cuda_${fold// /_}_${method}" scripts/train_cuda_8x3090.sh "$fold" "$method"
  done
done

scripts/analyze_results.sh
scripts/make_results_bundle.sh
