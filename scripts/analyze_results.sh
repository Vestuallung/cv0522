#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python -m officehome_dg.analysis \
  --runs-root outputs/runs \
  --manifest-dir artifacts/manifests \
  --output-dir outputs/analysis
