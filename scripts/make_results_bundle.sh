#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python -m officehome_dg.make_results_bundle \
  --runs-root outputs/runs \
  --analysis-dir outputs/analysis \
  --archive outputs/bundles/officehome_dg_results.tar.gz
