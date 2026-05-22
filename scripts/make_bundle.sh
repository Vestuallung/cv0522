#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python -m officehome_dg.make_bundle \
  --data-root ../OfficeHomeDataset_10072016 \
  --archive outputs/bundles/officehome_dg_bundle.tar.gz
