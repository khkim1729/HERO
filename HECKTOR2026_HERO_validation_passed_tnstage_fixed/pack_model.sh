#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-model}"
OUT_TAR="${2:-model.tar.gz}"

if [ ! -d "${MODEL_DIR}" ]; then
  echo "ERROR: model directory not found: ${MODEL_DIR}"
  exit 1
fi

chmod -R a+rX "${MODEL_DIR}"

tar -czf "${OUT_TAR}" -C "${MODEL_DIR}" .

gzip -t "${OUT_TAR}"
sha256sum "${OUT_TAR}"
ls -lh "${OUT_TAR}"

echo "Checking RFS files:"
tar -tzf "${OUT_TAR}" | grep -E "rfs/coxph_model.pkl|rfs/scaler.pkl|rfs/rfs_model_config.json" || true

echo "Checking nnU-Net folds:"
tar -tzf "${OUT_TAR}" | grep -E "fold_[0-4]/checkpoint_final\.pth$" | wc -l
