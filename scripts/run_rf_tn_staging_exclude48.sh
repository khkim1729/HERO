#!/usr/bin/env bash
set -euo pipefail

cd /home/introai17/tn-staging-nnunet

FEATURE_DIR="/home/introai17/salamanca/outputs/tn_staging/features"
OUT_DIR="/home/introai17/tn-staging-nnunet/results/tn_staging_rf_exclude48"

mkdir -p "${OUT_DIR}"

run_if_exists() {
  local csv_path="$1"
  local run_name="$2"

  if [ -f "${csv_path}" ]; then
    echo ""
    echo "=========================================="
    echo "Running RF: ${run_name}"
    echo "CSV: ${csv_path}"
    echo "=========================================="

    python scripts/train_rf_tn_staging_exclude48.py \
      --feature-csv "${csv_path}" \
      --exclude-file metadata/exclude_cases_48_tn_staging_only.txt \
      --output-dir "${OUT_DIR}" \
      --run-name "${run_name}" \
      --n-estimators 500 \
      --n-splits 5 \
      --random-state 42
  else
    echo "SKIP: file not found: ${csv_path}"
  fi
}

run_if_exists "${FEATURE_DIR}/tn_features_gt.csv" "gt_basic"
run_if_exists "${FEATURE_DIR}/tn_features_gt_with_components.csv" "gt_components"
run_if_exists "${FEATURE_DIR}/tn_features_gt_components_gland_anatomy.csv" "gt_components_gland"
run_if_exists "${FEATURE_DIR}/tn_features_gt_components_headneck_anatomy.csv" "gt_components_headneck"
run_if_exists "${FEATURE_DIR}/tn_features_gt_components_gland_headneck_anatomy.csv" "gt_components_gland_headneck"

run_if_exists "${FEATURE_DIR}/tn_features_pred_with_components.csv" "pred_components"
run_if_exists "${FEATURE_DIR}/tn_features_pred_oof_with_components.csv" "pred_oof_components"
run_if_exists "${FEATURE_DIR}/tn_features_pred_oof_components_headneck_anatomy.csv" "pred_oof_components_headneck"

echo ""
echo "All available RF runs complete."
echo "Results saved to: ${OUT_DIR}"
