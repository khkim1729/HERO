#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 6 ]; then
  echo ""
  echo "Usage:"
  echo "  bash scripts/run_end_to_end_totalseg_rf.sh CASE_ID CT_NII PET_NII GTVP_MASK GTVN_MASK OUTPUT_DIR"
  echo ""
  echo "Example:"
  echo "  bash scripts/run_end_to_end_totalseg_rf.sh TEST001 ct.nii.gz pet.nii.gz gtvp.nii.gz gtvn.nii.gz outputs/end_to_end/TEST001"
  echo ""
  exit 1
fi

CASE_ID="$1"
CT_NII="$2"
PET_NII="$3"
GTVP_MASK="$4"
GTVN_MASK="$5"
OUTPUT_DIR="$6"

REPO_ROOT="/home/introai17/tn-staging-nnunet"
PYTHON="/home/introai17/.conda/envs/talaria/bin/python"

TOTALSEG_DIR="${OUTPUT_DIR}/totalseg"
FEATURE_CSV="${OUTPUT_DIR}/${CASE_ID}_totalseg_rf_features.csv"
PRED_JSON="${OUTPUT_DIR}/${CASE_ID}_tn_staging_prediction.json"

T_MODEL="${REPO_ROOT}/results/tn_staging_rf_exclude48/totalseg_pred_oof_headneck_anatomy/T/T_stage_rf_model.joblib"
N_MODEL="${REPO_ROOT}/results/tn_staging_rf_exclude48/totalseg_pred_oof_headneck_anatomy/N/N_stage_rf_model.joblib"

mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "End-to-end TotalSegmentator RF TN staging"
echo "=========================================="
echo "CASE_ID:    ${CASE_ID}"
echo "CT_NII:     ${CT_NII}"
echo "PET_NII:    ${PET_NII}"
echo "GTVP_MASK:  ${GTVP_MASK}"
echo "GTVN_MASK:  ${GTVN_MASK}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "=========================================="

echo ""
echo "[1/3] Running TotalSegmentator..."
"${PYTHON}" "${REPO_ROOT}/scripts/run_totalseg_headneck.py" \
  --ct "${CT_NII}" \
  --out-dir "${TOTALSEG_DIR}" \
  --task total

echo ""
echo "[2/3] Extracting TotalSeg anatomy features..."
"${PYTHON}" "${REPO_ROOT}/scripts/extract_totalseg_headneck_features.py" \
  --case-id "${CASE_ID}" \
  --ct "${CT_NII}" \
  --pet "${PET_NII}" \
  --tumor-mask "${GTVP_MASK}" \
  --node-mask "${GTVN_MASK}" \
  --totalseg-dir "${TOTALSEG_DIR}" \
  --rf-model "${T_MODEL}" \
  --out-csv "${FEATURE_CSV}"

echo ""
echo "[3/3] Running RF TN staging inference..."
"${PYTHON}" "${REPO_ROOT}/scripts/infer_tn_staging_totalseg_rf.py" \
  --feature-csv "${FEATURE_CSV}" \
  --t-model "${T_MODEL}" \
  --n-model "${N_MODEL}" \
  --out-json "${PRED_JSON}"

echo ""
echo "Done."
echo "Prediction JSON:"
echo "${PRED_JSON}"
