#!/usr/bin/env bash
set -euo pipefail

show_help() {
  echo ""
  echo "Usage:"
  echo "  bash scripts/train_nnunet.sh DATASET CONFIGURATION FOLD NNUNET_RAW NNUNET_PREPROCESSED NNUNET_RESULTS"
  echo ""
  echo "Example:"
  echo "  bash scripts/train_nnunet.sh 123 3d_fullres 0 /data/nnUNet_raw /data/nnUNet_preprocessed /data/nnUNet_results"
  echo ""
  echo "Arguments:"
  echo "  DATASET              nnU-Net dataset ID or name, e.g. 123 or Dataset123_TNSegExclude48"
  echo "  CONFIGURATION        e.g. 2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres"
  echo "  FOLD                 e.g. 0, 1, 2, 3, 4, or all"
  echo "  NNUNET_RAW           path to nnUNet_raw"
  echo "  NNUNET_PREPROCESSED  path to nnUNet_preprocessed"
  echo "  NNUNET_RESULTS       path to nnUNet_results"
  echo ""
}

if [ "$#" -ne 6 ]; then
  show_help
  exit 1
fi

DATASET="$1"
CONFIGURATION="$2"
FOLD="$3"
NNUNET_RAW_PATH="$4"
NNUNET_PREPROCESSED_PATH="$5"
NNUNET_RESULTS_PATH="$6"

export nnUNet_raw="${NNUNET_RAW_PATH}"
export nnUNet_preprocessed="${NNUNET_PREPROCESSED_PATH}"
export nnUNet_results="${NNUNET_RESULTS_PATH}"

echo "=========================================="
echo "nnU-Net training"
echo "=========================================="
echo "DATASET:              ${DATASET}"
echo "CONFIGURATION:        ${CONFIGURATION}"
echo "FOLD:                 ${FOLD}"
echo "nnUNet_raw:           ${nnUNet_raw}"
echo "nnUNet_preprocessed:  ${nnUNet_preprocessed}"
echo "nnUNet_results:       ${nnUNet_results}"
echo "=========================================="

if ! command -v nnUNetv2_plan_and_preprocess >/dev/null 2>&1; then
  echo "ERROR: nnUNetv2_plan_and_preprocess was not found."
  echo "Please activate your nnU-Net environment first."
  exit 1
fi

if ! command -v nnUNetv2_train >/dev/null 2>&1; then
  echo "ERROR: nnUNetv2_train was not found."
  echo "Please activate your nnU-Net environment first."
  exit 1
fi

if [ ! -d "${nnUNet_raw}" ]; then
  echo "ERROR: nnUNet_raw directory does not exist: ${nnUNet_raw}"
  exit 1
fi

mkdir -p "${nnUNet_preprocessed}"
mkdir -p "${nnUNet_results}"

echo ""
echo "[1/2] Planning and preprocessing..."
nnUNetv2_plan_and_preprocess -d "${DATASET}" --verify_dataset_integrity

echo ""
echo "[2/2] Training..."
nnUNetv2_train "${DATASET}" "${CONFIGURATION}" "${FOLD}"

echo ""
echo "Done."
echo "Results are saved under:"
echo "${nnUNet_results}"
