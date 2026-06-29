#!/usr/bin/env bash
set -euo pipefail

show_help() {
  echo ""
  echo "Usage:"
  echo "  bash scripts/predict_nnunet.sh DATASET CONFIGURATION FOLD NNUNET_RAW NNUNET_PREPROCESSED NNUNET_RESULTS INPUT_DIR OUTPUT_DIR"
  echo ""
  echo "Example:"
  echo "  bash scripts/predict_nnunet.sh 123 3d_fullres 0 /data/nnUNet_raw /data/nnUNet_preprocessed /data/nnUNet_results /data/imagesTs /data/predictions"
  echo ""
  echo "Arguments:"
  echo "  DATASET              nnU-Net dataset ID or name, e.g. 123 or Dataset123_TNSegExclude48"
  echo "  CONFIGURATION        e.g. 2d, 3d_fullres"
  echo "  FOLD                 e.g. 0, 1, 2, 3, 4, or all"
  echo "  NNUNET_RAW           path to nnUNet_raw"
  echo "  NNUNET_PREPROCESSED  path to nnUNet_preprocessed"
  echo "  NNUNET_RESULTS       path to nnUNet_results"
  echo "  INPUT_DIR            folder containing input images"
  echo "  OUTPUT_DIR           folder where predicted masks will be saved"
  echo ""
}

if [ "$#" -ne 8 ]; then
  show_help
  exit 1
fi

DATASET="$1"
CONFIGURATION="$2"
FOLD="$3"
NNUNET_RAW_PATH="$4"
NNUNET_PREPROCESSED_PATH="$5"
NNUNET_RESULTS_PATH="$6"
INPUT_DIR="$7"
OUTPUT_DIR="$8"

export nnUNet_raw="${NNUNET_RAW_PATH}"
export nnUNet_preprocessed="${NNUNET_PREPROCESSED_PATH}"
export nnUNet_results="${NNUNET_RESULTS_PATH}"

echo "=========================================="
echo "nnU-Net prediction"
echo "=========================================="
echo "DATASET:              ${DATASET}"
echo "CONFIGURATION:        ${CONFIGURATION}"
echo "FOLD:                 ${FOLD}"
echo "nnUNet_raw:           ${nnUNet_raw}"
echo "nnUNet_preprocessed:  ${nnUNet_preprocessed}"
echo "nnUNet_results:       ${nnUNet_results}"
echo "INPUT_DIR:            ${INPUT_DIR}"
echo "OUTPUT_DIR:           ${OUTPUT_DIR}"
echo "=========================================="

if ! command -v nnUNetv2_predict >/dev/null 2>&1; then
  echo "ERROR: nnUNetv2_predict was not found."
  echo "Please activate your nnU-Net environment first."
  exit 1
fi

if [ ! -d "${INPUT_DIR}" ]; then
  echo "ERROR: INPUT_DIR does not exist: ${INPUT_DIR}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

nnUNetv2_predict \
  -i "${INPUT_DIR}" \
  -o "${OUTPUT_DIR}" \
  -d "${DATASET}" \
  -c "${CONFIGURATION}" \
  -f "${FOLD}"

echo ""
echo "Done."
echo "Predictions are saved under:"
echo "${OUTPUT_DIR}"
