#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: bash scripts/copy_nnunet_artifacts.sh SOURCE_NNUNET_RESULTS DATASET_NAME"
  exit 1
fi

SOURCE_NNUNET_RESULTS="$1"
DATASET_NAME="$2"
SOURCE_DATASET_DIR="${SOURCE_NNUNET_RESULTS}/${DATASET_NAME}"

if [ ! -d "${SOURCE_DATASET_DIR}" ]; then
  echo "ERROR: Dataset result folder not found: ${SOURCE_DATASET_DIR}"
  exit 1
fi

DEST_RESULTS_DIR="results/nnunet/${DATASET_NAME}"
DEST_CKPT_DIR="checkpoints/nnunet/${DATASET_NAME}"

mkdir -p "${DEST_RESULTS_DIR}"
mkdir -p "${DEST_CKPT_DIR}"

find "${SOURCE_DATASET_DIR}" -name "summary.json" -print | while read -r f; do
  rel="${f#${SOURCE_DATASET_DIR}/}"
  dest="${DEST_RESULTS_DIR}/${rel}"
  mkdir -p "$(dirname "${dest}")"
  cp "${f}" "${dest}"
  echo "Copied summary: ${rel}"
done

find "${SOURCE_DATASET_DIR}" -name "progress.png" -print | while read -r f; do
  rel="${f#${SOURCE_DATASET_DIR}/}"
  dest="${DEST_RESULTS_DIR}/${rel}"
  mkdir -p "$(dirname "${dest}")"
  cp "${f}" "${dest}"
  echo "Copied progress: ${rel}"
done

find "${SOURCE_DATASET_DIR}" -name "dataset.json" -print | while read -r f; do
  rel="${f#${SOURCE_DATASET_DIR}/}"
  dest="${DEST_RESULTS_DIR}/${rel}"
  mkdir -p "$(dirname "${dest}")"
  cp "${f}" "${dest}"
  echo "Copied dataset.json: ${rel}"
done

find "${SOURCE_DATASET_DIR}" -name "plans.json" -print | while read -r f; do
  rel="${f#${SOURCE_DATASET_DIR}/}"
  dest="${DEST_RESULTS_DIR}/${rel}"
  mkdir -p "$(dirname "${dest}")"
  cp "${f}" "${dest}"
  echo "Copied plans.json: ${rel}"
done

find "${SOURCE_DATASET_DIR}" -name "checkpoint_best.pth" -print | while read -r f; do
  rel="${f#${SOURCE_DATASET_DIR}/}"
  safe_rel="$(echo "${rel}" | tr '/' '_')"
  dest="${DEST_CKPT_DIR}/${safe_rel}"
  cp "${f}" "${dest}"
  echo "Copied checkpoint_best: ${rel}"
done

echo "Done."
echo "Results copied to: ${DEST_RESULTS_DIR}"
echo "Checkpoints copied to: ${DEST_CKPT_DIR}"
