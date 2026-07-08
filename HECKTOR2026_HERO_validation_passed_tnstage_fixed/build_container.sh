#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-hecktor2026-task}"
OUT_TAR="${2:-hecktor2026-task.tar.gz}"

cd "$(dirname "$0")"

BUILD_TS="$(date -u +%Y%m%dT%H%M%SZ)"

docker build --no-cache \
  --build-arg BUILD_TS="${BUILD_TS}" \
  -t "${IMAGE_NAME}" .

docker save "${IMAGE_NAME}" | gzip -c > "${OUT_TAR}"

gzip -t "${OUT_TAR}"
sha256sum "${OUT_TAR}"
ls -lh "${OUT_TAR}"
