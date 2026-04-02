#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="${HRT_DREAMPLACE_IMAGE_TAG:-hrt-chip-dreamplace:local}"
docker build -f "${ROOT}/docker/Dockerfile.dreamplace" -t "${TAG}" "${ROOT}/docker"
echo "Built ${TAG}"
