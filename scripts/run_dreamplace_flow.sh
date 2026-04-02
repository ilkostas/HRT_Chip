#!/usr/bin/env bash
# Manual smoke: mount a host workdir that contains input.json; writes output.json + logs on host.
set -euo pipefail
WORKDIR="${1:?usage: run_dreamplace_flow.sh /path/to/workdir}"
IMAGE="${HRT_DREAMPLACE_IMAGE:-hrt-chip-dreamplace:local}"
EXTRA=()
if [[ -n "${HRT_TESTCASE_ROOT:-}" ]]; then
  EXTRA+=(-v "${HRT_TESTCASE_ROOT}:/testcase:ro")
fi
docker run --rm -v "$(realpath "$WORKDIR"):/work" "${EXTRA[@]}" "$IMAGE"
