"""JSON I/O contract for Docker mixed-size flow (host <-> container)."""

from __future__ import annotations

INPUT_SCHEMA_V1 = "hrt_mixed_size_input_v1"
OUTPUT_SCHEMA_V1 = "hrt_mixed_size_output_v1"
# Optional input key ``flow`` (e.g. ``mixed_size_real``) selects container-side toolchain branch.

INPUT_JSON_NAME = "input.json"
OUTPUT_JSON_NAME = "output.json"
DOCKER_LOG_NAME = "docker.log"
