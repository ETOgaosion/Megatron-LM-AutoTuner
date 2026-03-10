#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

if [ -f .secrets/env.sh ]; then
    source .secrets/env.sh
else
    echo "Warning: .secrets/env.sh not found. Continuing with current environment."
fi

CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/runtime_baseline_config.json}"
if [[ $# -ge 1 ]]; then
    CONFIG_FILE="$1"
    shift
fi

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6010}"
NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"

PY_ARGS=(--config "${CONFIG_FILE}")
if [[ -n "${MODEL_FILTER:-}" ]]; then
    PY_ARGS+=(--model-filter "${MODEL_FILTER}")
fi
PY_ARGS+=(
    --master-addr "${MASTER_ADDR}"
    --master-port "${MASTER_PORT}"
    --num-nodes "${NUM_NODES}"
    --node-rank "${NODE_RANK}"
)

"$PYTHON_BIN" "${SCRIPT_DIR}/runtime_baseline_run_from_config.py" "${PY_ARGS[@]}" "$@"
