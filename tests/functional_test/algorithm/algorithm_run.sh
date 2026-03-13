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

CONFIG_FILE="${CONFIG_FILE:-${SCRIPT_DIR}/algorithm_config_auto.json}"
if [[ $# -ge 1 ]]; then
    CONFIG_FILE="$1"
    shift
fi

"${PYTHON_BIN}" -m AutoTuner.algorithm.main --config "${CONFIG_FILE}" "$@"
