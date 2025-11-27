#!/bin/bash

source .secrets/env.sh

MODEL_BASE_DIR=${BASE_DATA_DIR}/models

TO_DOWNLOAD=(
    "Qwen/Qwen2.5-0.5B"
    "Qwen/Qwen2.5-7B"
    "Qwen/Qwen2.5-72B"
    "Qwen/Qwen3-0.6B-Base"
    "Qwen/Qwen3-1.7B-Base"
    "Qwen/Qwen3-4B-Base"
    "Qwen/Qwen3-8B-Base"
    "Qwen/Qwen3-14B-Base"
    "Qwen/Qwen3-30B-A3B-Base"
    "Qwen/Qwen3-235B-A22B"
)

# Check if HFD_LOCATION is set and the file exists
if [[ -z "$HFD_LOCATION" ]] || [[ ! -f "$HFD_LOCATION" ]]; then
    echo "Error: HFD_LOCATION environment variable is not set or the file does not exist."
    echo "Please download hfd.sh from https://hf-mirror.com/hfd/hfd.sh"
    echo "Then set HFD_LOCATION to point to the downloaded script."
    exit 1
fi

for MODEL_NAME in "${TO_DOWNLOAD[@]}"; do
    DEST_DIR=${MODEL_BASE_DIR}/${MODEL_NAME}
    # mkdir -p ${DEST_DIR}
    echo "Downloading model config for ${MODEL_NAME} to ${DEST_DIR}"
    bash ${HFD_LOCATION} ${MODEL_NAME} --include config.json --local-dir ${DEST_DIR}
done