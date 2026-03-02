#!/bin/bash

set -e

cd flash-attention && \
    MAX_JOBS=16 FLASH_ATTN_CUDA_ARCHS="120" pip wheel --no-build-isolation .