#!/bin/bash

DIR_LIST=(
    AutoTuner
    tests
    tools
)

for dir in "${DIR_LIST[@]}"; do
    isort $dir
    black $dir
done
