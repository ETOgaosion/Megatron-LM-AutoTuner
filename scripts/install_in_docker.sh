#!/bin/bash

cd /workspace/Megatron-LM-AutoTuner

pip install --no-deps -e .
pip install --no-deps -e verl

rm -rf TransformerEngine/build
rm -rf TransformerEngine/transformer_engine.egg-info
rm -rf TransformerEngine/transformer_engine.so
rm -rf TransformerEngine/tranformer_engine_pytorch.*
pip install nvidia-mathdx
MAX_JOBS=16 NVTE_FRAMEWORK=pytorch pip install --no-build-isolation -e TransformerEngine