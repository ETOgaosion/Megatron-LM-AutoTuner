#!/bin/bash

apt install -y libibmad5

cd /workspace/Megatron-LM-AutoTuner

pip install --no-deps -e .
pip install --no-deps -e verl
pip install --no-deps -e Megatron-LM

rm -rf TransformerEngine/build
rm -rf TransformerEngine/transformer_engine.egg-info
rm -rf TransformerEngine/transformer_engine.so
rm -rf TransformerEngine/tranformer_engine_pytorch.*
pip install nvidia-mathdx
MAX_JOBS=16 NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS=120 pip install --no-build-isolation -e TransformerEngine