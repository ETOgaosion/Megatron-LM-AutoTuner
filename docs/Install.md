# Install

## Clone Project

Make sure include all submodules:

```sh
git clone git@github.com:ETOgaosion/Megatron-LM-AutoTuner.git --recurse-submodules
```

Or you have already cloned:

```sh
git submodule update --init --recursive
```

## Environment

- CUDA 12.8: To use RTX-5090
- torch 2.8.0+cu128
- flash_attn 2.8.1 (local), 2.7.4 (Image)
    - directly use wheels
- Megatron: keep the latest, as a submodule (dev)
- TransformerEngine: keep the latest, as a submodule (dev)
- verl: keep the latest, as a submodule (dev)

## Image

Use `verlai/verl:app-verl0.6-transformers4.56.1-sglang0.5.2-mcore0.13.0-te2.2`, too slow

```bash
tmux new -s megatron-auto-tuner-0
bash scripts/docker_robust_pull.sh 
verlai/verl:app-verl0.6-transformers4.56.1-sglang0.5.2-mcore0.13.0-te2.2
```

You should use `Ctrl-C` to activate stop-then-continue download

## Local Development

- CUDA 12.8: Use [runfile installer](https://developer.nvidia.com/cuda-12-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=runfile_local)
- Download Miniconda
- `conda create -n megatron-lm-autotuner python==3.11`
- Download torch, torchvision, torchaudio 2.8.0
- Install apex
- Install flash attention wheel
    - check CXXABI: `ABI_FLAG=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")`
    - download: https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abi${ABI_FLAG}-cp311-cp311-linux_x86_64.whl
    - install
- Install TransformerEngine in current repo: `export NVTE_FRAMEWORK=pytorch && pip3 install --resume-retries 999 --no-deps --no-cache-dir --no-build-isolation -e .`
- Apply patch to TransformerEngine to enable flash_attention: `cd TransformerEngine && git apply ../patches/TransformerEngine.patch`
- Install Megatron in current repo
- Install verl in current repo

## Environment Migration

If you have an existing environment archive (e.g., megatron-lm-autotuner.tar.gz), you can quickly reuse it with the following steps:

- Copy the archive and adjust permissions

```sh
sudo cp /path/to/megatron-lm-autotuner.tar.gz ~/
sudo chown your_username:your_group ~/megatron-lm-autotuner.tar.gz
```

- Extract to the conda environment directory

```sh
tar -xzf ~/megatron-lm-autotuner.tar.gz -C ~/miniconda3/envs/megatron-lm-autotuner
```

- Activate the environment and fix path issues

```sh
conda activate megatron-lm-autotuner
conda-unpack
# Check for packages with incorrect paths
pip list --editable
# For each package with a path error, reinstall it using:
pip install -e /path/to/package
```
