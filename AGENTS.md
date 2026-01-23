# AGENTS.md

## Purpose
This file is a concise, durable map of the Megatron-LM-AutoTuner repo so future readers do not need to scan the entire project.

## Project Summary
AutoTuner for Megatron + TransformerEngine, focused on MFU tuning for MCore training and post-training (e.g., RLHF/verl). The repo includes multiple upstream submodules and enhanced forks.

## High-Level Layout (Top-Level)
- `AutoTuner/`: core auto-tuner logic and profiling/testbench tooling.
- `Megatron-LM/`: upstream Megatron submodule.
- `Megatron-LM-Enhanced/`: enhanced Megatron submodule.
- `TransformerEngine/`: upstream TransformerEngine submodule.
- `TransformerEngine-Enhanced/`: enhanced TransformerEngine submodule.
- `verl/`: upstream verl submodule.
- `verl-enhanced/`: enhanced verl submodule.
- `docs/`: user-facing docs (note: `docs/README.md` is empty).
- `tests/functional_test/`: functional testbench scripts.
- `patches/`: local patch files (e.g., TransformerEngine patch).
- `scripts/`: helper scripts (docker setup, installs, etc.).
- `vibe-coding-workspace/`: contributor guidance and workflow notes (Codex/Claude).

## Primary Docs (Read These First)
- `README.md`: project overview and scope.
- `docs/Install.md`: install and environment setup.
- `docs/QuickStart.md`: functional tests and profiling workflows.
- `docs/design.md`: design notes.
- `docs/tools.md`: tooling notes.
- `docs/AutoTuner/testbench/README.md`: testbench specifics.

## Vibe-Coding Guidance (Authoritative Workflow Notes)
- `vibe-coding-workspace/codex/guidance/README.md`
- `vibe-coding-workspace/codex/guidance/Env.md`
- `vibe-coding-workspace/codex/guidance/QuickStart.md`
- `vibe-coding-workspace/codex/guidance/Test.md`

Key constraints from guidance:
- Do not modify a submodule and the main repo in the same task.
- Submodules require their own branches; main branches are protected.
- After submodule changes, test on remote machine and wait for review/merge before touching main repo.
- For main repo: branch from `main`, commit milestones, avoid stray files.

## Environment Notes
- CUDA 12.8, torch 2.8.0+cu128, FlashAttention 2.7.4 (image) / 2.8.x (local).
- Conda env: `megatron-lm-autotuner` (local), `megatron-lm-autotuner-enhanced` (remote).
- Docker image: `whatcanyousee/megatron-autotuner-env:mcore0.15.1_te2.7`.
- Remote machine: `5090-1` (see `.secrets` for access).

## Functional Testbench Scripts
- `tests/functional_test/testbench_collect_data.sh`
- `tests/functional_test/testbench_torch_profile.sh`
- `tests/functional_test/testbench_nsys_profile.sh` (docker only, root-needed)

## Typical Data Outputs
- `outputs/<timestamp>/<model>/collect_data/`
- `outputs/<timestamp>/<model>/torch_profiler/`
- `outputs/<timestamp>/<model>/nsys_profile/`

## Where to Start for AutoTuner Work
- Profiling and testbench: `AutoTuner/testbench/`
- Theoretical estimation base: `AutoTuner/testbench/ops_test/theoretical_base.py`
- Profile launcher: `AutoTuner/testbench/profile/launcher/`

## Known Gaps
- `docs/README.md` and `docs/AutoTuner/README.md` are empty.

## Change Log
- 2025-02-XX: Created this AGENTS.md summary for future quick reference.
