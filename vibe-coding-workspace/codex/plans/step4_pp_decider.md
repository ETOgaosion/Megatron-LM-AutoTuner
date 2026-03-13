# Step 4 PP Decider Plan

- Extend `AutoTuner/algorithm/dense_tuning.py` to include a step-4 PP decision result and wire it into the dense-tuning report.
- Reuse `AutoTuner/runtime/baseline` outputs instead of duplicating runtime accounting logic.
- Support both modes:
  - run a fresh PP=1 runtime-baseline probe for the chosen TP/CP/seqlen/recompute setup
  - consume an existing `runtime_baseline.json` for testability and offline reuse
- Estimate:
  - per-layer decoder parameter bytes
  - post-process overhead bytes
  - unit activation bytes from measured peak memory minus unit-model bytes
  - per-GPU usable memory budget
- Choose the minimal PP size whose stage distribution fits the regular-stage and post-process-stage memory capacities.
- Add targeted tests that exercise the step-4 decider from a saved runtime-baseline report.
