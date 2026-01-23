# Runtime Functional Test

This directory provides a minimal runtime check for the baseline GPTModel runtime.

## Usage

1. Create `tests/functional_test/runtime/test_env.sh` from `test_env_sample.sh`.
2. Run:

```bash
bash tests/functional_test/runtime/runtime_run.sh
```

The script uses `torchrun` to execute `AutoTuner.runtime.main`.
