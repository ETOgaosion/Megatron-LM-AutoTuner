# Runtime Plan

1. Review GPTModel testbench operators and megatron pipeline helpers to decide minimal changes for PP/VPP/DP support.
2. Implement AutoTuner runtime module and CLI that build GPTModel-based pipeline and run forward/backward across test cases.
3. Add functional runtime test script and docs, including module-queue transparency check.
