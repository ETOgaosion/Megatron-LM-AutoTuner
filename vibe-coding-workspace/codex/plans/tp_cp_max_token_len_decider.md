# TP CP Max Token Len Decider Plan

1. Add a main-repo algorithm entry point that orchestrates TP overlap tuning and CP overlap profiling, or reads existing outputs, then normalizes their reports into one decision input.
2. Implement the config decider so it picks TP from the TP report and the smallest valid CP whose `CP * step1_max_token_len` covers the requested single-sample seqlen while respecting `CP < nprocs_per_node / TP`.
3. Add focused unit tests for report parsing and decision rules, then run the targeted test suite.
