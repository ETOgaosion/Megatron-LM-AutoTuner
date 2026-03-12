# CP Overlap Plan

1. Mirror the existing TP overlap workflow structure with a CP-specific runner and trace analyzer that use `TEAttenWithCPEnhanced` and fixed `CP=2`.
2. Implement trace analysis anchored on `context_parallel_nvshmem.py(479): forward`, selecting compute as `flash::flash_fwd_kernel` streams and communication as `Memcpy PtoP (Device -> Device)` streams.
3. Add a small synthetic-trace test and update overlap docs so the new CP path is discoverable.
