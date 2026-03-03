# Requiremet: TP overlap test

Now you should refer to [AutoTuner/Profiler/overlap/trace_analyzer.py](../../../AutoTuner/Profiler/overlap/trace_analyzer.py) and create a single file tp_overlap_trace_analyzer.py

OBJECT: Since TP is hard to fully overlap, you should calculate overlap ratio of each config
INPUT: a trace file location and whether it is ColumnParallelLinear or RowParallelLinear
TASK: 
1. Analyze: in trace file like [outputs/2026-01-16_07-59-44/Qwen/Qwen3-0.6B/torch_profiler/jss-Rack-Server_37176.1768550485691720099.pt.trace.json](../../../outputs/2026-01-16_07-59-44/Qwen/Qwen3-0.6B/torch_profiler/jss-Rack-Server_37176.1768550485691720099.pt.trace.json), you should detect communication stream and computation stream within a random-pick `AutoTuner/testbench/ops_test/common.py(111): run_micro_batch` nvtx range, there are cutlass/gemm/relu in computation and Memory PtoP or rr/rs in communication stream
2. You should calculate all computation time and all communication time on timeline, and  calculate the `overlap ratio` = `overlapped communication time`/`all computation time`
3. generate output