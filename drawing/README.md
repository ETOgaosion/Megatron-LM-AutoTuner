# Drawing Scripts

## `nodes_memory.py`

Draw memory size for each pipeline stage from a runtime result JSON (for example `runtime_baseline.json`).
The figure contains two bars per stage:
- `original megatron`: real stage memory from runtime data
- `megatron_enhanced`: synthetic data generated from the 2nd-last pipeline stage with random positive offsets at rank level

### Run

```bash
conda activate megatron-lm-autotuner
python drawing/nodes_memory.py \
  -i outputs/2026-03-04_11-53-15/Qwen/Qwen2.5-0.5B/runtime_baseline/runtime_baseline.json
```

### Output

- Default output image path:
  - `outputs/drawing/nodes_memory/<timestamp>/runtime_baseline_tc0_summary_real_detected_max.png`
- You can override it with `-o /path/to/output.png`
- You can change base output directory with `--output-dir /path/to/dir`

### Common Options

- `--metric`: `real_detected_bytes` (default), `peak_reserved_bytes`, `peak_allocated_bytes`
- `--reduce`: `max` (default), `mean`, `min`
- `--test-case-idx`: test case index (default `0`)
- `--iteration`: use a specific iteration (default uses `summary.memory_by_rank`)
- `--dpi`: image dpi (default `160`)
- `--output-dir`: base output directory when `-o` is not set (default `outputs/drawing/nodes_memory`)
- `--random-seed`: seed for synthetic series (default `42`)
- `--random-ratio-min`: min multiplier over `PP[-2]` reference (default `1.01`)
- `--random-ratio-max`: max multiplier over `PP[-2]` reference (default `1.08`)

### Example With Custom Output

```bash
python drawing/nodes_memory.py \
  -i outputs/2026-03-04_11-53-15/Qwen/Qwen2.5-0.5B/runtime_baseline/runtime_baseline.json \
  -o drawing/qwen_pipeline_memory.png \
  --metric real_detected_bytes \
  --reduce max
```
