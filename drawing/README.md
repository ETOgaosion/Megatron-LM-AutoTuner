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

## `comm_memory/plot_comm_memory.py`

Draw communication memory utilization across systems with dual y-axes:
- left y-axis: `total_memory` bar (computed as `runtime + offload`)
- right y-axis: `runtime_memory` bar
- both y-axes share the same scale and start value

Input data is read from `drawing/comm_memory/data.json` by default.

### Run

```bash
python drawing/comm_memory/plot_comm_memory.py
```

### Output

- Default output image path:
  - `outputs/drawing/comm_memory/<timestamp>/comm_memory.png`
- You can override it with `-o /path/to/output.png`

### Common Options

- `-i`, `--input`: input JSON path (default `drawing/comm_memory/data.json`)
- `-o`, `--output`: output image path
- `--dpi`: image dpi (default `180`)
- `--font-size`: base font size (default `12`)

### Example With Custom Paths

```bash
python drawing/comm_memory/plot_comm_memory.py \
  -i drawing/comm_memory/data.json \
  -o outputs/drawing/comm_memory/custom/comm_memory.png \
  --dpi 200 \
  --font-size 13
```

## `tp_overlap_effect/plot_tp_overlap_effect.py`

Draw TP overlap overhead from `drawing/tp_overlap_effect/data.json` with grouped bars:
- `original`: baseline latency
- `our`: optimized latency
- per-model reduction ratio annotation (e.g. `-3.2%`)

### Run

```bash
python drawing/tp_overlap_effect/plot_tp_overlap_effect.py
```

### Output

- Default output image path:
  - `outputs/drawing/tp_overlap_effect/<timestamp>/tp_overlap_effect.png`
- You can override it with `-o /path/to/output.png`

### Common Options

- `-i`, `--input`: input JSON path (default `drawing/tp_overlap_effect/data.json`)
- `-o`, `--output`: output image path
- `--title`: figure title (default `TP Overlap Overhead`)
- `--font-size`: base font size (default `14`, slightly large)
- `--dpi`: image dpi (default `180`)

### Example With Custom Output

```bash
python drawing/tp_overlap_effect/plot_tp_overlap_effect.py \
  -i drawing/tp_overlap_effect/data.json \
  -o outputs/drawing/tp_overlap_effect/custom/tp_overlap_effect.png \
  --font-size 16
```

## `cp_overlap_effect/plot_cp_overlap_effect.py`

Draw CP overlap overhead from `drawing/cp_overlap_effect/data.json` with grouped bars:
- `original`: baseline latency
- `our`: optimized latency
- per-model reduction ratio annotation (e.g. `-2.7%`)

### Run

```bash
python drawing/cp_overlap_effect/plot_cp_overlap_effect.py
```

### Output

- Default output image path:
  - `outputs/drawing/cp_overlap_effect/<timestamp>/cp_overlap_effect.png`
- You can override it with `-o /path/to/output.png`

### Common Options

- `-i`, `--input`: input JSON path (default `drawing/cp_overlap_effect/data.json`)
- `-o`, `--output`: output image path
- `--title`: figure title (default `CP Overlap Overhead`)
- `--font-size`: base font size (default `14`, slightly large)
- `--dpi`: image dpi (default `180`)

### Example With Custom Output

```bash
python drawing/cp_overlap_effect/plot_cp_overlap_effect.py \
  -i drawing/cp_overlap_effect/data.json \
  -o outputs/drawing/cp_overlap_effect/custom/cp_overlap_effect.png \
  --font-size 16
```
