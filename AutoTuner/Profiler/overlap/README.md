# Overlap Utilities

`AutoTuner/Profiler/overlap` now contains shared trace parsing and overlap
analysis utilities that can be reused by TP and CP overlap tests.

## Shared Modules

- `trace_analyzer.py`: torch profiler JSON parsing and event classification.
- `overlap_detector.py`: overlap interval aggregation and overlap metrics.

## TP Overlap Tuner

The TP-specific tuner, config generation, reports, CLI entry points, and TP-only
trace helpers live under `AutoTuner/Profiler/overlap/tp/`.

Start with:

- `AutoTuner/Profiler/overlap/tp/README.md`
- `python -m AutoTuner.Profiler.overlap.tp.main ...`

Python API:

```python
from AutoTuner.Profiler.overlap import OverlapDetector, TraceAnalyzer
from AutoTuner.Profiler.overlap.tp import TPOverlapTuner, TPOverlapTunerConfig
```
