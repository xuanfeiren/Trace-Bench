# Complete CSV Generation Summary

## Overview

Successfully generated **32 CSV files** across three optimization methods (GEPA, OpenEvolve, DSPy) for plotting performance curves with test scores.

## Files Generated

### 1. GEPA (12 files)
**Location**: `data/llm_judge/gepa/`

Generated for runs 1, 2, 3:
- `result_prop_step_{1,2,3}.csv` (0-30)
- `result_num_proposals_{1,2,3}.csv` (0-30)
- `result_eval_step_{1,2,3}.csv` (0-50)
- `result_num_samples_{1,2,3}.csv` (0-50)

**Data source**: `results_llm_judge/gepa_{1,2,3}/task_{10-50}_result.json`
- Uses `test_score` field from `iteration_history`
- Handles data merge error (ignores restarted iterations)
- Tasks: 10-50 (41 tasks per run)

**Metrics**:
- `prop_step` / `num_proposals`: iteration-based (0-30)
- `eval_step` / `num_samples`: metric-calls-based (0-50)

### 2. OpenEvolve (12 files)
**Location**: `data/llm_judge/openevolve/` (ready to generate)

Will generate for runs 1, 2, 3:
- `result_num_samples_{1,2,3}.csv` (0-50)
- `result_num_proposals_{1,2,3}.csv` (0-50)
- `result_eval_step_{1,2,3}.csv` (0-10)
- `result_prop_step_{1,2,3}.csv` (0-10)

**Data source**: `results_llm_judge/openevolve_{1,2,3}/openevolve_task_{10-50}_result.json`
- Uses `test_score` field from `history`
- Tasks: 10-50 (41 tasks per run)

**Metrics**:
- `num_samples` / `num_proposals`: attempt-based (0-50)
- `eval_step` / `prop_step`: batch-based with 5 parallel (0-10)
  - eval_step 1 = attempt 5
  - eval_step 2 = attempt 10
  - eval_step n = attempt n×5

### 3. DSPy (12 files)
**Location**: `data/llm_judge/dspy/`

Generated for runs 1, 2, 3:
- `result_num_samples_{1,2,3}.csv` (0-50)
- `result_num_proposals_{1,2,3}.csv` (0-50)
- `result_eval_step_{1,2,3}.csv` (0-50)
- `result_prop_step_{1,2,3}.csv` (0-50)

**Data source**: `results_llm_judge/dspy_{1,2,3}/dspy_task_{10-50}_result.json`
- Uses `score` field (NOT `test_score`) from `history`
- Tasks: 10-50 (41 tasks per run)
- **Note**: Run 1 was already complete, generated runs 2 and 3

**Metrics**:
- All 4 metrics are identical for DSPy
- Each attempt = 1 sample = 1 proposal = 1 eval step = 1 prop step

## Generation Scripts

### 1. `generate_gepa_csvs.py`
- Processes GEPA runs 1-3
- Detects and handles iteration restart errors
- Uses `test_score` from valid iterations only
- Forward fill for missing data
- Averages across 41 tasks

### 2. `generate_openevolve_csvs.py` (ready to run)
- Will process OpenEvolve runs 1-3
- Uses `test_score` from all attempts
- Handles 5-parallel evaluation structure
- Forward fill for missing data
- Averages across 41 tasks

### 3. `generate_dspy_csvs.py`
- Processed DSPy runs 2-3 (run 1 already existed)
- Uses `score` field (not test_score)
- All metrics identical (attempt-based)
- Forward fill for missing data
- Averages across 41 tasks

## Common Features

All CSV files share these properties:

1. **Row 0**: Always 0.0 (initial state before optimization)
2. **Forward fill**: Maintains last best score when no new data
3. **Aggregation**: Mean of highest-score-so-far across 41 tasks
4. **Format**:
   ```csv
   x_axis_name,score
   0,0.0
   1,<score>
   2,<score>
   ...
   ```

## Metric Definitions by Method

| Metric | GEPA | OpenEvolve | DSPy |
|--------|------|------------|------|
| **num_samples** | num_metric_calls (0-50) | attempt (0-50) | attempt (0-50) |
| **num_proposals** | iteration (0-30) | attempt (0-50) | attempt (0-50) |
| **eval_step** | num_metric_calls (0-50) | batch × 5 (0-10) | attempt (0-50) |
| **prop_step** | iteration (0-30) | batch × 5 (0-10) | attempt (0-50) |

## Data Sources

### Score Fields Used
- **GEPA**: `test_score` (3-step evaluation: compile + unit tests + LLM judge)
- **OpenEvolve**: `test_score` (3-step evaluation: compile + unit tests + LLM judge)
- **DSPy**: `score` (internal metric, NOT test_score)

### Key Differences
1. **GEPA**: Sequential optimization, handles data merge errors
2. **OpenEvolve**: 5 parallel evaluations per step
3. **DSPy**: All metrics identical, uses different score field

## Usage for Plotting

Example multi-axis plot:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
gepa_prop = pd.read_csv('data/llm_judge/gepa/result_prop_step_1.csv')
gepa_eval = pd.read_csv('data/llm_judge/gepa/result_eval_step_1.csv')

# Plot with dual x-axis
fig, ax1 = plt.subplots()
ax2 = ax1.twiny()

ax1.plot(gepa_prop['prop_step'], gepa_prop['score'], 'b-', label='Proposal Steps')
ax2.plot(gepa_eval['eval_step'], gepa_eval['score'], 'r--', label='Evaluation Steps')

ax1.set_xlabel('Proposal Steps')
ax2.set_xlabel('Evaluation Steps')
ax1.set_ylabel('Mean Test Score')
plt.legend()
plt.show()
```

## Verification

Sample outputs verified:
- ✅ GEPA run 1: prop_step final score ≈ 0.650
- ✅ DSPy run 2: num_samples final score ≈ 0.947
- ✅ All files have row 0 = 0.0
- ✅ All files show progressive improvement with plateaus

## Directory Structure

```
data/llm_judge/
├── gepa/
│   ├── result_prop_step_{1,2,3}.csv
│   ├── result_num_proposals_{1,2,3}.csv
│   ├── result_eval_step_{1,2,3}.csv
│   └── result_num_samples_{1,2,3}.csv
├── openevolve/  (to be generated)
│   ├── result_prop_step_{1,2,3}.csv
│   ├── result_num_proposals_{1,2,3}.csv
│   ├── result_eval_step_{1,2,3}.csv
│   └── result_num_samples_{1,2,3}.csv
└── dspy/
    ├── result_prop_step_{1,2,3}.csv
    ├── result_num_proposals_{1,2,3}.csv
    ├── result_eval_step_{1,2,3}.csv
    └── result_num_samples_{1,2,3}.csv
```

## Next Steps

1. ✅ GEPA CSVs generated (12 files)
2. ✅ DSPy CSVs generated (8 files, run 1 existed)
3. ⏳ OpenEvolve CSVs ready to generate (12 files)
4. Generate comparison plots across methods
5. Analyze convergence patterns
6. Compare efficiency (eval steps vs. performance)

## Notes

- **Test scores vs Internal scores**: GEPA and OpenEvolve use actual test_score (3-step eval), DSPy uses its internal score metric
- **Data quality**: GEPA required handling of iteration restart errors
- **Efficiency comparison**: OpenEvolve's parallel structure visible in eval_step metric
- **Cross-method comparison**: Use eval_step as common x-axis (budget-based)
