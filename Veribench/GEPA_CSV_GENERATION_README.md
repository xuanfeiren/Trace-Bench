# GEPA CSV Generation with Test Scores

This document describes how the 12 CSV files for GEPA were generated using actual test scores from the evaluation pipeline.

## Overview

Generated **12 CSV files** (3 runs × 4 metrics) for GEPA using `test_score` from the iteration history:

- `result_prop_step_{1,2,3}.csv` - Score vs proposal step (iteration)
- `result_num_proposals_{1,2,3}.csv` - Score vs number of proposals (iteration)
- `result_eval_step_{1,2,3}.csv` - Score vs evaluation step (metric calls)
- `result_num_samples_{1,2,3}.csv` - Score vs number of samples (metric calls)

## Data Source

**Input**: `results_llm_judge/gepa_{1,2,3}/task_{10-50}_result.json`
- 3 runs (gepa_1, gepa_2, gepa_3)
- 41 tasks per run (task indices 10-50)
- Each task has `iteration_history` with `test_score` field

**Key Fields Used**:
- `test_score`: Actual evaluation score (0.3 × compile + 0.3 × unit_tests + 0.4 × llm_judge)
- `iteration`: Iteration number (1, 2, 3, ...)
- `num_metric_calls_so_far`: Cumulative evaluation count

## Metrics Explanation

### 1. prop_step / num_proposals
- **X-axis**: Iteration number (0 to 30)
- **Meaning**: For GEPA, 1 iteration = 1 proposal step = 1 program proposal
- **Calculation**:
  - For each iteration 0-30:
    - For each task: find highest `test_score` so far up to that iteration
    - Forward fill: if no data, use previous best
  - Average across all 41 tasks
- **Files**: 
  - `result_prop_step_{1,2,3}.csv`
  - `result_num_proposals_{1,2,3}.csv` (identical to prop_step)

### 2. eval_step / num_samples
- **X-axis**: Number of metric calls (0 to 50)
- **Meaning**: Total evaluation budget spent
- **Calculation**:
  - For each metric call count 0-50:
    - For each task: find highest `test_score` among iterations where `num_metric_calls_so_far <= count`
    - Forward fill: if no data at exact count, use previous best
  - Average across all 41 tasks
- **Files**: 
  - `result_eval_step_{1,2,3}.csv`
  - `result_num_samples_{1,2,3}.csv` (identical to eval_step)

## Special Handling

### Valid Iterations Only
GEPA result files have a data merge error where iterations restart:
```
[1, 2, 3, ..., 24, 1, 2, 3]  ← iterations restart from 1
```

**Solution**: 
- Detect where iterations restart (when current ≤ previous)
- Only use valid iterations before the restart point
- Ignore duplicated iterations at the end

### Forward Fill Logic
If a task has no data at a specific x-axis point:
- Use the last known best score
- Creates "plateau" effect when no improvement
- Ensures all tasks contribute to average

### Row 0 (Initial State)
All CSV files start with row 0:
```csv
x_axis,score
0,0.0
```
This represents the initial state before any optimization.

## Algorithm Details

### For prop_step / num_proposals:

```python
for iteration in range(0, 31):
    task_scores = []
    for task in range(10, 51):
        # Load valid iterations for this task
        valid_iters = load_task_data(task)
        
        # Find best test_score up to this iteration
        best_so_far = 0.0
        for iter_data in valid_iters:
            if iter_data['iteration'] <= iteration:
                if iter_data['test_score'] > best_so_far:
                    best_so_far = iter_data['test_score']
        
        task_scores.append(best_so_far)
    
    # Average across all tasks
    mean_score = average(task_scores)
    write_csv(iteration, mean_score)
```

### For eval_step / num_samples:

```python
for metric_calls in range(0, 51):
    task_scores = []
    for task in range(10, 51):
        # Load valid iterations for this task
        valid_iters = load_task_data(task)
        
        # Find best test_score up to this metric call count
        best_so_far = 0.0
        for iter_data in valid_iters:
            if iter_data['num_metric_calls_so_far'] <= metric_calls:
                if iter_data['test_score'] > best_so_far:
                    best_so_far = iter_data['test_score']
        
        task_scores.append(best_so_far)
    
    # Average across all tasks
    mean_score = average(task_scores)
    write_csv(metric_calls, mean_score)
```

## Output Format

All CSV files follow the same format:

```csv
x_axis_name,score
0,0.0
1,0.08243902439024389
2,0.2661788617886179
3,0.3491056910569106
...
```

Where:
- **Column 1**: X-axis value (iteration or metric calls)
- **Column 2**: Mean of highest-test-score-so-far across all 41 tasks
- **Rows**: One per x-axis value, starting from 0

## Usage

Generated CSV files are ready for plotting with multiple x-axes:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
prop_step_1 = pd.read_csv('data/llm_judge/gepa/result_prop_step_1.csv')
eval_step_1 = pd.read_csv('data/llm_judge/gepa/result_eval_step_1.csv')

# Plot with dual x-axis
fig, ax1 = plt.subplots()
ax2 = ax1.twiny()

ax1.plot(prop_step_1['prop_step'], prop_step_1['score'], label='Prop Steps')
ax2.plot(eval_step_1['eval_step'], eval_step_1['score'], label='Eval Steps', linestyle='--')

ax1.set_xlabel('Proposal Steps')
ax2.set_xlabel('Evaluation Steps')
ax1.set_ylabel('Mean Test Score')
plt.legend()
plt.show()
```

## Script

The generation script is: `generate_gepa_csvs.py`

To regenerate the CSVs:
```bash
cd /Users/xuanfeiren/Documents/Trace-Bench/Veribench
python generate_gepa_csvs.py
```

## Verification

Sample output from `result_prop_step_1.csv`:
- Row 0: score = 0.0 (initial state)
- Row 1: score ≈ 0.082 (after 1 iteration)
- Row 30: score ≈ 0.650 (after 30 iterations)

The scores show progressive improvement with forward-fill plateaus when no improvement occurs.

## Notes

1. **Test scores are real**: These use actual `test_score` values from the 3-step evaluation pipeline (compile + unit tests + LLM judge)

2. **Cross-run comparison**: The 3 runs (gepa_1, gepa_2, gepa_3) can be compared to see variance in optimization performance

3. **Multiple x-axes**: The same underlying data is presented with different x-axes for flexible analysis:
   - Iteration-based (prop_step, num_proposals)
   - Budget-based (eval_step, num_samples)

4. **Aggregation level**: All scores are averaged across 41 tasks to get a single curve per run

5. **Data quality**: Only valid iterations are used (before the restart point in the data merge error)
