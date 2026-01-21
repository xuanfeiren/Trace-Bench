# Test Score Addition Scripts

This document describes the two scripts created to add test scores to GEPA and OpenEvolve result files.

## Overview

Both scripts evaluate all Lean programs in the result files using the `evaluate()` function from `eval_utils.py`, which performs a 3-step evaluation:
1. **Compilation** (30%): Lean code must compile without errors
2. **Unit Tests** (30%): Combined code with gold reference unit tests must pass
3. **LLM Judge** (40%): Semantic equivalence scored by LLM judge

Final score = 0.3 × compile + 0.3 × unit_tests + 0.4 × llm_judge

## Scripts

### 1. `add_test_scores_to_gepa.py`

**Purpose**: Add `test_score` field to each iteration in GEPA iteration_history

**Scope**:
- Runs: gepa_1, gepa_2, gepa_3
- Tasks: 10-50 (41 tasks per run)
- Files: `results_llm_judge/gepa_${run_num}/task_${task_idx}_result.json`

**What it does**:
1. Reads each GEPA result file
2. Finds where iteration numbers restart (due to data merge error)
3. Only processes valid iterations before the restart
4. For each iteration:
   - Gets `best_lean_program` field
   - Checks cache (by code hash) to avoid re-evaluating same code
   - If not in cache, calls `evaluate(task_idx, lean_code)`
   - Adds `test_score` field to the iteration
5. Saves updated JSON file

**Output structure**:
```json
{
  "iteration_history": [
    {
      "iteration": 1,
      "score": 0.0,
      "best_score_so_far": 0.0,
      "best_lean_program": "...",
      "test_score": 0.0
    },
    ...
  ]
}
```

**Note**: The script handles the data merge error by detecting where iterations restart from 1 and only processing valid iterations.

### 2. `add_test_scores_to_openevolve.py`

**Purpose**: Add `test_score` field to each attempt in OpenEvolve history

**Scope**:
- Runs: openevolve_1, openevolve_2, openevolve_3
- Tasks: 10-50 (41 tasks per run)
- Files: `results_llm_judge/openevolve_${run_num}/openevolve_task_${task_id}_result.json`

**What it does**:
1. Reads each OpenEvolve result file
2. For each attempt in the history:
   - Gets `best_lean_program` field
   - Checks cache (by code hash) to avoid re-evaluating same code
   - If not in cache, calls `evaluate(task_id, lean_code)`
   - Adds `test_score` field to the attempt
3. Saves updated JSON file

**Output structure**:
```json
{
  "history": [
    {
      "attempt": 1,
      "score": 0.3,
      "best_score": 0.3,
      "iteration_found": 1,
      "best_lean_program": "...",
      "test_score": 0.3
    },
    ...
  ]
}
```

## Key Features

### Caching
Both scripts implement intelligent caching:
- **Key**: (task_id/task_idx, code_hash)
- **Value**: (score, feedback)
- Avoids re-evaluating identical code across iterations/attempts
- Cache is shared across all runs for maximum efficiency
- Significantly reduces evaluation time

### Error Handling
- Handles missing files gracefully
- Catches JSON parsing errors
- Catches evaluation errors and stores them as `None` with error message
- Continues processing even if individual evaluations fail

### Progress Reporting
- Shows which file is being processed
- Reports evaluation progress for each iteration/attempt
- Displays cache statistics every 5 tasks
- Final summary with:
  - Total files processed
  - Total evaluations performed
  - Cache hits
  - Cache efficiency percentage
  - Unique programs in cache

## Running the Scripts

Both scripts require the full environment with pantograph and dependencies:

```bash
# For GEPA
cd /Users/xuanfeiren/Documents/Trace-Bench/Veribench
uv run python add_test_scores_to_gepa.py

# For OpenEvolve  
cd /Users/xuanfeiren/Documents/Trace-Bench/Veribench
uv run python add_test_scores_to_openevolve.py
```

**Note**: These are long-running scripts. Each evaluation can take several seconds, and there are hundreds of programs to evaluate. The caching mechanism helps significantly reduce total time.

## Expected Output

### Console Output Example:
```
================================================================================
ADDING TEST SCORES TO GEPA ITERATION HISTORIES
================================================================================

================================================================================
Processing GEPA Run 1
================================================================================

📁 gepa_1/task_10_result.json
  📊 Processing 24 valid iterations (restart at index 24)
    Evaluating iteration 1... Score: 0.000
    Evaluating iteration 2... Score: 0.000
    ...
  ✅ Saved updated file
  💾 Cache size: 15 unique programs

...

================================================================================
FINAL SUMMARY
================================================================================
Total files processed: 123
Total evaluations performed: 450
Total cache hits: 350
Cache efficiency: 77.8%
Unique programs in cache: 450
================================================================================
```

## Data Validation

Both scripts perform data validation:
- Check if files exist
- Validate JSON structure
- Handle empty histories
- Detect iteration restart patterns (GEPA only)
- Evaluate ALL lean programs including placeholders (which should score 0)

## Important Notes

1. **All code is evaluated**: Even placeholder code like `"-- This is initial placeholder..."` gets evaluated (should return score 0)

2. **Cache key uses code content**: Two attempts with identical code will hit the cache, even across different runs

3. **GEPA data merge error**: The scripts detect and handle the iteration numbering error where iterations restart from 1 after reaching some number

4. **Non-destructive**: Original files are overwritten with the updated version, but the only change is adding the `test_score` field

5. **Network required**: The LLM judge step requires API calls, so network permission is needed

## Troubleshooting

If you encounter errors:

1. **ModuleNotFoundError**: Make sure to use `uv run python` to run with the proper environment
2. **Timeout errors**: The LLM judge or Lean interpreter may timeout on complex programs - these get score 0.0
3. **File not found**: Check that the result directories exist and contain the expected files
4. **JSON errors**: Check if any result files were corrupted during previous processing

## Next Steps

After running these scripts, the result files will have `test_score` added to each iteration/attempt, allowing for:
- Analysis of test score vs. internal score correlation
- Tracking actual code quality over iterations
- Comparing different optimization methods more accurately
- Understanding when internal scores mislead the optimization process
