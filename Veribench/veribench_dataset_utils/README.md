# VeriBench Dataset Utils

This directory contains utilities and dataset for VeriBench - a collection of 140 Lean 4 code generation tasks.

## Dataset

The `dataset/` directory contains 140 tasks in JSON format (`task_0.json` to `task_139.json`), following the same sequence as [VeriBench on HuggingFace](https://huggingface.co/datasets/allenanie/veribench_with_prompts).

Each task includes:
- Task description
- Gold reference Lean 4 code
- Unit tests (when available)

**Note**: 17 tasks do not have unit tests in their gold reference code:
- Task 78
- Tasks 112-122 (11 tasks)
- Tasks 129-133 (5 tasks)

These are placeholder/stub tasks. For unit test-based evaluation, use the remaining **123 tasks**.

## Evaluation Utilities

### `eval_utils.py`

Core functions for evaluating LLM-generated Lean 4 code:

**`compile(response)`**
- Compiles Lean code using `lean_interpreter`
- Returns: `(score, feedback)` where score is 1.0 for success, 0.0 for failure
- Automatically handles invalid import statements

**`combine_code_with_tests(task_id, lean_code)`**
- Combines generated implementation with unit tests from gold reference
- Automatically matches function names and creates aliases
- Returns: Combined Lean code ready for compilation

**`_extract_implementation_robust(lean_code)`**
- Extracts implementation from generated code
- Filters out test sections and examples

## Demo Scripts

The `demo/` directory provides example usage:

### `compile.py` - Basic Compilation Demo
Demonstrates basic compilation of Lean 4 code (CountingSort example from task 2).

**Run:**
```bash
cd /Users/xuanfeiren/Documents/Trace-Bench/Veribench
uv run python veribench_dataset_utils/demo/compile.py
```

### `compile_with_unit_tests.py` - Unit Test Integration Demo
Demonstrates the full evaluation workflow:
1. Compile implementation only
2. Combine with unit tests from gold reference
3. Compile combined code
4. Report score (1.0 = pass all tests, 0.5 = compiles but fails tests, 0.0 = compilation error)

**Run:**
```bash
cd /Users/xuanfeiren/Documents/Trace-Bench/Veribench
uv run python veribench_dataset_utils/demo/compile_with_unit_tests.py
```

This workflow mirrors `VeribenchGuidewithUnitTests` from `guide/guide.py`.

## Usage Example

```python
from veribench_dataset_utils.eval_utils import compile, combine_code_with_tests

# Basic compilation
lean_code = "def myFunc := 42"
score, feedback = compile(lean_code)

# With unit tests (task 2 = CountingSort)
task_id = 2
lean_code = "namespace CountingSort\ndef countingSort (arr : List Nat) : List Nat := ..."
combined = combine_code_with_tests(task_id, lean_code)
score, feedback = compile(combined)
```

## Other Files

- `create_datasets.py` - Script for creating dataset from VeriBench source
- `unit_tests.py` - Unit test extraction utilities
- `test_get_unit_tests.py` - Tests for unit test extraction

## Requirements

- Python 3.10+
- `uv` package manager
- Lean 4 interpreter (via `my_processing_agents.lean_interpretor`)
- VeriBench bundle (for unit test integration)
