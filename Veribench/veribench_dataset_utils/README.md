# VeriBench Dataset Utils

This directory contains utilities and tools for working with the VeriBench dataset, a collection of 140 Lean 4 code generation tasks.

## Dataset

The `dataset/` directory contains 140 tasks in JSON format (`task_0.json` to `task_139.json`), maintaining the same sequence as the [VeriBench dataset on HuggingFace](https://huggingface.co/datasets/allenanie/veribench_with_prompts).

Each task file includes:
- Task description and requirements
- Gold reference Lean 4 code
- Unit tests
- Expected behavior

## Evaluation Utilities

### `eval_utils.py`
Core evaluation functions for testing LLM-generated Lean 4 code:

- **`compile(response)`**: Compiles Lean code and returns score (1.0 for success, 0.0 for failure) with feedback
- **`combine_code_with_tests(task_id, lean_code)`**: Combines generated implementation with unit tests from gold reference
- **`_extract_implementation_robust(lean_code)`**: Extracts implementation code, filtering out tests and examples

These utilities handle:
- Lean compilation via `lean_interpreter`
- Automatic removal of invalid import statements
- Detailed error reporting with context
- Timeout detection and handling

## Demo Scripts

The `demo/` directory provides example usage for evaluating Lean 4 code:

### 1. `compile.py` - Basic Compilation
Demonstrates how to compile Lean code and get feedback.

**Example Usage:**
```bash
cd /Users/xuanfeiren/Documents/Trace-Bench/Veribench
uv run python veribench_dataset_utils/demo/compile.py
```

**What it does:**
- Takes a complete Lean 4 implementation (CountingSort example for task 2)
- Compiles using `compile()` from `eval_utils`
- Reports score and feedback

### 2. `compile_with_unit_tests.py` - Unit Test Integration
Demonstrates the full evaluation workflow with unit test validation.

**Example Usage:**
```bash
cd /Users/xuanfeiren/Documents/Trace-Bench/Veribench
uv run python veribench_dataset_utils/demo/compile_with_unit_tests.py
```

**What it does:**
1. Compiles the implementation only (without tests)
2. Combines implementation with unit tests from gold reference (`task_2.json`)
3. Compiles the combined code
4. Reports final score:
   - `1.0`: All unit tests pass
   - `0.5`: Compiles but fails unit tests
   - `0.0`: Compilation errors

This mirrors the `VeribenchGuidewithUnitTests` workflow from `guide/guide.py`.

## Dataset Analysis

### Unit Test Coverage

**Coverage**: 123 out of 140 tasks (87.9%) have unit tests in their gold reference code.

**`task_lists.py`** - Pre-computed task categorization:
```python
from veribench_dataset_utils.task_lists import (
    TASKS_WITH_UNIT_TESTS,      # 123 tasks with unit tests
    TASKS_WITHOUT_UNIT_TESTS,   # 17 placeholder tasks
    TASKS_WITH_BOTH_EVAL_AND_EXAMPLE,  # 120 tasks
    has_unit_tests,             # Helper function
)

# Filter tasks for evaluation
for task_id in TASKS_WITH_UNIT_TESTS:
    evaluate_with_unit_tests(task_id)
```

**`check_unit_tests.py`** - Verification script:
```bash
python veribench_dataset_utils/check_unit_tests.py
```
Analyzes all 140 tasks and reports:
- Tasks with/without unit tests
- Test type distribution (#eval vs example)
- Detailed statistics

See **`unit_test_coverage_report.md`** for full analysis.

## Other Files

- **`create_datasets.py`**: Script for creating the dataset from VeriBench source
- **`unit_tests.py`**: Unit test extraction and processing utilities
- **`test_get_unit_tests.py`**: Tests for unit test extraction functionality

## Integration with Guide

The evaluation utilities in this directory are used by:
- `guide/guide.py` - Contains `VeribenchGuide` and `VeribenchGuidewithUnitTests` classes
- These guides are used for training and evaluating LLM agents on Lean 4 code generation

## Requirements

- Python 3.10+
- `uv` package manager
- Access to Lean 4 interpreter (via `my_processing_agents.lean_interpretor`)
- VeriBench bundle dependencies (for unit test integration)

## Quick Start

```python
from veribench_dataset_utils.eval_utils import compile, combine_code_with_tests

# Basic compilation
lean_code = "def myFunc := 42"
score, feedback = compile(lean_code)
print(f"Score: {score}, Feedback: {feedback}")

# With unit tests
task_id = 2
lean_code = "namespace CountingSort\ndef countingSort (arr : List Nat) : List Nat := ..."
combined = combine_code_with_tests(task_id, lean_code)
score, feedback = compile(combined)
print(f"Unit Test Score: {score}")
```