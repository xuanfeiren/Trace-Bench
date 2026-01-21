# KernelBench DSPy Sequential Optimization

This directory contains optimization agents for KernelBench CUDA kernel tasks.

## Files

- **kernel_PS.py**: Priority Search optimization using Trace framework
- **kernel_PS_modal.py**: Priority Search with Modal GPU evaluation
- **kernel_dspy.py**: DSPy sequential refinement optimization (NEW)
- **evaluate_with_modal.py**: Modal-based evaluation utilities
- **secrets_local.py**: Local API keys and configuration

## kernel_dspy.py - DSPy Sequential Refinement

This script uses DSPy to iteratively improve CUDA kernel implementations based on performance feedback.

### How it works

1. **Initial Generation**: DSPy agent generates an initial CUDA kernel from the task description
2. **Evaluation**: The kernel is evaluated using KernelBench's evaluation function
3. **Refinement**: Based on performance feedback, DSPy generates an improved version
4. **Iteration**: Steps 2-3 repeat until score >= 1.0 or max attempts reached

### Key Components

- **DSPyCUDAAgent**: DSPy module with two signatures:
  - `CUDAKernelGenerator`: Generates initial CUDA code from task description
  - `CUDAKernelRefiner`: Refines CUDA code based on performance feedback
  - Note: No hardcoded instructions - the task description from `task['inputs'][0]` is used directly

- **sequential_optimization()**: Main optimization loop that:
  - Uses `task['inputs'][0]` as the task description (no additional instructions)
  - Uses `evaluate(ref_arch_src=info, custom_cuda=response, ...)` for evaluation
  - Returns score and feedback from KernelBench evaluation

### Usage

```bash
# Basic usage - optimize task 0 with default settings
python kernel_dspy.py --task_idx 0

# With verbose output and save results
python kernel_dspy.py --task_idx 0 --verbose --save_results

# Use different model
python kernel_dspy.py --task_idx 0 --model gpt-4o --max_attempts 30

# Full example with all parameters
python kernel_dspy.py \
  --task_idx 0 \
  --model claude-3.5-sonnet \
  --max_attempts 50 \
  --num_correct_trials 1 \
  --num_perf_trials 5 \
  --verbose \
  --save_results
```

### Command-line Arguments

- `--task_idx`: Task index from KernelBench dataset (default: 0)
- `--model`: LLM model name (default: claude-3.5-sonnet)
  - Supported: `claude-3.5-sonnet`, `gpt-4o`, `gemini-2.0-flash`, etc.
- `--max_attempts`: Maximum refinement attempts (default: 50)
- `--num_correct_trials`: Number of correctness trials (default: 1)
- `--num_perf_trials`: Number of performance trials (default: 5)
- `--verbose`: Print detailed logs during optimization
- `--save_results`: Save results to JSON file in `results/kernel_dspy/`

### Evaluation Metrics

- **Score < 1.0**: Kernel is slower than PyTorch reference (or incorrect)
- **Score = 1.0**: Kernel matches PyTorch reference performance
- **Score > 1.0**: Kernel is faster than PyTorch reference (SUCCESS!)

### Example Output

```
Configuring DSPy with anthropic/claude-3.5-sonnet...

Loading task 0 from KernelBench dataset...
Task loaded successfully. Task ID: 0

Starting DSPy sequential optimization:
  Model: anthropic/claude-3.5-sonnet
  Max attempts: 50
  Target: Achieve score >= 1.0 (correct and faster than PyTorch)

======================================================================
Attempt 1/50
======================================================================
Score: 0.8234
Feedback: Kernel is correct but slower than reference...

======================================================================
Attempt 2/50
======================================================================
Score: 0.9512
★ New best score: 0.9512
...

======================================================================
✓ SUCCESS at attempt 8!
  Score: 1.0234 (>= 1.0 means faster than PyTorch reference)
======================================================================
```

### Comparison with kernel_PS.py

| Feature | kernel_dspy.py | kernel_PS.py |
|---------|---------------|--------------|
| Framework | DSPy | Trace/Opto |
| Algorithm | Sequential refinement | Priority Search |
| Approach | Generate → Evaluate → Refine | Explore multiple candidates |
| Feedback | Direct text feedback | Numeric scores + feedback |
| Best for | Simple iterative improvement | Complex optimization landscape |

## Prerequisites

Make sure you have:
1. CUDA evaluation server running (or use Modal)
2. API keys configured in `secrets_local.py` or environment variables
3. Required packages: `dspy`, `litellm`, `datasets`

## Notes

- All scripts use the `create_single_task_dataset()` function to load tasks
- Evaluation uses the `evaluate()` function from `guide/evaluate.py`
- Task description is accessed via `task['inputs'][0]`
- Reference implementation is accessed via `task['infos'][0]`

