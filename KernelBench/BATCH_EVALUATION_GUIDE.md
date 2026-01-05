# Batch Evaluation Guide - Evaluating Multiple Kernels with Modal

## Problem 1: Evaluation Error Fixed ✅

### The Issue

In [kernelbench_simple_test.py](my_process_agents/kernelbench_simple_test.py), the code was trying to access Modal results as **object attributes** instead of **dictionary keys**.

**Why?** Modal returns a plain dict (not a Pydantic object) to avoid serialization issues on macOS. See [evaluate_with_modal.py:96-112](evaluate_with_modal.py#L96-L112):

```python
# Modal explicitly converts to dict for safe serialization
if hasattr(result, 'model_dump'):
    return result.model_dump()  # ← Returns dict
```

### The Fix

Changed from object attributes to dictionary access:

```python
# ❌ BEFORE (Wrong)
print(f"Compiled: {result.compiled}")
print(f"Correct: {result.correctness}")

# ✅ AFTER (Correct)
print(f"Compiled: {result.get('compiled')}")
print(f"Correct: {result.get('correctness')}")
```

All errors in [kernelbench_simple_test.py](my_process_agents/kernelbench_simple_test.py) have been fixed!

---

## Problem 2: Can Modal Evaluate Multiple Kernels? ✅ YES!

### Short Answer

**YES!** Modal is **perfect** for evaluating multiple kernels. Modal supports:

1. ✅ **Parallel execution** - Run multiple evaluations concurrently
2. ✅ **Auto-scaling** - Modal spins up as many GPU containers as needed
3. ✅ **Cost-efficient** - Only pay for what you use
4. ✅ **No local GPU needed** - Everything runs on cloud GPUs

---

## Three Ways to Evaluate Multiple Kernels

### Method 1: Sequential Evaluation (Simple)

**Good for:** Small batches (< 10 kernels), debugging

```python
from evaluate_with_modal import app, eval_single_sample_modal, gpu_arch_mapping

with app.run():
    results = []

    for problem in problems:
        # Generate CUDA code
        cuda_code = generate_cuda(problem['input'])

        # Evaluate (blocks until done)
        result = eval_single_sample_modal.remote(
            ref_arch_src=problem['ref_arch_src'],
            custom_cuda=cuda_code,
            verbose=True,
            gpu_arch=gpu_arch_mapping["L40S"]
        )

        results.append(result)
```

**Pros:**
- Simple, easy to understand
- Good for debugging (see results immediately)

**Cons:**
- ❌ Slow (evaluations run one-by-one)
- ❌ Doesn't utilize Modal's parallelism

**Time:** ~2-5 minutes per kernel = 10-25 minutes for 5 kernels

---

### Method 2: Parallel Evaluation with `map()` (Recommended) ⚡

**Good for:** Large batches (10+ kernels), production workflows

```python
from evaluate_with_modal import app, eval_single_sample_modal, gpu_arch_mapping

with app.run():
    # Generate all CUDA codes first
    cuda_codes = [generate_cuda(p['input']) for p in problems]

    # Evaluate ALL in parallel using Modal's map()!
    results = list(eval_single_sample_modal.map(
        [p['ref_arch_src'] for p in problems],     # List of reference codes
        cuda_codes,                                 # List of CUDA codes
        [True] * len(problems),                    # List of verbose flags
        [gpu_arch_mapping["L40S"]] * len(problems) # List of GPU architectures
    ))

    # All done! Results is a list of dicts
    for idx, result in enumerate(results):
        print(f"Problem {idx}: {result.get('correctness')}")
```

**Pros:**
- ✅ **MUCH FASTER** - Modal runs evaluations concurrently
- ✅ Scales to 100+ kernels easily
- ✅ Efficient use of cloud resources

**Cons:**
- Requires all inputs prepared upfront
- Results come back all at once (not streaming)

**Time:** ~2-5 minutes total (regardless of how many kernels, up to Modal's concurrency limit)

**How it works:**
```
Sequential:
Problem 1 → [2 min] → Done → Problem 2 → [2 min] → Done → ... (10 min total for 5)

Parallel (map):
Problem 1 → [2 min] ┐
Problem 2 → [2 min] ├─→ All done in ~2-3 min!
Problem 3 → [2 min] │
Problem 4 → [2 min] │
Problem 5 → [2 min] ┘
```

---

### Method 3: Streaming Parallel with `starmap()` (Advanced)

**Good for:** Very large batches, want progress updates

```python
from evaluate_with_modal import app, eval_single_sample_modal, gpu_arch_mapping

with app.run():
    # Generate all inputs
    inputs = [
        (p['ref_arch_src'], generate_cuda(p['input']), True, gpu_arch_mapping["L40S"])
        for p in problems
    ]

    # Stream results as they complete
    results = []
    for idx, result in enumerate(eval_single_sample_modal.starmap(inputs)):
        print(f"✓ Problem {idx+1} completed: {result.get('correctness')}")
        results.append(result)
```

**Pros:**
- ✅ Parallel execution
- ✅ See results as they complete (streaming)
- ✅ Good for monitoring long-running batches

---

## Complete Example: Batch Evaluation Script

I've created [batch_evaluate_modal.py](my_process_agents/batch_evaluate_modal.py) that demonstrates both sequential and parallel evaluation.

### Usage

```bash
cd KernelBench/my_process_agents

# Run batch evaluation (parallel mode)
modal run batch_evaluate_modal.py
```

### Configuration

Edit the `main()` function in [batch_evaluate_modal.py](my_process_agents/batch_evaluate_modal.py):

```python
# Configuration
LEVEL = 1                    # Difficulty level (1, 2, or 3)
NUM_PROBLEMS = 5            # How many problems to evaluate
GPU = "L40S"                # GPU type (L40S, H100, A100, T4, A10G)
MODE = "parallel"           # "sequential" or "parallel"
VERBOSE = False             # Print verbose evaluation output
```

### Example Output

```
================================================================================
PARALLEL EVALUATION: 5 problems
Modal will run these concurrently on cloud GPUs!
================================================================================

📝 Generating CUDA code for all problems...
  [1/5] Generating for problem 100...
  [2/5] Generating for problem 101...
  [3/5] Generating for problem 102...
  [4/5] Generating for problem 103...
  [5/5] Generating for problem 104...
✓ Generated 5 CUDA kernels

🚀 Submitting all evaluations to Modal (parallel execution)...
✓ Completed 5 evaluations

[1/5] Problem 100: ✅ PASS
          Runtime: 0.000245s (Speedup: 3.20x)
[2/5] Problem 101: ✅ PASS
          Runtime: 0.000189s (Speedup: 4.15x)
[3/5] Problem 102: ❌ FAIL
[4/5] Problem 103: ✅ PASS
          Runtime: 0.000312s (Speedup: 2.85x)
[5/5] Problem 104: ✅ PASS
          Runtime: 0.000156s (Speedup: 5.20x)

💾 Results saved to: batch_results_parallel_level1.json

================================================================================
SUMMARY
================================================================================
Total Problems:      5
Compiled:           5/5 (100.0%)
Correct:            4/5 (80.0%)
Average Speedup:    3.85x
Max Speedup:        5.20x
================================================================================
```

---

## Scalability: How Many Kernels Can Modal Handle?

### Modal's Limits

| Scenario | Kernels | Time | Cost Estimate* |
|----------|---------|------|----------------|
| **Small batch** | 1-10 | ~2-5 min | $0.10-0.50 |
| **Medium batch** | 10-50 | ~5-10 min | $0.50-2.50 |
| **Large batch** | 50-100 | ~10-20 min | $2.50-5.00 |
| **Very large** | 100-500 | ~20-60 min | $5.00-25.00 |

*Estimates based on L40S GPU pricing (~$1/hour)

### Concurrency

Modal automatically scales to your workload:
- ✅ Default: Up to **100 concurrent containers** per account
- ✅ Can request higher limits from Modal support
- ✅ Each container uses 1 GPU (L40S/H100/A100)

### Best Practices

1. **Use `map()` for large batches** (10+ kernels)
   ```python
   results = list(eval_single_sample_modal.map(...))
   ```

2. **Generate all codes before evaluating**
   - Don't interleave LLM generation and evaluation
   - Generate → Then evaluate in batch

3. **Save intermediate results**
   ```python
   # Save after each batch
   with open(f'results_batch_{batch_num}.json', 'w') as f:
       json.dump(results, f, indent=2)
   ```

4. **Use batching for very large datasets** (500+ kernels)
   ```python
   # Process in batches of 100
   for batch_idx in range(0, len(problems), 100):
       batch = problems[batch_idx:batch_idx+100]
       results = evaluate_batch(batch)
       save_results(results, f'batch_{batch_idx}.json')
   ```

---

## Workflow for Your Use Case

You mentioned: **"Later I may build a workflow that generates many kernels and should eval for multiple times"**

### Recommended Workflow

```python
# Step 1: Generate many CUDA kernels (e.g., with evolution/search algorithm)
generated_kernels = []
for iteration in range(num_iterations):
    for variant in variants:
        cuda_code = your_generation_algorithm(variant)
        generated_kernels.append({
            'iteration': iteration,
            'variant': variant,
            'code': cuda_code
        })

# Step 2: Evaluate all kernels in parallel with Modal
from evaluate_with_modal import app, eval_single_sample_modal, gpu_arch_mapping

with app.run():
    # Prepare inputs
    ref_codes = [problem['ref_arch_src']] * len(generated_kernels)
    cuda_codes = [k['code'] for k in generated_kernels]
    verbose = [False] * len(generated_kernels)
    gpu_archs = [gpu_arch_mapping["L40S"]] * len(generated_kernels)

    # Evaluate ALL in parallel
    print(f"Evaluating {len(generated_kernels)} kernels in parallel...")
    results = list(eval_single_sample_modal.map(
        ref_codes, cuda_codes, verbose, gpu_archs
    ))

    # Process results
    for kernel_info, result in zip(generated_kernels, results):
        kernel_info['result'] = result

# Step 3: Analyze and select best kernels
best_kernels = sorted(
    [k for k in generated_kernels if k['result'].get('correctness')],
    key=lambda k: k['result'].get('runtime', float('inf'))
)[:10]

print(f"Top 10 fastest kernels:")
for i, k in enumerate(best_kernels):
    print(f"{i+1}. Runtime: {k['result']['runtime']:.6f}s, "
          f"Speedup: {k['result'].get('metadata', {}).get('speedup', 'N/A')}x")
```

---

## Comparison: Modal vs Server for Batch Evaluation

| Feature | Modal | Self-Hosted Server |
|---------|-------|-------------------|
| **Setup** | ✅ Easy (cloud account) | ⚠️ Need GPU hardware |
| **Scalability** | ✅ Auto-scales to 100+ GPUs | ⚠️ Limited by your GPUs |
| **Cost (small batches)** | ✅ Pay-per-use (~$0.50-2) | ✅ Free (if you own GPUs) |
| **Cost (large batches)** | ⚠️ Can get expensive | ✅ Fixed cost |
| **Parallel execution** | ✅ Up to 100 concurrent | ⚠️ Limited to your GPUs (e.g., 4-8) |
| **Best for** | ✅ Prototyping, experiments | ✅ Production, large-scale |

---

## Summary

### ✅ Problem 1: Fixed

The evaluation error has been fixed in [kernelbench_simple_test.py](my_process_agents/kernelbench_simple_test.py). The code now correctly accesses Modal results as dictionaries.

### ✅ Problem 2: Answered

**YES, Modal is excellent for evaluating multiple kernels!**

- Use **sequential** for small batches (< 10 kernels)
- Use **parallel with `map()`** for large batches (10+ kernels) ⚡ **RECOMMENDED**
- Modal auto-scales up to 100+ concurrent GPUs
- See [batch_evaluate_modal.py](my_process_agents/batch_evaluate_modal.py) for complete example

### Quick Start

```bash
# Test with your fixed simple test
cd KernelBench/my_process_agents
modal run kernelbench_simple_test.py

# Run batch evaluation (5 problems in parallel)
modal run batch_evaluate_modal.py
```

### For Your Workflow

If you're building an evolution/search algorithm that generates many kernels:

1. ✅ Use Modal's `map()` for parallel evaluation
2. ✅ Generate all kernels first, then evaluate in batch
3. ✅ Process results to select best performers
4. ✅ Can easily handle 100+ kernels per run

Modal is **perfect** for this use case! 🚀
