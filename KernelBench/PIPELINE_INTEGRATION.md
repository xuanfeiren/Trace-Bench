# Pipeline Integration Guide

## Summary

You have created:
1. **evaluate_with_modal.py** - Low-level Modal function for CUDA kernel evaluation
2. **kernel_evaluator.py** - High-level wrapper class (currently requires `modal run`)
3. **pipeline_integration_example.py** - Working example for pipeline integration

## How to Integrate Into Your Pipeline

### Option 1: Direct Modal Function Calls (Recommended for Pipelines)

Use the `@app.local_entrypoint()` decorator on your pipeline code:

```python
from evaluate_with_modal import app, eval_single_sample_modal, gpu_arch_mapping

@app.local_entrypoint()
def my_pipeline():
    # Your pipeline code here

    # Call evaluations
    result = eval_single_sample_modal.remote(
        ref_arch_src, custom_cuda, verbose=False,
        gpu_arch=gpu_arch_mapping["L40S"],
        num_correct_trials=5,
        num_perf_trials=100
    )

    # Process result
    if result['correctness']:
        print(f"Kernel passed! Runtime: {result['runtime']}ms")
```

**Run with**: `modal run your_pipeline.py`

### Option 2: Wrapper Class (In Development)

The `KernelEvaluator` wrapper class is cleaner but currently also requires `modal run`. It's useful for organizing your code:

```python
from kernel_evaluator import KernelEvaluator
from evaluate_with_modal import app

@app.local_entrypoint()
def my_pipeline():
    evaluator = KernelEvaluator(gpu="L40S", num_perf_trials=50)

    result = evaluator.evaluate(ref_arch_src, custom_cuda)
    print(f"Runtime: {result['runtime']}ms")
```

## Key Points

### Container Warmth
- With `min_containers=1`, Modal keeps 1 GPU container always warm
- First evaluation: ~30-60s (container startup)
- Subsequent evaluations: Fast (reuses warm container)
- **Cost**: You pay for the container even when idle

### Sequential vs Parallel

**Sequential** (one at a time):
```python
result1 = eval_single_sample_modal.remote(...)
result2 = eval_single_sample_modal.remote(...)
```

**Parallel** (runs simultaneously):
```python
future1 = eval_single_sample_modal.spawn(...)
future2 = eval_single_sample_modal.spawn(...)
result1 = future1.get()
result2 = future2.get()
```

## Complete Pipeline Example

```python
from evaluate_with_modal import app, eval_single_sample_modal, gpu_arch_mapping

@app.local_entrypoint()
def kernel_optimization_pipeline():
    # Load your kernels
    kernels = load_kernel_variants()

    gpu = "L40S"
    gpu_arch = gpu_arch_mapping[gpu]

    best_runtime = float('inf')
    best_kernel = None

    # Evaluate each kernel
    for i, kernel_code in enumerate(kernels):
        print(f"Evaluating kernel variant {i+1}/{len(kernels)}")

        result = eval_single_sample_modal.remote(
            ref_arch_src,
            kernel_code,
            verbose=False,
            gpu_arch=gpu_arch,
            num_correct_trials=5,
            num_perf_trials=100
        )

        if result['correctness'] and result['runtime'] < best_runtime:
            best_runtime = result['runtime']
            best_kernel = kernel_code
            print(f"  New best! Runtime: {best_runtime:.2f}ms")
        elif result['correctness']:
            print(f"  Passed. Runtime: {result['runtime']:.2f}ms")
        else:
            print(f"  Failed correctness tests")

    print(f"\nBest kernel runtime: {best_runtime:.2f}ms")
    return best_kernel
```

Run with: `modal run my_pipeline.py`

## Files Created

- `evaluate_with_modal.py` - Modal app and GPU evaluation function
- `kernel_evaluator.py` - Wrapper class for easier use
- `simple_eval_example.py` - Simple example (needs work)
- `pipeline_integration_example.py` - Working pipeline example ✓
- `demo_kernel_evaluator.py` - Multiple demo scenarios

## Quick Test

To verify your setup works:

```bash
modal run pipeline_integration_example.py
```

This should:
1. Initialize Modal app
2. Run 2 sequential evaluations
3. Run 3 parallel evaluations
4. Print runtimes (~6.26ms for the example kernel)
