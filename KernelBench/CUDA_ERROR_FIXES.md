# CUDA Context Corruption Fixes

## Problem Summary

When running kernel optimization with PrioritySearch, multiple errors occurred:

1. **Compilation errors**: LLM-generated kernels had C++ type mismatches
2. **CUDA XID 31 errors**: Buggy kernels caused GPU hardware faults (illegal memory access)
3. **Persistent CUDA corruption**: After one kernel crashed, all subsequent evaluations failed
4. **Missing exception handling**: KernelBench didn't properly catch all error types

## Root Causes

### 1. No Process Isolation
- Modal container reused the same Python process for all evaluations
- CUDA context was shared across evaluations
- When one kernel caused illegal memory access, CUDA context became corrupted
- Future evaluations failed at `torch.manual_seed()` before even running

### 2. Incomplete Exception Handling in KernelBench
**File**: `external/KernelBench/src/eval.py:532`

Original code only caught `RuntimeError`:
```python
except RuntimeError as e:  # BUG: Doesn't catch TypeError, AttributeError, etc.
```

When `ModelNew` was `None`, calling `ModelNew(*init_inputs)` raised `TypeError` (not `RuntimeError`), causing:
- Exception propagated uncaught
- `graceful_eval_cleanup()` never called
- CUDA resources left in corrupted state

### 3. Missing ModelNew Validation
**File**: `external/KernelBench/src/eval.py:202`

Original code returned `None` silently when `ModelNew` class wasn't defined:
```python
ModelNew = context.get("ModelNew")
return ModelNew  # Returns None without raising exception
```

This caused the error to appear later as `TypeError` instead of being caught during compilation.

## Fixes Applied

### Fix 1: Process Isolation (Critical)
**File**: `my_process_agents/evaluate_with_modal.py`

Wrapped `eval_kernel_against_ref` in a subprocess using `multiprocessing.Pool`:

```python
def _eval_in_process(ref_arch_src, custom_cuda, verbose, gpu_arch, num_correct_trials, num_perf_trials):
    # ... evaluation code ...
    return result.model_dump()

@app.function(...)
def eval_single_sample_modal(...):
    with mp.Pool(1) as pool:
        result = pool.apply_async(
            _eval_in_process,
            args=(...)
        ).get(timeout=3500)
    return result
```

**Why this works**:
- Each kernel evaluation runs in an isolated subprocess
- When subprocess exits, CUDA context is completely destroyed
- Buggy kernels can't contaminate future evaluations
- GPU memory is fully cleaned up between evaluations

### Fix 2: Broader Exception Catching
**File**: `external/KernelBench/src/eval.py:532`

Changed from `except RuntimeError` to `except Exception`:

```python
except Exception as e:  # Catches TypeError, RuntimeError, AttributeError, etc.
    print(f"Failed to load custom CUDA kernel...")
    graceful_eval_cleanup(context, device, tempfile)
    return KernelExecResult(compiled=True, correctness=False, metadata=metadata)
```

**Why this works**:
- Now catches `TypeError` when `ModelNew` is `None`
- Ensures `graceful_eval_cleanup()` is always called
- Prevents CUDA context corruption

### Fix 3: Validate ModelNew Exists
**File**: `external/KernelBench/src/eval.py:202-208`

Added validation before returning:

```python
ModelNew = context.get("ModelNew")
if ModelNew is None:
    raise AttributeError(
        "ModelNew class not found in custom model source code. "
        "The code must define a 'ModelNew' class that inherits from nn.Module."
    )
return ModelNew
```

**Why this works**:
- Catches missing `ModelNew` during compilation phase
- Raises proper exception that's caught by outer `try/except`
- Returns `compiled=False` instead of causing later `TypeError`
- Consistent with Triton backend behavior

### Fix 4: Proper Initial Code
**File**: `my_process_agents/kernel_PS_modal.py:155`

Initialize agent with reference implementation:

```python
agent = KernelAgent(initial_kernel_code=ref_arch_src)
```

**Why this helps**:
- Gives optimizer a valid working kernel as starting point
- Avoids starting with dummy code that doesn't define `ModelNew`

## How the Fixes Work Together

1. **Before evaluation**: Kernel code is validated for `ModelNew` class (Fix 3)
2. **During evaluation**: Runs in isolated subprocess (Fix 1)
3. **If error occurs**: Broad exception handler catches it (Fix 2)
4. **After evaluation**: Subprocess exits, CUDA fully cleaned up (Fix 1)
5. **Next evaluation**: Fresh subprocess with clean CUDA context (Fix 1)

## Expected Behavior Now

### Compilation Errors
```
✓ Caught during compilation phase
✓ Returns: compiled=False, correctness=False
✓ CUDA context remains healthy
✓ Next evaluation works fine
```

### Runtime Errors (illegal memory access, XID faults)
```
✓ Caught in isolated subprocess
✓ Returns: compiled=True, correctness=False
✓ Subprocess exits and cleans up CUDA
✓ Next evaluation gets fresh subprocess with clean CUDA
```

### Multiple Sequential Evaluations
```
✓ Each runs in its own subprocess
✓ No CUDA context contamination
✓ All evaluations are isolated
✓ Can run indefinitely without corruption
```

## Testing

To verify the fixes work:

```bash
modal run my_process_agents/kernel_PS_modal.py --num-steps 10
```

Expected:
- Compilation errors should be handled gracefully
- CUDA errors should not propagate to future evaluations
- All evaluations should get proper feedback scores
- No persistent "illegal memory access" errors

## Trade-offs

### Process Isolation Overhead
- **Pro**: Complete isolation, prevents contamination
- **Con**: ~1-2 second overhead per evaluation for subprocess creation
- **Verdict**: Worth it - correctness over speed

### Container Warmth
- Kept `min_containers=1` for fast startup
- Process isolation prevents most CUDA issues
- Severe GPU XID errors may still require container restart (Modal handles this automatically)

## Files Changed

1. `external/KernelBench/src/eval.py` (2 fixes)
2. `my_process_agents/evaluate_with_modal.py` (process isolation)
3. `my_process_agents/kernel_PS_modal.py` (proper initialization)
