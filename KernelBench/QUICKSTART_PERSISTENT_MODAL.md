# Quick Start: Persistent Modal GPU Evaluator

## What Changed

I've set up a **persistent Modal GPU evaluator** that significantly improves performance by reusing the same Modal container across multiple kernel evaluations instead of creating a new container for each evaluation.

### Files Created/Modified

1. **NEW: [modal_kernel_evaluator.py](modal_kernel_evaluator.py)**
   - Defines persistent Modal container that stays warm
   - `KernelEvaluator` class runs on Modal GPU
   - `LocalKernelEvaluatorWrapper` context manager for local use

2. **MODIFIED: [my_process_agents/kernel_PS.py](my_process_agents/kernel_PS.py)**
   - Updated `KernelGuide` to use persistent Modal evaluator
   - Added command-line arguments for GPU selection and verbose output
   - Main function now uses context manager to keep Modal container alive

3. **NEW: [test_persistent_modal.py](test_persistent_modal.py)**
   - Test script to verify setup works correctly
   - Demonstrates container reuse across multiple evaluations

4. **NEW: [PERSISTENT_MODAL_EVALUATOR_README.md](PERSISTENT_MODAL_EVALUATOR_README.md)**
   - Comprehensive documentation with examples

## How It Works

### Before (Old approach - slow)
```python
# Each call creates a NEW container
result = eval_single_sample_modal.remote(ref, code, verbose, gpu_arch)
# Container is destroyed after each evaluation
```

### After (New approach - fast)
```python
# Container created ONCE
with LocalKernelEvaluatorWrapper(gpu="L40S") as evaluator:
    guide = KernelGuide(modal_evaluator=evaluator)

    # All evaluations reuse the SAME container
    for i in range(100):
        result = evaluator.evaluate(ref, code, verbose)
```

## Quick Start

### Step 1: Test the Setup

```bash
cd /Users/xuanfeiren/Documents/Trace-Bench/KernelBench
python test_persistent_modal.py
```

This will:
- Load a test kernel
- Spawn a Modal container once
- Run 3 evaluations using the same container
- Verify everything works correctly

**Expected output:**
```
Testing Persistent Modal Evaluator Setup
[1/4] Loading test kernel from KernelBench dataset...
✓ Test kernel loaded successfully
[2/4] Loading custom kernel example...
✓ Custom kernel loaded successfully
[3/4] Initializing persistent Modal GPU container (L40S)...
✓ Modal container spawned successfully
[4/4] Running 3 evaluations using the SAME container...
   Evaluation 1/3...
   ✓ Compiled: True, Correct: True, Runtime: X.XX ms
   ...
SUCCESS! All evaluations completed using the same container.
```

### Step 2: Run Kernel Optimization

```bash
cd my_process_agents
python kernel_PS.py --task_idx 0 --num_steps 10 --verbose
```

This will:
- Initialize a persistent Modal GPU container (L40S by default)
- Load KernelBench task #0
- Run PrioritySearch optimization for 10 steps
- All evaluations reuse the same Modal container
- Container stays warm for 10 minutes after completion

### Step 3: Full Optimization

```bash
python kernel_PS.py \
    --task_idx 0 \
    --num_steps 50 \
    --num_candidates 5 \
    --gpu L40S \
    --algorithm PS_Summarizer
```

## Command Line Options

### Essential Options
- `--task_idx 0` - Which KernelBench task to optimize
- `--num_steps 50` - Number of optimization iterations
- `--gpu L40S` - GPU type (L40S, H100, A100, L4, T4, A10G)
- `--verbose` - Show detailed evaluation output

### Advanced Options
- `--algorithm PS` - Algorithm: PS, PS_Summarizer, PS_epsNet_Summarizer, PS_epsNet
- `--num_candidates 5` - Number of exploration candidates
- `--num_threads 20` - Parallel processing threads
- `--use_wandb` - Enable W&B logging
- `--project_name my-project` - W&B project name

## Key Benefits

1. **Much Faster** - No container recreation overhead per evaluation
2. **Cost Efficient** - Reduced Modal compute costs (fewer cold starts)
3. **Simple** - Same interface, just faster
4. **Flexible** - Easy GPU switching via `--gpu` argument

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Local Machine (kernel_PS.py)                          │
│                                                         │
│  ┌──────────────┐      ┌────────────────────┐          │
│  │ KernelAgent  │──────│ PrioritySearch     │          │
│  │ (trainable)  │      │ Optimizer          │          │
│  └──────────────┘      └─────────┬──────────┘          │
│                                  │                      │
│                        ┌─────────▼──────────┐           │
│                        │  KernelGuide       │           │
│                        │  (evaluator)       │           │
│                        └─────────┬──────────┘           │
│                                  │                      │
│            ┌─────────────────────▼──────────┐           │
│            │ LocalKernelEvaluatorWrapper    │           │
│            │ (context manager)              │           │
│            └─────────────┬──────────────────┘           │
└──────────────────────────┼──────────────────────────────┘
                           │ Modal API
                           │
┌──────────────────────────▼──────────────────────────────┐
│  Modal Cloud (GPU Container - stays warm!)             │
│                                                         │
│  ┌───────────────────────────────────────────────┐     │
│  │  KernelEvaluator (persistent)                 │     │
│  │                                               │     │
│  │  - Initialized once                          │     │
│  │  - Evaluates all kernels                     │     │
│  │  - Stays warm for 10 min after last use      │     │
│  │  - GPU: L40S/H100/A100/etc                   │     │
│  └───────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## Troubleshooting

### "Module not found: modal_kernel_evaluator"
**Solution:** Make sure you're in the correct directory
```bash
cd /Users/xuanfeiren/Documents/Trace-Bench/KernelBench/my_process_agents
python kernel_PS.py ...
```

### "Modal authentication required"
**Solution:** Authenticate with Modal
```bash
modal token new
```

### GPU unavailable
**Solution:** Try a different GPU type
```bash
python kernel_PS.py --gpu L4  # Budget option
python kernel_PS.py --gpu L40S  # Default, usually available
```

## Next Steps

1. ✅ Test the setup: `python test_persistent_modal.py`
2. ✅ Run quick test: `python kernel_PS.py --task_idx 0 --num_steps 5`
3. ✅ Run full optimization: `python kernel_PS.py --task_idx 0 --num_steps 50`
4. 📖 Read full docs: [PERSISTENT_MODAL_EVALUATOR_README.md](PERSISTENT_MODAL_EVALUATOR_README.md)

## Questions?

- **"How much faster is this?"** - Container reuse eliminates ~30-60s initialization per evaluation
- **"Does it cost more?"** - No, it costs less due to reduced cold starts
- **"Can I use different GPUs?"** - Yes, use `--gpu` argument
- **"How long does container stay warm?"** - 10 minutes after last use (configurable in code)
