# Persistent Modal GPU Evaluator for Kernel Optimization

This setup provides a persistent Modal GPU container that stays warm across multiple kernel evaluations, significantly reducing overhead by avoiding container recreation for each evaluation.

## Architecture

### Key Components

1. **[modal_kernel_evaluator.py](modal_kernel_evaluator.py)**: Defines the persistent Modal container
   - `KernelEvaluator`: Modal class that runs on GPU and evaluates kernels
   - `LocalKernelEvaluatorWrapper`: Local context manager for interacting with the Modal container
   - Reuses the same image configuration as [evaluate_with_modal.py](evaluate_with_modal.py)

2. **[my_process_agents/kernel_PS.py](my_process_agents/kernel_PS.py)**: Main optimization script
   - `KernelAgent`: Trainable container for kernel code
   - `KernelGuide`: Evaluation guide that uses the persistent Modal evaluator
   - `main()`: Sets up and runs PrioritySearch optimization with persistent GPU

## How It Works

### Without Persistent Modal (Old Approach)
```python
# Each evaluation spawns a new container - slow!
for i in range(100):
    result = eval_single_sample_modal.remote(ref, code, verbose, gpu_arch)
    # Container is created, used once, then destroyed
```

### With Persistent Modal (New Approach)
```python
# Container is created once and reused
with LocalKernelEvaluatorWrapper(gpu="L40S") as evaluator:
    guide = KernelGuide(modal_evaluator=evaluator)
    for i in range(100):
        result = evaluator.evaluate(ref, code, verbose)
        # Same container is reused - fast!
```

### Container Lifecycle
1. **Initialization**: When entering the context manager, Modal spawns a GPU container
2. **Warm Period**: Container stays alive and warm during the entire training process
3. **Idle Timeout**: After the context exits, container stays warm for 10 minutes (configurable via `container_idle_timeout`)
4. **Automatic Shutdown**: If no new requests arrive within the idle timeout, Modal shuts down the container

## Usage

### Basic Usage

```bash
cd KernelBench/my_process_agents
python kernel_PS.py --task_idx 0 --num_steps 50 --gpu L40S
```

### Command Line Arguments

#### Required Arguments
- `--task_idx`: Task index from KernelBench dataset (default: 0)

#### Optimization Parameters
- `--num_steps`: Maximum number of optimization steps (default: 50)
- `--num_candidates`: Number of candidates for exploration (default: 1)
- `--num_threads`: Number of threads for parallel processing (default: 20)
- `--num_proposals`: Number of proposals per candidate (default: 1)
- `--algorithm`: Algorithm to use - `PS`, `PS_Summarizer`, `PS_epsNet_Summarizer`, `PS_epsNet` (default: PS)

#### Modal GPU Parameters
- `--gpu`: GPU type - `L40S`, `H100`, `A100`, `L4`, `T4`, `A10G` (default: L40S)
- `--verbose`: Print verbose evaluation output (flag, default: False)

#### Logging Parameters
- `--log_frequency`: How often to log results (default: 1)
- `--test_frequency`: How often to run evaluation (default: None)
- `--use_wandb`: Use Weights & Biases for logging (flag, default: False)
- `--project_name`: W&B project name (default: kernelbench-single-task)
- `--run_name`: W&B run name (default: kernel_task_{task_idx})

### Examples

#### Quick Test (10 steps, L40S GPU)
```bash
python kernel_PS.py --task_idx 0 --num_steps 10 --gpu L40S
```

#### Full Optimization with W&B Logging
```bash
python kernel_PS.py \
    --task_idx 0 \
    --num_steps 100 \
    --num_candidates 5 \
    --gpu H100 \
    --algorithm PS_Summarizer \
    --use_wandb \
    --project_name my-kernel-optimization \
    --run_name experiment_1
```

#### Verbose Output for Debugging
```bash
python kernel_PS.py --task_idx 0 --num_steps 20 --verbose
```

## Testing the Setup

### Test Persistent Evaluator Only
```bash
cd KernelBench
modal run modal_kernel_evaluator.py
```

This will:
1. Spawn a Modal container once
2. Run 3 evaluations using the same container
3. Demonstrate container reuse

### Test Full Optimization Pipeline
```bash
cd KernelBench/my_process_agents
python kernel_PS.py --task_idx 0 --num_steps 5 --verbose
```

## Key Benefits

1. **Performance**: Container initialization happens once instead of per-evaluation
2. **Cost**: Reduced Modal compute costs by minimizing cold starts
3. **Simplicity**: Context manager ensures proper resource cleanup
4. **Flexibility**: Easy to switch GPU types via command line arguments

## Technical Details

### Modal Configuration
- **Image**: CUDA 12.4.0 development image with Ubuntu 22.04
- **Python**: 3.10
- **Packages**: PyTorch 2.5.0, Triton, LiteLLM, and KernelBench dependencies
- **Environment**: KernelBench added to PYTHONPATH
- **GPU**: Configurable via `--gpu` argument
- **Timeout**: 3600 seconds (1 hour) per container session
- **Idle Timeout**: 600 seconds (10 minutes) after last evaluation

### Evaluation Scoring
- **Compilation Failed**: score = 0.0
- **Compiles but Incorrect**: score = 0.0
- **Compiles and Correct**: score = 1 / runtime_seconds (higher is better)

### KernelGuide Implementation
The `KernelGuide.get_feedback()` method:
1. Takes the LLM-generated kernel code (`response`)
2. Evaluates against reference implementation (`info`)
3. Returns score and detailed feedback
4. Reuses the same Modal container for all evaluations

## Troubleshooting

### Import Errors
If you see `ModuleNotFoundError: No module named 'modal_kernel_evaluator'`:
- Ensure you're running from `KernelBench/my_process_agents/` directory
- The script automatically adds the parent directory to Python path

### Modal Authentication
If you see Modal authentication errors:
```bash
modal token new
```

### GPU Availability
If your requested GPU is unavailable, try:
- `--gpu L40S` (most commonly available)
- `--gpu L4` (budget option)
- Check Modal dashboard for GPU availability

## Future Improvements

- [ ] Support for multi-GPU parallel evaluation
- [ ] Batch evaluation of multiple kernels
- [ ] Custom performance metrics beyond runtime
- [ ] Integration with automated hyperparameter tuning
- [ ] Support for Triton kernels (currently CUDA only)
