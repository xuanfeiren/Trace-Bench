# Use Priority Search in Trace to search for the best kernel implementation
import sys
import os
import numpy as np
# import torch
import time
import litellm
from datasets import load_dataset

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from opto import trace
from opto.trainer.loggers import DefaultLogger, WandbLogger
from opto.optimizers import OptoPrimeV2
from opto.trainer.guide import Guide
import secrets_local

litellm.drop_params = True
litellm.suppress_debug_info = True

# Import the persistent Modal evaluator
kernelbench_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(kernelbench_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from evaluate_with_modal import app, eval_single_sample_modal, gpu_arch_mapping
from opto.optimizers.utils import print_color

np.random.seed(10)

@trace.model
class KernelAgent():
    """This is a kernel program container."""
    def __init__(self, initial_kernel_code: str="# This is a dummy kernel code. You should replace it with your own kernel code based on the task prompt and optimization objectives."):
        self.kernel_code = trace.node(initial_kernel_code, trainable=True)

    @trace.bundle()
    def get_kernel_code(self, task_prompt: str, kernel_code: str) -> str:
        return kernel_code

    def forward(self, task: str) -> str:
        return self.get_kernel_code(task, self.kernel_code)

class DummyGuide(Guide):
    """This is a dummy guide. Used for debugging the other parts of the code."""
    def __init__(self,*args, **kwargs):
        pass

    def get_feedback(self, task, response, info, **kwargs):
        return 1.0, "Dummy feedback"    
    
    def metric(self, task, response, info=None, **kwargs):
        return 1.0

class KernelGuide(Guide):
    """This is a kernel guide."""
    def __init__(self, gpu="L40S", verbose=False, num_correct_trials=5, num_perf_trials=100):
        self.gpu = gpu
        self.verbose = verbose
        self.num_correct_trials = num_correct_trials
        self.num_perf_trials = num_perf_trials

    def get_feedback(self, task, response, info, **kwargs):
        ref_arch_src = info 
        custom_cuda = response

        try:
            result = eval_single_sample_modal.remote(
                ref_arch_src,
                custom_cuda,
                self.verbose,
                gpu_arch_mapping[self.gpu],
                num_correct_trials=self.num_correct_trials,
                num_perf_trials=self.num_perf_trials
            )
        except Exception as e:
            error_msg = str(e)
            score = -1
            feedback = f"Evaluation failed with error:\n{error_msg}\n\n Please write a new kernel code based on the task prompt and optimization objectives."

            # save the response that cause the error
            # with open(f"error_response_{time.time()}.txt", "w") as f:
            #     f.write(custom_cuda)

            print_color(feedback, 'yellow')

            return score, feedback

        compiled = result.get('compiled', False)
        correctness = result.get('correctness', False)
        runtime = result.get('runtime', -1.0)
        runtime_stats = result.get('runtime_stats', {})
        metadata = result.get('metadata', {})

        if not compiled:
            score = 0.0
            # Extract detailed error information from metadata
            error_name = metadata.get('compilation_error_name', metadata.get('error_type', 'Unknown'))
            error_msg = metadata.get('compilation_error', metadata.get('error', 'Unknown error'))
            traceback_info = metadata.get('traceback', '')

            feedback = f"Compilation/Runtime failed. Please fix the error and try again.\n\n"
            feedback += f"Error Type: {error_name}\n"
            feedback += f"Error Message: {error_msg}\n"
            if traceback_info:
                feedback += f"\nDetailed Traceback:\n{traceback_info}\n"
            feedback += f"\nFull Metadata: {metadata}"

            # save the response that cause the error
            # with open(f"error_response_{time.time()}.txt", "w") as f:
            #     f.write(custom_cuda)
        elif not correctness:
            # save the response that cause the error
            # with open(f"error_response_{time.time()}.txt", "w") as f:
            #     f.write(custom_cuda)
            score = 0.0
            runtime_error_name = metadata.get('runtime_error_name', 'Unknown')
            runtime_error = metadata.get('runtime_error', 'Unknown error')

            feedback = f"The kernel code compiled but failed correctness tests.\n\n"
            feedback += f"Error Type: {runtime_error_name}\n"
            feedback += f"Error Details: {runtime_error}\n"
            feedback += f"\nPlease fix the logic error and try again.\n"
            feedback += f"\nFull Metadata: {metadata}"
        else:
            # save the response that cause the error
            # with open(f"success_response_{time.time()}.txt", "w") as f:
            #     f.write(custom_cuda)
            runtime_seconds = runtime / 1000.0 if runtime > 0 else float('inf')
            score = 1.0 / runtime_seconds if runtime_seconds > 0 else 0.0
            feedback = (
                f"The kernel code compiled and is correct!\n"
                f"Runtime: {runtime:.4f} ms\n"
                f"Runtime Stats: {runtime_stats}\n"
                f"Score (1/runtime_sec): {score:.4f}\n"
                f"Metadata: {metadata}"
            )
        print_color(feedback, 'yellow')
        return score, feedback

    def metric(self, task, response, info=None, **kwargs):
        score, _ = self.get_feedback(task, response, info, **kwargs)
        return score

def create_single_task_dataset(task_idx: int):
    ds = load_dataset("allenanie/kernelbench_with_prompts")
    examples = [ex for ex in ds['train'] if ex['backend'] == 'cuda']
    task = examples[task_idx]
    return {
        'inputs': [task['input']],
        'infos': [task['ref_arch_src']]
    }

@app.local_entrypoint()
def kernel_PS_train(
    task_idx: int = 0,
    num_steps: int = 20,
    num_candidates: int = 1,
    num_threads: int = 1,
    num_proposals: int = 1,
    log_frequency: int = 1,
    test_frequency: int = 1,
    algorithm_name: str = 'PS',
    gpu: str = 'L40S',
    verbose: bool = False,
    use_wandb: bool = False,
    project_name: str = 'kernelbench-single-task',
    run_name: str = None
):
    """
    Optimize a single kernel using PrioritySearch with persistent Modal GPU.
    Modal automatically maps CLI flags (e.g., --task-idx) to these arguments.
    """

    # Step 1: Load the task
    print(f"Loading task {task_idx} from KernelBench dataset...")
    task = create_single_task_dataset(task_idx)
    input_text = task['inputs'][0]
    ref_arch_src = task['infos'][0]
    print(f"Task loaded successfully. Task ID: {task_idx}")

    # Step 2: Initialize Agent with reference implementation as starting point
    # Shouldn't initialize with reference implementation here, because the reference implementation doesn't meet the requirements of custom CUDA kernel (for example, a ModelNew class is not defined)
    initial_kernel_code = open("level1_prob1_cuda_custom_cuda_gpt5_example.txt").read()
    agent = KernelAgent(initial_kernel_code=initial_kernel_code)

    # Step 3: Initialize Optimizer
    optimizer = OptoPrimeV2(agent.parameters(), max_tokens=8192, initial_var_char_limit=10000)
    optimizer.objective = """
    You are an expert CUDA kernel optimizer. Your goal is to write CORRECT, COMPILABLE, and FAST custom CUDA kernels.

    ## CRITICAL REQUIREMENTS (Must follow ALL of these):

    1. **Code Structure** - Your code MUST include:
       - A `ModelNew` class that inherits from `nn.Module`
       - `get_inputs()` function that returns a list of input tensors
       - `get_init_inputs()` function that returns initialization parameters
       - Proper imports: torch, torch.nn, torch.utils.cpp_extension.load_inline

    2. **CUDA Kernel Safety** - ALWAYS add bounds checking:
       ```cpp
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
       if (idx < total_elements) {  // ← CRITICAL: Prevent out-of-bounds access
           // Your kernel code here
       }
       ```
       **Without bounds checking, you WILL get "illegal memory access" errors!**

    3. **Type Consistency** - Avoid type mismatches:
       - Cast literals to correct type: `min(65535, (int)((size + threads - 1) / threads))`
       - Use consistent types: don't mix `int` and `long`
       - Be explicit with numeric types in CUDA kernels

    4. **Grid/Block Dimensions** - Calculate correctly:
       ```cpp
       const int threads = 256;  // Good default for most GPUs
       const int blocks = min(65535, (size + threads - 1) / threads);
       kernel<<<blocks, threads>>>(...);
       ```

    5. **Memory Access Patterns** - Optimize for GPU:
       - **Coalesced access**: Adjacent threads access adjacent memory
       - **Shared memory**: Use for frequently accessed data
       - **Grid-stride loops**: `for (int i = idx; i < size; i += blockDim.x * gridDim.x)`

    ## OPTIMIZATION STRATEGIES (in priority order):

    1. **Correctness First**: Code must compile and pass correctness tests before optimizing
    2. **Memory Bandwidth**: Minimize global memory access, use shared memory for reused data
    3. **Parallelization**: Maximize thread utilization, avoid warp divergence
    4. **Computation**: Use fast math operations, avoid expensive operations (division, sqrt)
    5. **Register Usage**: Keep register usage low to maximize occupancy

    ## COMMON ERRORS TO AVOID:

    - ❌ Missing bounds checking → Illegal memory access (XID 31 error)
    - ❌ Type mismatches in min/max → Compilation error
    - ❌ Missing ModelNew class → AttributeError
    - ❌ Wrong tensor dimensions → Runtime error
    - ❌ Race conditions in shared memory → Incorrect results
    - ❌ Incorrect __syncthreads() usage → Deadlock or incorrect results

    ## EXAMPLE TEMPLATE:

    ```python
    import torch
    import torch.nn as nn
    from torch.utils.cpp_extension import load_inline

    cuda_source = '''
    __global__ void my_kernel(const float* input, float* output, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {  // Bounds check!
            output[idx] = input[idx] * 2.0f;  // Your operation here
        }
    }

    torch::Tensor my_operation(torch::Tensor input) {
        auto output = torch::zeros_like(input);
        int size = input.numel();
        const int threads = 256;
        const int blocks = (size + threads - 1) / threads;

        my_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            size
        );
        return output;
    }
    '''

    cpp_source = "torch::Tensor my_operation(torch::Tensor input);"

    module = load_inline(
        name='my_kernel',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=['my_operation'],
        verbose=False
    )

    class ModelNew(nn.Module):
        def __init__(self):
            super(ModelNew, self).__init__()
            self.op = module

        def forward(self, input):
            return self.op.my_operation(input)

    def get_inputs():
        return [torch.randn(1024, device='cuda')]

    def get_init_inputs():
        return []
    ```

    ## YOUR TASK:
    - Study the reference implementation to understand the operation
    - Write a custom CUDA kernel that is FASTER than the reference
    - ALWAYS include bounds checking to prevent memory errors
    - Test your logic carefully - correctness is more important than speed
    - If compilation fails, fix the error based on the feedback
    - If correctness fails, debug the kernel logic
    - Once correct, optimize for performance using the strategies above
    """

    # Step 4: Setup Logging
    config_dict = {
        'task_idx': task_idx,
        'num_steps': num_steps,
        'num_candidates': num_candidates,
        'num_threads': num_threads,
        'num_proposals': num_proposals,
        'gpu': gpu,
        'algorithm': algorithm_name,
    }
    
    actual_run_name = run_name if run_name else f"kernel_task_{task_idx}"
    
    if use_wandb:
        logger = WandbLogger(project=project_name, verbose=True, name=actual_run_name, config=config_dict)
    else:
        logger = DefaultLogger(verbose=True)

    # Step 5: Initialize Guide
    print(f"\nUsing Modal GPU evaluator (GPU: {gpu})...")
    guide = KernelGuide(
        gpu=gpu,
        verbose=verbose,
        num_correct_trials=5,
        num_perf_trials=100
    )

    # Step 6: Create Algorithm
    if algorithm_name == 'PS':
        from opto.features.priority_search.priority_search import PrioritySearch
        algorithm = PrioritySearch(
            agent=agent,
            optimizer=optimizer,
            logger=logger
        )
    else:
        raise ValueError(f"Algorithm {algorithm_name} not implemented in this entrypoint.")

    # Step 7: Run Training
    print(f"\nStarting PrioritySearch optimization (max {num_steps} steps)...")
    start_time = time.time()

    algorithm.train(
        guide=guide,
        train_dataset=task,
        validate_dataset=task,
        test_dataset=task,
        batch_size=1,
        num_batches=1,
        num_steps=num_steps,
        num_threads=num_threads,
        num_eval_samples=1,
        validate_exploration_candidates=False,
        num_candidates=num_candidates,
        num_proposals=num_proposals,
        score_function='mean',
        log_frequency=log_frequency,
        test_frequency=test_frequency
    )

    duration = time.time() - start_time
    print(f"\nOptimization completed in {duration:.2f} seconds")

if __name__ == "__main__":
    # When running via 'modal run', this block is ignored.
    pass