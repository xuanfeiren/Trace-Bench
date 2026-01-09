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
class KernelCode():
    """This is a kernel program container."""
    def __init__(self, initial_kernel_code: str="# This is a dummy kernel code. You should replace it with your own kernel code based on the task prompt and optimization objectives."):
        self.kernel_code = trace.node(initial_kernel_code, trainable=True)

    def forward(self, task: str) -> str:
        return self.kernel_code

class DummyGuide(Guide):
    """This is a dummy guide. Used for debugging the other parts of the code."""
    def __init__(self,*args, **kwargs):
        pass

    def get_feedback(self, task, response, info, **kwargs):
        return 1.0, "Dummy feedback"    
    
    def metric(self, task, response, info=None, **kwargs):
        return 1.0

class KernelGuide(DummyGuide):
    pass

from dataset.utils import create_matrix_multiplication_dataset
def create_single_task_dataset(task_idx: int):
    """
    Create a single task dataset from the matrix multiplication dataset.
    """
    ds = create_matrix_multiplication_dataset()
    return {'inputs': [ds[task_idx]['input']], 'infos': [ds[task_idx]['ref_arch_src']]}


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
    # initial_kernel_code = open("level1_prob1_cuda_custom_cuda_gpt5_example.txt").read()
    # agent = KernelAgent(initial_kernel_code=initial_kernel_code)
    agent = KernelCode()

    # Step 3: Initialize Optimizer
    optimizer = OptoPrimeV2(agent.parameters(), max_tokens=8192, initial_var_char_limit=10000)
    optimizer.objective = input_text

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
        use_best_candidate_to_explore=False,
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
    print(create_single_task_dataset(3)['infos'][0])