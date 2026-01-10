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
# import secrets_local

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

# class DummyGuide(Guide):
#     """This is a dummy guide. Used for debugging the other parts of the code."""
#     def __init__(self,*args, **kwargs):
#         pass

#     def get_feedback(self, task, response, info, **kwargs):
#         return 1.0, "Dummy feedback"    
    
#     def metric(self, task, response, info=None, **kwargs):
#         return 1.0
# address the import issue
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from guide.evaluate import evaluate
class KernelGuide(Guide):
    def __init__(self, gpu="L40S", verbose=False, num_correct_trials=1, num_perf_trials=5):
        self.gpu = gpu
        self.verbose = verbose
        self.num_correct_trials = num_correct_trials
        self.num_perf_trials = num_perf_trials
    def get_feedback(self, task, response, info, **kwargs):
        score, feedback = evaluate(
                    ref_arch_src=info,
                    custom_cuda=response,
                    num_correct_trials=self.num_correct_trials,
                    num_perf_trials=self.num_perf_trials
            )
        print_color(f"Score: {score}", 'green')
        print_color(f"Feedback: {feedback}", 'yellow')
        # breakpoint()
        return score, feedback
        
    def metric(self, task, response, info=None, **kwargs):
        score, _ = self.get_feedback(task, response, info, **kwargs)
        return score

from dataset.utils import create_matrix_multiplication_dataset
def create_single_task_dataset(task_idx: int):
    """
    Create a single task dataset from the matrix multiplication dataset.
    """
    ds = create_matrix_multiplication_dataset()
    return {'inputs': [ds[task_idx]['input']], 'infos': [ds[task_idx]['ref_arch_src']]}


def main():
    """
    Main function to optimize a single kernel using PrioritySearch.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Optimize CUDA kernels using PrioritySearch algorithm"
    )
    
    # Task parameters
    parser.add_argument(
        '--task-idx',
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        help='Task index from dataset (default: 0, choices: 0-15)'
    )
    
    # Training parameters
    parser.add_argument(
        '--num-steps',
        type=int,
        default=20,
        help='Number of optimization steps (default: 20)'
    )
    parser.add_argument(
        '--num-candidates',
        type=int,
        default=1,
        help='Number of candidates per step (default: 1)'
    )
    parser.add_argument(
        '--num-threads',
        type=int,
        default=1,
        help='Number of threads for parallel evaluation (default: 1)'
    )
    parser.add_argument(
        '--num-proposals',
        type=int,
        default=1,
        help='Number of proposals per candidate (default: 1)'
    )
    
    # Logging parameters
    parser.add_argument(
        '--log-frequency',
        type=int,
        default=1,
        help='Frequency of logging (default: 1)'
    )
    parser.add_argument(
        '--test-frequency',
        type=int,
        default=1,
        help='Frequency of testing (default: 1)'
    )
    
    # Algorithm parameters
    parser.add_argument(
        '--algorithm',
        type=str,
        default='PS',
        choices=['PS', 'PS_Summarizer', 'PS_epsNet_Summarizer', 'PS_epsNet'],
        help='Algorithm to use (default: PS)'
    )
    parser.add_argument(
        '--epsilon',
        type=float,
        default=0.1,
        help='Epsilon value for EpsilonNetPS (default: 0.1)'
    )
    parser.add_argument(
        '--gpu',
        type=str,
        default='L40S',
        help='GPU type for Modal evaluation (default: L40S)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    # WandB parameters
    parser.add_argument(
        '--use-wandb',
        action='store_true',
        help='Enable Weights & Biases logging'
    )
    parser.add_argument(
        '--project-name',
        type=str,
        default='kernelbench-single-task',
        help='WandB project name (default: kernelbench-single-task)'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='WandB run name (default: kernel_task_{task_idx})'
    )
    
    args = parser.parse_args()
    num_threads = args.num_threads

    # Step 1: Load the task
    print(f"Loading task {args.task_idx} from KernelBench dataset...")
    task = create_single_task_dataset(args.task_idx)
    input_text = task['inputs'][0]
    ref_arch_src = task['infos'][0]
    print(f"Task loaded successfully. Task ID: {args.task_idx}")

    # Step 2: Initialize Agent
    agent = KernelCode()

    # Step 3: Initialize Optimizer
    optimizer = OptoPrimeV2(agent.parameters(), max_tokens=8192, initial_var_char_limit=10000)
    optimizer.objective = input_text

    # Step 4: Setup Logging
    config_dict = {
        'task_idx': args.task_idx,
        'epsilon': args.epsilon,
        'num_steps': args.num_steps,
        'num_candidates': args.num_candidates,
        'num_threads': args.num_threads,
        'num_proposals': args.num_proposals,
        'gpu': args.gpu,
        'algorithm': args.algorithm,
    }
    
    actual_run_name = args.run_name if args.run_name else f"task_{args.task_idx}"
    
    if args.use_wandb:
        logger = WandbLogger(project=args.project_name, verbose=True, name=actual_run_name, config=config_dict)
    else:
        logger = DefaultLogger(verbose=True)

    # Step 5: Initialize Guide
    print(f"\nUsing Modal GPU evaluator (GPU: {args.gpu})...")
    guide = KernelGuide(
        gpu=args.gpu,
        verbose=args.verbose,
        num_correct_trials=1,
        num_perf_trials=5
    )

    # Step 6: Create Algorithm
    # Algorithm selection
    from opto.features.priority_search.priority_search_ablation import EpsilonNetPS 
    if args.algorithm == 'PS':
        algorithm = EpsilonNetPS(
            epsilon=0,
            use_summarizer=False,
            agent=agent,
            optimizer=optimizer,
            logger=logger,
            num_threads=num_threads,
        )
    elif args.algorithm == 'PS_Summarizer':
        algorithm = EpsilonNetPS(
            epsilon=0,
            use_summarizer=True,
            agent=agent,
            optimizer=optimizer,
            logger=logger,
            num_threads=num_threads
        )
    elif args.algorithm == 'PS_epsNet_Summarizer':
        algorithm = EpsilonNetPS(
            epsilon=args.epsilon,
            use_summarizer=True,
            agent=agent,
            optimizer=optimizer,
            logger=logger)
    elif args.algorithm == 'PS_epsNet':
        algorithm = EpsilonNetPS(
            agent=agent,
            optimizer=optimizer,
            use_summarizer=False,
            logger=logger,
            epsilon=args.epsilon,
            num_threads=num_threads
            )
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
    
    # Step 7: Run Training
    print(f"\nStarting PrioritySearch optimization (max {args.num_steps} steps)...")
    start_time = time.time()

    algorithm.train(
        guide=guide,
        train_dataset=task,
        validate_dataset=task,
        test_dataset=task,
        batch_size=1,
        num_batches=1,
        num_steps=args.num_steps,
        num_threads=args.num_threads,
        num_eval_samples=1,
        validate_exploration_candidates=False,
        use_best_candidate_to_explore=False,
        num_candidates=args.num_candidates,
        num_proposals=args.num_proposals,
        score_function='mean',
        log_frequency=args.log_frequency,
        test_frequency=args.test_frequency
    )

    duration = time.time() - start_time
    print(f"\nOptimization completed in {duration:.2f} seconds")


if __name__ == "__main__":
    main()