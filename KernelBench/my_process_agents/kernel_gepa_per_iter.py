# Optimize CUDA kernels using GEPA with per-iteration tracking
# This version saves detailed per-iteration results for analysis

import sys
import os
# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
np.random.seed(10)
torch.manual_seed(10)

import time
import argparse
import json
import re
import shutil
import pickle
from datetime import datetime

import gepa  # GEPA optimization library
from gepa.core.state import GEPAState
from gepa.utils.stop_condition import StopperProtocol

# Import dataset and evaluation utilities
from dataset.utils import create_matrix_multiplication_dataset
from guide.evaluate import evaluate
from opto.optimizers.utils import print_color

import litellm
litellm.drop_params = True
litellm.suppress_debug_info = True
litellm.num_retries = 3
litellm.request_timeout = 300


class MaxIterationsStopper(StopperProtocol):
    """Custom stopper that stops after a maximum number of iterations."""

    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations

    def __call__(self, state: GEPAState) -> bool:
        """Return True if we should stop (reached max iterations).
        
        Note: GEPA checks this BEFORE incrementing state.i and running the iteration.
        So for max_iterations=10:
        - state.i=0: check 0>=10? No, run iteration 1
        - state.i=9: check 9>=10? No, run iteration 10
        - state.i=10: check 10>=10? Yes, STOP
        """
        return state.i >= self.max_iterations


# Minimal reflection prompt - only task description as context
REFLECTION_PROMPT_TEMPLATE = """<task_description>

Current CUDA kernel code:
<curr_instructions>

Evaluation feedback:
<inputs_outputs_feedback>

Generate an improved CUDA kernel code."""


class KernelAdapter(gepa.GEPAAdapter):
    """
    Adapter for optimizing CUDA kernel code to maximize speedup.

    The candidate is a dict: {"cuda_code": "<cuda code string>"}
    The objective is to maximize speedup (higher score = better).
    """

    def __init__(self, task_description: str, ref_arch_src: str, num_correct_trials: int = 1, num_perf_trials: int = 5):
        """
        Initialize the kernel adapter.

        Args:
            task_description: Task description from dataset
            ref_arch_src: Reference PyTorch implementation
            num_correct_trials: Number of correctness validation trials
            num_perf_trials: Number of performance measurement trials
        """
        self.task_description = task_description
        self.ref_arch_src = ref_arch_src
        self.num_correct_trials = num_correct_trials
        self.num_perf_trials = num_perf_trials

    def evaluate(self, batch, candidate, capture_traces=False):
        """
        Evaluate the candidate CUDA kernel.

        For kernel optimization, we only have one task, so batch size is always 1.

        Args:
            batch: List with one example (the task)
            candidate: {"cuda_code": "<cuda code string>"}
            capture_traces: Whether to capture trajectories for reflection

        Returns:
            EvaluationBatch with scores and feedback
        """
        cuda_code = candidate["cuda_code"]

        # Strip markdown code fences if LLM generated them
        cuda_code = self._strip_markdown_fences(cuda_code)

        # Evaluate the CUDA kernel
        try:
            score, feedback = evaluate(
                ref_arch_src=self.ref_arch_src,
                custom_cuda=cuda_code,
                num_correct_trials=self.num_correct_trials,
                num_perf_trials=self.num_perf_trials
            )

            # Debug output
            print_color("\n" + "=" * 70, 'blue')
            print_color("CUDA KERNEL:", 'cyan')
            print_color(cuda_code[:500] + ("..." if len(cuda_code) > 500 else ""), 'white')
            print_color(f"\nScore (Speedup): {score:.4f}", 'green' if score > 0 else 'red')
            print_color(f"Feedback: {feedback[:300]}{('...' if len(feedback) > 300 else '')}", 'yellow')
            print_color("=" * 70 + "\n", 'blue')

        except Exception as e:
            score = 0.0
            feedback = f"Error during evaluation: {str(e)}"
            print_color(f"\nEvaluation error: {e}", 'red')

        # Prepare outputs
        outputs = [{
            "cuda_code": cuda_code,
            "score": score,
            "feedback": feedback
        }]

        scores = [score]

        # Capture trajectory if requested (needed for reflection)
        trajectories = None
        if capture_traces:
            trajectories = [{
                "task_description": self.task_description,
                "cuda_code": cuda_code,
                "score": score,
                "feedback": feedback
            }]

        return gepa.EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories
        )

    def _strip_markdown_fences(self, code: str) -> str:
        """Strip markdown code fences and language identifiers from LLM output."""
        # Remove opening fence with optional language identifier (```python, ```cuda, ```cpp, etc.)
        code = re.sub(r'^```[a-zA-Z0-9+]*\n?', '', code.strip())
        # Remove standalone language identifier at the beginning
        code = re.sub(r'^(python|cuda|cpp|c\+\+|c)\s*\n', '', code, flags=re.IGNORECASE)
        # Remove closing fence
        code = re.sub(r'\n?```\s*$', '', code)
        return code.strip()

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        """
        Build reflective dataset for GEPA to propose improved CUDA kernel.

        This extracts the evaluation feedback so GEPA can reflect on performance
        and propose improvements.

        Args:
            candidate: {"cuda_code": "<current cuda code>"}
            eval_batch: Results from evaluate() with trajectories
            components_to_update: ["cuda_code"]

        Returns:
            {"cuda_code": [examples with inputs, outputs, and feedback]}
        """
        reflective_data = {}

        if "cuda_code" in components_to_update:
            examples = []

            # GEPA expects examples with Inputs, Generated Outputs, and Feedback
            for traj in eval_batch.trajectories:
                example = {
                    "Inputs": {
                        "Task Description": traj["task_description"]
                    },
                    "Generated Outputs": {
                        "Current CUDA Code": traj["cuda_code"]
                    },
                    "Feedback": traj["feedback"]
                }
                examples.append(example)

            reflective_data["cuda_code"] = examples

        return reflective_data


def process_iteration_trace(iteration_trace, gepa_result, initial_seed_score):
    """
    Process the full iteration trace to extract per-iteration statistics.

    For single-task optimization, tracks the best score achieved and cumulative metric calls
    at each iteration.

    Args:
        iteration_trace: List of iteration dictionaries from gepa_state.full_program_trace
        gepa_result: GEPA result with candidate discovery information
        initial_seed_score: Score of the initial seed program on the task

    Returns:
        List of dictionaries, each containing:
        - iteration: Iteration number (starts from 1)
        - best_score_so_far: Best score achieved on the task up to this iteration
        - num_metric_calls_so_far: Total metric calls (evaluations) used so far
        - calls_this_iter: Number of metric calls in this iteration only
        - accepted_new_candidate: Whether a new candidate was accepted this iteration
        - new_candidate_score: Validation score of new candidate if accepted, None otherwise
    """
    detailed_history = []

    # Initialize with seed evaluation
    # The seed is evaluated on the full validation set before iteration 0
    num_val_tasks = len(gepa_result.val_subscores[0]) if gepa_result.val_subscores else 1
    cumulative_metric_calls = num_val_tasks  # Initial seed evaluation
    best_score_so_far = initial_seed_score  # Start with seed score

    # If no iterations were run (e.g., seed was perfect), create a synthetic iteration 1 for the seed
    if not iteration_trace:
        detailed_history.append({
            "iteration": 1,
            "best_score_so_far": initial_seed_score,
            "num_metric_calls_so_far": num_val_tasks,
            "calls_this_iter": num_val_tasks,
            "accepted_new_candidate": True,  # Seed is always "accepted"
            "new_candidate_score": initial_seed_score,
        })
        return detailed_history

    for iter_dict in iteration_trace:
        iteration_num = iter_dict.get("i", -1) + 1  # Convert to 1-indexed

        # Count metric calls in THIS iteration
        calls_this_iter = 0

        # 1. Parent evaluation on subsample
        subsample_ids = iter_dict.get("subsample_ids", [])
        if subsample_ids:
            calls_this_iter += len(subsample_ids)

        # 2. New candidate evaluation on subsample (if proposal generated)
        new_subsample_scores = iter_dict.get("new_subsample_scores", [])
        if new_subsample_scores:
            calls_this_iter += len(subsample_ids)

        # 3. Merge evaluation (if merge was invoked)
        if iter_dict.get("invoked_merge", False):
            # Merge requires evaluation of merged candidate
            calls_this_iter += len(subsample_ids)

        # 4. Full validation evaluation (if candidate was accepted)
        evaluated_val_indices = iter_dict.get("evaluated_val_indices", [])
        if evaluated_val_indices:
            calls_this_iter += len(evaluated_val_indices)

        # Update cumulative count
        cumulative_metric_calls += calls_this_iter

        # Track best score - check both subsample and validation scores
        # For single-task optimization, we want the maximum score seen on the task

        # Check subsample scores from this iteration
        subsample_scores = iter_dict.get("subsample_scores", [])
        all_scores_this_iter = list(subsample_scores) + list(new_subsample_scores)
        if all_scores_this_iter:
            iter_max_score = max(all_scores_this_iter)
            best_score_so_far = max(best_score_so_far, iter_max_score)

        # Check if new candidate was accepted (has validation score)
        new_program_idx = iter_dict.get("new_program_idx")
        new_candidate_score = None
        accepted_new_candidate = False

        if new_program_idx is not None:
            accepted_new_candidate = True
            # Get validation score for accepted candidate
            if new_program_idx < len(gepa_result.val_aggregate_scores):
                new_candidate_score = gepa_result.val_aggregate_scores[new_program_idx]
                # Update best score with validation score
                best_score_so_far = max(best_score_so_far, new_candidate_score)

        # Build iteration info
        iter_info = {
            "iteration": iteration_num,
            "best_score_so_far": best_score_so_far,
            "num_metric_calls_so_far": cumulative_metric_calls,
            "calls_this_iter": calls_this_iter,
            "accepted_new_candidate": accepted_new_candidate,
            "new_candidate_score": new_candidate_score,
        }

        detailed_history.append(iter_info)

    return detailed_history


def main():
    parser = argparse.ArgumentParser(description='Optimize CUDA kernel using GEPA with per-iteration tracking')

    # Task parameters
    parser.add_argument('--task_idx', type=int, default=0,
                       help='Task index from KernelBench dataset (0-15)')

    # Model parameters
    parser.add_argument('--model', type=str, default='claude-3.7-sonnet',
                       help='Reflection LM model name (e.g., claude-3.7-sonnet, gpt-4o)')

    # Optimization parameters - use max_iterations for exact iteration control
    parser.add_argument('--max_iterations', type=int, default=10,
                       help='Maximum number of optimization iterations (default: 10)')
    parser.add_argument('--num_correct_trials', type=int, default=1,
                       help='Number of correctness trials per evaluation')
    parser.add_argument('--num_perf_trials', type=int, default=5,
                       help='Number of performance trials per evaluation')

    # Logging parameters
    parser.add_argument('--save_results', action='store_true', default=True,
                       help='Save full results including iteration history (default: True)')
    parser.add_argument('--log_dir', type=str, default='results/kernel_gepa_logs',
                       help='Directory for GEPA logs and checkpoints')

    args = parser.parse_args()

    # Load task from dataset
    print(f"\nLoading task {args.task_idx} from KernelBench dataset...")
    ds = create_matrix_multiplication_dataset()
    if args.task_idx >= len(ds):
        raise ValueError(f"Task index {args.task_idx} out of range (0-{len(ds)-1})")

    task_description = ds[args.task_idx]['input']
    ref_arch_src = ds[args.task_idx]['ref_arch_src']

    print(f"Task loaded successfully. Task ID: {args.task_idx}")
    print("=" * 70)

    # Create dataset for GEPA (single task)
    trainset = [{"task_id": args.task_idx}]
    valset = trainset  # Same task for validation

    # Create adapter
    print("Creating KernelAdapter...")
    adapter = KernelAdapter(
        task_description=task_description,
        ref_arch_src=ref_arch_src,
        num_correct_trials=args.num_correct_trials,
        num_perf_trials=args.num_perf_trials
    )

    # Initial seed candidate
    seed_candidate = {
        "cuda_code": "# This is a dummy kernel code. You should replace it with your own kernel code based on the task prompt and optimization objectives."
    }

    print(f"\nInitial seed CUDA code: '{seed_candidate['cuda_code']}'")

    # Prepare reflection_lm parameter (add provider if needed)
    reflection_lm = args.model
    if "/" not in reflection_lm:
        # Add provider prefix based on model name
        if "claude" in reflection_lm.lower():
            reflection_lm = f"anthropic/{reflection_lm}"
        elif "gpt" in reflection_lm.lower():
            reflection_lm = f"openai/{reflection_lm}"
        elif "gemini" in reflection_lm.lower():
            reflection_lm = f"gemini/{reflection_lm}"

    # Prepare reflection prompt with task description
    reflection_prompt = REFLECTION_PROMPT_TEMPLATE.replace("<task_description>", task_description)

    # Use MaxIterationsStopper for exact iteration control
    stop_callback = MaxIterationsStopper(args.max_iterations)

    # Create unique log directory with timestamp to prevent resuming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:21]
    log_dir = f"{args.log_dir}_task_{args.task_idx}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"Fresh log directory (no resume): {log_dir}")

    # Run GEPA optimization
    print(f"\nStarting GEPA optimization:")
    print(f"  Reflection LM: {reflection_lm}")
    print(f"  Max iterations: {args.max_iterations} (exact)")
    print(f"  Correctness trials: {args.num_correct_trials}")
    print(f"  Performance trials: {args.num_perf_trials}")
    print(f"  Objective: Maximize speedup")
    print(f"  Per-iteration tracking: ENABLED")
    print(f"  Early stopping: DISABLED (will run all {args.max_iterations} iterations)")
    print()

    start_time = time.time()

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,

        # Reflection configuration
        reflection_lm=reflection_lm,
        reflection_prompt_template=reflection_prompt,
        candidate_selection_strategy='pareto',
        reflection_minibatch_size=1,  # Single task
        skip_perfect_score=False,  # Don't stop early

        # Component selection
        module_selector='all',

        # Budget - use max_iterations via stop_callback
        max_metric_calls=None,  # Don't use max_metric_calls
        stop_callbacks=stop_callback,

        # Logging
        run_dir=log_dir,
        track_best_outputs=True,
        display_progress_bar=True,

        # No wandb
        use_wandb=False,

        # Reproducibility
        seed=10,
    )

    duration = time.time() - start_time

    # Extract best result
    best_candidate = result.best_candidate
    best_speedup = result.val_aggregate_scores[result.best_idx]
    initial_seed_score = result.val_aggregate_scores[0] if result.val_aggregate_scores else 0.0

    # Find when best speedup was achieved
    best_at = None
    if result.discovery_eval_counts:
        best_at = result.discovery_eval_counts[result.best_idx]

    # Load detailed iteration trace from saved state
    iteration_trace = None
    state_path = os.path.join(log_dir, "gepa_state.bin")
    if os.path.exists(state_path):
        try:
            with open(state_path, "rb") as f:
                state_data = pickle.load(f)
                iteration_trace = state_data.get("full_program_trace", [])
                print(f"\nLoaded detailed iteration trace with {len(iteration_trace)} iterations")
        except Exception as e:
            print(f"Warning: Could not load iteration trace: {e}")
            iteration_trace = None

    # Process iteration trace to get detailed history
    detailed_iteration_history = process_iteration_trace(
        iteration_trace if iteration_trace else [],
        result,
        initial_seed_score
    )

    num_iterations_total = len(detailed_iteration_history)
    final_best_score = detailed_iteration_history[-1]['best_score_so_far'] if detailed_iteration_history else best_speedup

    # Always save summary with per-iteration history
    os.makedirs('results/kernel_gepa', exist_ok=True)

    summary_path = f"results/kernel_gepa/gepa_task_{args.task_idx}_summary.json"
    summary_data = {
        'task_idx': args.task_idx,
        'best_speedup': final_best_score,
        'num_metric_calls': result.total_metric_calls,
        'best_at_metric_call': best_at,
        'duration_seconds': duration,
        'model': reflection_lm,
        'method': 'gepa',
        'num_correct_trials': args.num_correct_trials,
        'num_perf_trials': args.num_perf_trials,
        'num_iterations_total': num_iterations_total,
        'history': detailed_iteration_history,  # Per-iteration tracking
    }

    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nSaved summary with per-iteration history to {summary_path}")

    # Full result with all details
    if args.save_results:
        save_path = f"results/kernel_gepa/gepa_task_{args.task_idx}_result.json"

        result_data = {
            'task_idx': args.task_idx,
            'best_speedup': final_best_score,
            'num_metric_calls': result.total_metric_calls,
            'num_iterations_total': num_iterations_total,
            'num_candidates_discovered': len(result.val_aggregate_scores),
            'best_at_metric_call': best_at,
            'best_cuda_code': best_candidate['cuda_code'],
            'duration_seconds': duration,
            'initial_seed_score': initial_seed_score,
            'iteration_history': detailed_iteration_history,  # Per-iteration details
            'val_aggregate_scores': result.val_aggregate_scores,
            'settings': {
                'reflection_lm': reflection_lm,
                'max_iterations': args.max_iterations,
                'reflection_minibatch_size': 1,
                'num_correct_trials': args.num_correct_trials,
                'num_perf_trials': args.num_perf_trials,
            }
        }

        with open(save_path, 'w') as f:
            json.dump(result_data, f, indent=2)

        print(f"Saved full result to {save_path}")

    # Print per-iteration breakdown
    print(f"\nIteration Statistics:")
    print(f"  Total iterations: {num_iterations_total}")
    print(f"  Successful candidates: {len(result.val_aggregate_scores)}")
    print(f"  Total metric calls used: {result.total_metric_calls}")
    print(f"  Final best score: {final_best_score}")
    print(f"\nPer-iteration breakdown:")
    for iter_info in detailed_iteration_history:
        accepted_str = "✓" if iter_info['accepted_new_candidate'] else "✗"
        score_str = f"score={iter_info['new_candidate_score']:.4f}" if iter_info['new_candidate_score'] is not None else "no new candidate"
        print(f"  Iter {iter_info['iteration']:2d}: "
              f"best={iter_info['best_score_so_far']:.4f}, "
              f"calls_so_far={iter_info['num_metric_calls_so_far']:3d}, "
              f"accepted={accepted_str} ({score_str})")

    # Clean one-line summary
    print("\n" + "=" * 70)
    print(f"Task {args.task_idx} | Duration: {duration:.1f}s | Model: {reflection_lm.split('/')[-1]} | Method: GEPA")
    print("=" * 70)
    if final_best_score > 0:
        print(f"✓ COMPLETED: Best speedup = {final_best_score:.4f}x")
        if best_at:
            print(f"  Achieved at metric call {best_at}/{result.total_metric_calls}")
    else:
        print(f"✗ FAILED: No valid kernel achieved positive speedup")
        print(f"  Max iterations ({num_iterations_total}) reached")
    print("=" * 70)


if __name__ == "__main__":
    main()
