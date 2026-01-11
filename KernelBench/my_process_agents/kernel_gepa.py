# Optimize CUDA kernels using GEPA (Generalized Evolutionary Prompting Algorithm)
# GEPA iteratively evolves CUDA kernel code to maximize speedup through reflection on evaluation feedback

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

import gepa  # GEPA optimization library
from gepa.core.state import GEPAState

# Import dataset and evaluation utilities
from dataset.utils import create_matrix_multiplication_dataset
from guide.evaluate import evaluate
from opto.optimizers.utils import print_color


# Custom stopper to limit GEPA iterations
class MaxIterationsStopper:
    """Stop GEPA after a maximum number of proposal iterations."""

    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations

    def __call__(self, gepa_state: GEPAState) -> bool:
        """Return True if max iterations reached."""
        return gepa_state.i >= self.max_iterations

import litellm
litellm.drop_params = True
litellm.suppress_debug_info = True
litellm.num_retries = 3
litellm.request_timeout = 300


# Minimal reflection prompt - only task description as context
# GEPA requires exactly these placeholders: <curr_instructions> and <inputs_outputs_feedback>
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


def main():
    parser = argparse.ArgumentParser(description='Optimize CUDA kernel using GEPA')

    # Task parameters
    parser.add_argument('--task_idx', type=int, default=0,
                       help='Task index from KernelBench dataset (0-15)')

    # Model parameters
    parser.add_argument('--model', type=str, default='claude-3.7-sonnet',
                       help='Reflection LM model name (e.g., claude-3.7-sonnet, gpt-4o)')

    # Optimization parameters
    parser.add_argument('--max_iterations', type=int, default=None,
                       help='Maximum number of proposal iterations (default: None, use max_metric_calls)')
    parser.add_argument('--max_metric_calls', type=int, default=10,
                       help='Budget for GEPA (number of evaluations, ignored if max_iterations is set)')
    parser.add_argument('--num_correct_trials', type=int, default=1,
                       help='Number of correctness trials per evaluation')
    parser.add_argument('--num_perf_trials', type=int, default=5,
                       help='Number of performance trials per evaluation')

    # Logging parameters
    parser.add_argument('--save_results', action='store_true', default=False,
                       help='Save full results including all candidates (default: only summary)')
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

    # Initial seed candidate - same as kernel_PS and kernel_openevolve
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

    # Prepare reflection prompt with task description (only context)
    reflection_prompt = REFLECTION_PROMPT_TEMPLATE.replace("<task_description>", task_description)

    # Prepare stopping condition based on max_iterations or max_metric_calls
    if args.max_iterations is not None:
        stop_callback = MaxIterationsStopper(args.max_iterations)
        max_metric_calls = None  # Don't use max_metric_calls when using iteration stopper
        budget_desc = f"Max iterations: {args.max_iterations}"
    else:
        stop_callback = None
        max_metric_calls = args.max_metric_calls
        budget_desc = f"Max metric calls: {max_metric_calls}"

    # Run GEPA optimization
    print(f"\nStarting GEPA optimization:")
    print(f"  Reflection LM: {reflection_lm}")
    print(f"  Budget: {budget_desc}")
    print(f"  Correctness trials: {args.num_correct_trials}")
    print(f"  Performance trials: {args.num_perf_trials}")
    print(f"  Objective: Maximize speedup")
    print(f"  Save results: {args.save_results}")
    if args.save_results:
        print(f"  Log directory: {args.log_dir}")
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
        skip_perfect_score=False,  # Don't stop early, use full budget

        # Component selection
        module_selector='all',

        # Budget - use either max_iterations (via stop_callback) or max_metric_calls
        max_metric_calls=max_metric_calls,
        stop_callbacks=stop_callback,

        # Logging
        run_dir=args.log_dir if args.save_results else None,
        log_frequency=1,
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

    # Find when best speedup was achieved
    best_at = None
    if result.discovery_eval_counts:
        best_at = result.discovery_eval_counts[result.best_idx]

    # Always save summary (similar to openevolve)
    os.makedirs('results/kernel_gepa', exist_ok=True)

    summary_path = f"results/kernel_gepa/gepa_task_{args.task_idx}_summary.json"
    summary_data = {
        'task_idx': args.task_idx,
        'best_speedup': best_speedup,
        'num_metric_calls': result.total_metric_calls,
        'best_at_metric_call': best_at,
        'duration_seconds': duration,
        'model': reflection_lm,
        'method': 'gepa',
        'num_correct_trials': args.num_correct_trials,
        'num_perf_trials': args.num_perf_trials,
    }

    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nSaved summary to {summary_path}")

    # Full result with all candidates saved only if requested (similar to openevolve)
    if args.save_results:
        save_path = f"results/kernel_gepa/gepa_task_{args.task_idx}_result.json"

        result_data = {
            'task_idx': args.task_idx,
            'best_speedup': best_speedup,
            'num_metric_calls': result.total_metric_calls,
            'best_at_metric_call': best_at,
            'best_cuda_code': best_candidate['cuda_code'],
            'num_candidates': len(result.candidates),
            'duration_seconds': duration,
            'val_aggregate_scores': result.val_aggregate_scores,
            'settings': {
                'reflection_lm': reflection_lm,
                'max_metric_calls': args.max_metric_calls,
                'reflection_minibatch_size': 1,
                'num_correct_trials': args.num_correct_trials,
                'num_perf_trials': args.num_perf_trials,
            }
        }

        with open(save_path, 'w') as f:
            json.dump(result_data, f, indent=2)

        print(f"Saved full result to {save_path}")

    # Clean one-line summary (similar to openevolve)
    print("\n" + "=" * 70)
    print(f"Task {args.task_idx} | Duration: {duration:.1f}s | Model: {reflection_lm.split('/')[-1]} | Method: GEPA")
    print("=" * 70)
    if best_speedup > 0:
        print(f"✓ COMPLETED: Best speedup = {best_speedup:.4f}x")
        if best_at:
            print(f"  Achieved at metric call {best_at}/{result.total_metric_calls}")
    else:
        print(f"✗ FAILED: No valid kernel achieved positive speedup")
        print(f"  Max metric calls ({result.total_metric_calls}) reached")
    print("=" * 70)


if __name__ == "__main__":
    main()
