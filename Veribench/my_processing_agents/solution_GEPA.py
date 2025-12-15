# Optimize a single Lean solution using GEPA (standalone)
# The Lean code is optimized as a text string based on compilation feedback

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

import gepa  # Use standalone GEPA

from my_processing_agents.solution_opt import extract_python_code, load_single_task
from guide.guide import VeribenchGuide

import litellm
litellm.drop_params = True
litellm.suppress_debug_info = True

# Add retry logic for rate limiting errors
from litellm import completion_with_retries
litellm.num_retries = 3  # Retry up to 3 times
litellm.request_timeout = 300  # 5 minute timeout per request

from my_processing_agents import secrets_local  # Load environment variables
from my_processing_agents.system_prompts import SYSTEM_PROMPT_WITH_EXAMPLES

# Custom reflection prompt for GEPA to generate actual Lean code, not meta-instructions
CUSTOM_REFLECTION_PROMPT = f"""Your task is to produce valid Lean 4 code that correctly translates a Python program.

You will see:
- The original Python program (in the inputs below)
- The current Lean 4 code attempt (in Current Lean 4 code below)
- Compilation feedback: either "success" or error messages from the Lean 4 compiler

Goal: Produce Lean 4 code that compiles successfully (score = 1).
- If the current Lean code is a dummy placeholder (e.g., just a comment), generate a complete Lean 4 translation from the Python program.
- Otherwise, fix the compilation errors in the current Lean code while preserving correct parts.

Key rules:
- Preserve the algorithm logic from the Python program
- Use correct Lean 4 syntax and type annotations
- Output only valid Lean 4 code (NOT meta-instructions or guidelines)

Original System and User Prompts which may be helpful:

{SYSTEM_PROMPT_WITH_EXAMPLES}

The translated Lean 4 code should be a faithful representation of the Python code.
It should be correct and compiles.
If a theorem keeps giving error, you can use := sorry to skip it.

---

Current Lean 4 code:
```lean
<curr_instructions>
```

Compilation results and feedback from testing this code:
```
<inputs_outputs_feedback>
```

YOUR TASK: Generate the COMPLETE, WORKING Lean 4 code that fixes all compilation errors.

CRITICAL: You must output ACTUAL LEAN 4 CODE, not instructions or guidelines about writing code.

Provide your complete Lean 4 code within ``` blocks."""

class VeribenchAdapter(gepa.GEPAAdapter):
    """
    Adapter for optimizing Lean4 code to correctly compile.
    
    The candidate is: {"lean_code": "<lean4 code string>"}
    The goal is to make the Lean code compile correctly (score = 1.0).
    """
    
    def __init__(self, guide: VeribenchGuide):
        self.guide = guide
    
    def evaluate(self, batch, candidate, capture_traces=False):
        """
        Evaluate the candidate Lean code on a batch of Python programs.
        
        Args:
            batch: List of examples with Python code
            candidate: {"lean_code": "<lean4 code string>"}
            capture_traces: Whether to capture execution traces for reflection
            
        Returns:
            EvaluationBatch with scores and feedback
        """
        outputs = []
        scores = []
        trajectories = []
        
        lean_code = candidate["lean_code"]
        
        for example in batch:
            python_program = example["python_program"]
            
            # Evaluate the Lean code using VeribenchGuide
            try:
                score, feedback = self.guide.get_feedback(
                    task=python_program,
                    response=lean_code
                )
            except Exception as e:
                score = 0.0
                feedback = f"Error during evaluation: {str(e)}"
            
            outputs.append({"lean_code": lean_code, "feedback": feedback})
            scores.append(score)
            
            # Capture trajectory if requested (needed for reflection)
            if capture_traces:
                trajectory = {
                    "python_program": python_program,
                    "lean_code": lean_code,
                    "score": score,
                    "feedback": feedback
                }
                trajectories.append(trajectory)
        
        return gepa.EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None
        )
    
    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        """
        Build reflective dataset for GEPA to propose improved Lean code.
        
        This extracts the failed examples with their compilation errors
        so GEPA can reflect on what went wrong and propose fixes.
        
        Args:
            candidate: {"lean_code": "<current lean code>"}
            eval_batch: Results from evaluate() with trajectories
            components_to_update: ["lean_code"]
            
        Returns:
            {"lean_code": [list of examples with inputs, outputs, and feedback]}
        """
        reflective_data = {}
        
        if "lean_code" in components_to_update:
            examples = []
            
            for i, (traj, score) in enumerate(zip(eval_batch.trajectories, eval_batch.scores)):
                # Include all examples (GEPA will learn from both successes and failures)
                # But focus on failures by only including them
                if score < 1.0:
                    example = {
                        "Inputs": {
                            "Python Program": traj["python_program"]
                        },
                        "Generated Outputs": {
                            "Current Lean Code": traj["lean_code"]
                        },
                        "Feedback": traj["feedback"]
                    }
                    examples.append(example)
            
            reflective_data["lean_code"] = examples
        
        return reflective_data


def create_single_task_dataset(task_idx: int):
    """
    Create a dataset with a single task for GEPA optimization.
    
    Args:
        task_idx: Index of the task to load
        
    Returns:
        List with one example dict containing Python program
    """
    task = load_single_task(task_idx)
    python_code = extract_python_code(task['user_query'])
    
    return [
        {
            "python_program": python_code,
            "task_id": task_idx,
            "user_query": task['user_query']
        }
    ]


def main():
    parser = argparse.ArgumentParser(description='Optimize Lean solution using GEPA')
    parser.add_argument('--task_idx', type=int, default=2, 
                       help='Task index from Veribench dataset')
    parser.add_argument('--model', type=str, default='claude-3.5-sonnet', 
                       help='Reflection LM model name (e.g., claude-3.5-sonnet, gpt-4o)')
    parser.add_argument('--max_metric_calls', type=int, default=50, 
                       help='Budget for GEPA (number of evaluations). Reduce if getting rate limit errors.')
    parser.add_argument('--reflection_minibatch_size', type=int, default=1, 
                       help='Minibatch size for reflection (default: 1 for single task)')
    
    # Logging parameters
    parser.add_argument('--use_wandb', action='store_true', default=False,
                       help='Use Weights & Biases for logging')
    parser.add_argument('--project', type=str, default='veribench-gepa',
                       help='WandB project name')
    parser.add_argument('--run_name', type=str, default=None,
                       help='WandB run name (default: gepa_task_{task_idx})')
    parser.add_argument('--log_dir', type=str, default='results/gepa_logs',
                       help='Directory for GEPA logs and snapshots')
    parser.add_argument('--log_frequency', type=int, default=1,
                       help='Save snapshots every N iterations')
    parser.add_argument('--save_results', action='store_true', default=False,
                       help='Save results to JSON file (default: False)')
    
    args = parser.parse_args()
    
    # Load task
    print(f"\nLoading task {args.task_idx} from Veribench dataset...")
    task = load_single_task(args.task_idx)
    python_program = extract_python_code(task['user_query'])
    print(f"Task loaded successfully. Task ID: {task['task_id']}")
    print(f"\nPython code to translate (first 500 chars):")
    print("=" * 70)
    print(python_program[:500] + ("..." if len(python_program) > 500 else ""))
    print("=" * 70)
    
    # Initialize Guide
    print("\nInitializing VeribenchGuide...")
    guide = VeribenchGuide()
    
    # Create dataset
    print("Creating dataset...")
    trainset = create_single_task_dataset(args.task_idx)
    valset = trainset  # Same task for validation
    
    # Create adapter
    print("Creating VeribenchAdapter...")
    adapter = VeribenchAdapter(guide=guide)
    
    # Initial seed candidate - same as solution_PS.py line 49
    seed_candidate = {
        "lean_code": "-- Lean 4 translation of the Python program"
    }
    
    print(f"\nInitial seed Lean code: '{seed_candidate['lean_code']}'")
    
    # Set run name
    run_name = args.run_name if args.run_name else f"gepa_task_{args.task_idx}"
    
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
    
    # Run GEPA optimization
    print(f"\nStarting GEPA optimization:")
    print(f"  Reflection LM: {reflection_lm}")
    print(f"  Max metric calls: {args.max_metric_calls}")
    print(f"  Target: Achieve score 1.0 (successful Lean compilation)")
    print(f"  Save results: {args.save_results}")
    if args.save_results:
        print(f"  Log directory: {args.log_dir}")
    if args.use_wandb:
        print(f"  WandB logging: project='{args.project}', run='{run_name}'")
    print()
    
    start_time = time.time()
    
    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        
        # Reflection configuration
        reflection_lm=reflection_lm,
        reflection_prompt_template=CUSTOM_REFLECTION_PROMPT,  # Use custom prompt for code generation
        candidate_selection_strategy='pareto',
        reflection_minibatch_size=args.reflection_minibatch_size,
        skip_perfect_score=True,
        perfect_score=1.0,
        
        # Component selection - optimize all components (just lean_code)
        module_selector='all',
        
        # Budget
        max_metric_calls=args.max_metric_calls,
        
        # Logging - only save checkpoints if save_results is enabled
        run_dir=args.log_dir if args.save_results else None,
        log_frequency=args.log_frequency,
        track_best_outputs=True,
        display_progress_bar=True,
        use_wandb=args.use_wandb,
        wandb_api_key=os.environ.get('WANDB_API_KEY') if args.use_wandb else None,
        wandb_init_kwargs={'project': args.project, 'name': run_name} if args.use_wandb else None,
        
        # Reproducibility
        seed=10,
    )
    
    duration = time.time() - start_time
    
    # Get best candidate
    best_candidate = result.best_candidate
    best_score = result.val_aggregate_scores[result.best_idx]
    
    # Always save summary, optionally save full result
    os.makedirs('results', exist_ok=True)
    
    # Get success metric call if successful
    success_at = None
    if best_score >= 1.0:
        first_success_idx = next((i for i, s in enumerate(result.val_aggregate_scores) if s >= 1.0), None)
        if first_success_idx is not None and result.discovery_eval_counts:
            success_at = result.discovery_eval_counts[first_success_idx]
    
    # Minimal summary saved always
    summary_path = f"results/gepa_task_{args.task_idx}_summary.json"
    summary_data = {
        'task_idx': args.task_idx,
        'task_id': task['task_id'],
        'success': best_score >= 1.0,
        'final_score': best_score,
        'num_metric_calls': result.total_metric_calls,
        'success_at_metric_call': success_at,
        'duration_seconds': duration,
        'model': reflection_lm,
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nSaved summary to {summary_path}")
    
    # Full result with all candidates saved only if requested
    if args.save_results:
        save_path = f"results/gepa_task_{args.task_idx}_result.json"
        
        result_data = {
            'task_idx': args.task_idx,
            'task_id': task['task_id'],
            'success': best_score >= 1.0,
            'final_score': best_score,
            'num_metric_calls': result.total_metric_calls,
            'success_at_metric_call': success_at,
            'best_lean_code': best_candidate['lean_code'],
            'num_candidates': len(result.candidates),
            'duration_seconds': duration,
            'val_aggregate_scores': result.val_aggregate_scores,
            'settings': {
                'reflection_lm': reflection_lm,
                'max_metric_calls': args.max_metric_calls,
                'reflection_minibatch_size': args.reflection_minibatch_size,
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"Saved full result to {save_path}")
    
    # Clean one-line summary
    print("\n" + "=" * 70)
    print(f"Task {args.task_idx} | Duration: {duration:.1f}s | Model: {reflection_lm.split('/')[-1]}")
    print("=" * 70)
    if best_score >= 1.0:
        first_success_idx = next((i for i, s in enumerate(result.val_aggregate_scores) if s >= 1.0), None)
        if first_success_idx is not None and result.discovery_eval_counts:
            success_at = result.discovery_eval_counts[first_success_idx]
            print(f"✓ SUCCESS: Reached score 1.0 at metric call {success_at}/{result.total_metric_calls}")
        else:
            print(f"✓ SUCCESS: Final score = {best_score}")
    else:
        print(f"✗ FAILED: Max metric calls ({result.total_metric_calls}) reached without achieving score 1.0")
        print(f"  Best score achieved: {best_score}")
    print("=" * 70)


if __name__ == "__main__":
    main()

