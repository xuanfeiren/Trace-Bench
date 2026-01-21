# Use openevolve to optimize CUDA kernel code.

# Optimize CUDA kernels using OpenEvolve
# Uses evolutionary algorithm to discover high-performance CUDA implementations

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
import tempfile
import shutil
from pathlib import Path

from openevolve.api import run_evolution, EvolutionResult
from openevolve.config import Config, LLMConfig, DatabaseConfig, EvaluatorConfig, PromptConfig


import litellm
litellm.drop_params = True
litellm.suppress_debug_info = True

# Import dataset utilities and evaluation function
from dataset.utils import create_matrix_multiplication_dataset
from guide.evaluate import evaluate


def write_initial_cuda_kernel(temp_dir: str, task_description: str, task_idx: int) -> str:
    """
    Write initial CUDA kernel with evolution markers.

    Args:
        temp_dir: Temporary directory
        task_description: Task description for CUDA kernel optimization
        task_idx: Task index

    Returns:
        Path to the initial program file
    """
    # Create initial CUDA kernel with same placeholder as kernel_PS
    initial_content = """// EVOLVE-BLOCK-START
# This is a dummy kernel code. You should replace it with your own kernel code based on the task prompt and optimization objectives.
// EVOLVE-BLOCK-END
"""

    program_path = os.path.join(temp_dir, f"initial_task_{task_idx}.cu")
    with open(program_path, 'w') as f:
        f.write(initial_content)

    return program_path


def write_evaluator_wrapper(temp_dir: str, ref_arch_src: str, task_idx: int, num_correct_trials: int = 1, num_perf_trials: int = 5) -> str:
    """
    Write evaluator that wraps CUDA kernel evaluation.

    Args:
        temp_dir: Temporary directory
        ref_arch_src: Reference PyTorch implementation
        task_idx: Task index
        num_correct_trials: Number of correctness trials
        num_perf_trials: Number of performance trials

    Returns:
        Path to evaluator file
    """
    # Use repr() to properly escape the ref_arch_src for embedding
    ref_arch_src_repr = repr(ref_arch_src)

    evaluator_code = f'''"""
Evaluator for CUDA kernel optimization using KernelBench evaluation
"""
import sys
sys.path.insert(0, '{project_root}')

from guide.evaluate import evaluate
from openevolve.evaluation_result import EvaluationResult
from opto.optimizers.utils import print_color

# Reference PyTorch implementation (passed from kernel_openevolve.py)
REF_ARCH_SRC = {ref_arch_src_repr}

# Evaluation configuration
NUM_CORRECT_TRIALS = {num_correct_trials}
NUM_PERF_TRIALS = {num_perf_trials}


def evaluate(program_path):
    """
    Evaluate evolved CUDA kernel using KernelBench evaluation.

    Args:
        program_path: Path to the evolved CUDA kernel

    Returns:
        EvaluationResult with score (speedup) and performance feedback
    """
    try:
        # Read the evolved CUDA code
        with open(program_path, 'r') as f:
            cuda_code_full = f.read()

        # Extract only the evolved block (between markers)
        start_marker = "// EVOLVE-BLOCK-START"
        end_marker = "// EVOLVE-BLOCK-END"

        if start_marker in cuda_code_full and end_marker in cuda_code_full:
            cuda_code = cuda_code_full.split(start_marker)[1].split(end_marker)[0].strip()
        else:
            # Fallback: use entire file if markers not found
            cuda_code = cuda_code_full.strip()

        # Strip markdown code fences (```python, ```cuda, ```cpp, ```, etc.)
        import re
        # Remove opening fence with optional language identifier
        cuda_code = re.sub(r'^```[a-zA-Z]*\\n?', '', cuda_code)
        # Remove closing fence
        cuda_code = re.sub(r'\\n?```\\s*$', '', cuda_code)
        # Strip again after fence removal
        cuda_code = cuda_code.strip()

        # Handle empty code
        if not cuda_code or cuda_code == "":
            print_color("\\n" + "="*70, 'blue')
            print_color("CUDA KERNEL (Empty):", 'cyan')
            print_color("(No code generated)", 'red')
            print_color("\\nScore: 0.0", 'red')
            print_color("Feedback: Generated empty CUDA code", 'yellow')
            print_color("="*70 + "\\n", 'blue')
            return EvaluationResult(
                metrics={{"combined_score": 0.0}},
                artifacts={{
                    "stderr": "Generated empty CUDA code. Please generate complete CUDA kernel code, not empty output."
                }}
            )

        # Evaluate with KernelBench
        from guide.evaluate import evaluate as kernel_evaluate
        score, feedback = kernel_evaluate(
            ref_arch_src=REF_ARCH_SRC,
            custom_cuda=cuda_code,
            num_correct_trials=NUM_CORRECT_TRIALS,
            num_perf_trials=NUM_PERF_TRIALS
        )

        # Debug output: Print kernel, score, and feedback
        print_color("\\n" + "="*70, 'blue')
        print_color("CUDA KERNEL:", 'cyan')
        print_color(cuda_code[:500] + ("..." if len(cuda_code) > 500 else "") , 'white')
        print_color(f"\\nScore: {{score}}", 'green' if score > 0 else 'red')
        print_color(f"Feedback: {{feedback[:300]}}{{('...' if len(feedback) > 300 else '')}}", 'yellow')
        print_color("="*70 + "\\n", 'blue')

        # Pass through evaluation feedback completely unmodified
        return EvaluationResult(
            metrics={{
                "combined_score": float(score),
            }},
            artifacts={{
                "stderr": feedback,  # Use stderr for OpenEvolve compatibility
            }}
        )

    except Exception as e:
        # Handle evaluation errors
        import traceback
        error_msg = f"Evaluation error: {{str(e)}}\\n\\nTraceback:\\n{{traceback.format_exc()}}"
        print_color("\\n" + "="*70, 'blue')
        print_color("EVALUATION ERROR:", 'red')
        print_color(error_msg[:500], 'red')
        print_color("="*70 + "\\n", 'blue')
        return EvaluationResult(
            metrics={{"combined_score": 0.0}},
            artifacts={{
                "stderr": error_msg
            }}
        )
'''

    evaluator_path = os.path.join(temp_dir, f"evaluator_task_{task_idx}.py")
    with open(evaluator_path, 'w') as f:
        f.write(evaluator_code)

    return evaluator_path


def create_openevolve_config(model_name: str, max_iterations: int, task_description: str, num_workers: int = 4) -> Config:
    """
    Create OpenEvolve configuration for CUDA kernel optimization.

    Args:
        model_name: LLM model name
        max_iterations: Maximum evolution iterations
        task_description: Task description from dataset (task['inputs'][0])
        num_workers: Number of parallel workers for evaluation

    Returns:
        Config object for OpenEvolve
    """
    config = Config()
    
    # Set max iterations
    config.max_iterations = max_iterations
    
    # LLM configuration - determine API base and key based on model
    api_base = None
    api_key = None
    
    if "/" not in model_name:
        if "claude" in model_name.lower():
            api_base = os.environ.get('ANTHROPIC_API_BASE', 'https://api.anthropic.com/v1')
            # Ensure API base ends with /v1 for OpenAI-compatible endpoints
            if api_base and not api_base.endswith('/v1'):
                api_base = api_base.rstrip('/') + '/v1'
            api_key = os.environ.get('ANTHROPIC_API_KEY')
        elif "gpt" in model_name.lower():
            api_base = os.environ.get('OPENAI_API_BASE', 'https://api.openai.com/v1')
            api_key = os.environ.get('OPENAI_API_KEY')
        elif "gemini" in model_name.lower():
            api_base = 'https://generativelanguage.googleapis.com/v1beta/openai/'
            api_key = os.environ.get('OPENAI_API_KEY')  # Gemini uses OPENAI_API_KEY
        else:
            api_base = os.environ.get('OPENAI_API_BASE', 'https://api.openai.com/v1')
            api_key = os.environ.get('OPENAI_API_KEY')
    else:
        # Has provider prefix, extract model name
        provider, model_name = model_name.split('/', 1)
        if provider == 'anthropic':
            api_base = os.environ.get('ANTHROPIC_API_BASE', 'https://api.anthropic.com/v1')
            api_key = os.environ.get('ANTHROPIC_API_KEY')
        elif provider == 'openai':
            api_base = os.environ.get('OPENAI_API_BASE', 'https://api.openai.com/v1')
            api_key = os.environ.get('OPENAI_API_KEY')
        elif provider == 'gemini':
            api_base = 'https://generativelanguage.googleapis.com/v1beta/openai/'
            api_key = os.environ.get('OPENAI_API_KEY')
    
    # Create LLMConfig with primary_model set - this will auto-populate models list
    config.llm = LLMConfig(
        primary_model=model_name,
        api_base=api_base,
        api_key=api_key,
        temperature=0.7,
        max_tokens=8192,
        timeout=300,
        retries=10
        # Note: random_seed is not set here to prevent propagation to models
    )
    
    # Use task description directly as system message (same context as other methods)
    config.prompt = PromptConfig()
    config.prompt.system_message = task_description

    # Database configuration - optimized for single objective (speedup)
    config.database = DatabaseConfig()
    config.database.population_size = 50  # Population size for evolution
    config.database.num_islands = 1  # Single island for single task
    config.database.elite_selection_ratio = 0.2
    config.database.exploitation_ratio = 0.8  # High exploitation for optimization
    # Use built-in features for diversity
    # "complexity" = code length, "diversity" = structural difference
    config.database.feature_dimensions = ["complexity", "diversity"]

    # Evaluator configuration
    config.evaluator = EvaluatorConfig()
    config.evaluator.enable_artifacts = True  # CRITICAL for feedback loop!
    config.evaluator.timeout = 300  # Longer timeout for CUDA compilation/evaluation
    config.evaluator.parallel_evaluations = num_workers  # Number of parallel workers
    config.evaluator.cascade_evaluation = False  # Disable cascade

    # Evolution settings
    config.diff_based_evolution = False  # Full rewrites for better exploration
    config.file_suffix = ".cu"  # Use CUDA extension
    config.language = "cuda"
    config.random_seed = None  # Don't set master seed (prevents LLM seed propagation)
    config.database.random_seed = 10  # Reproducibility for sampling/database only
    config.checkpoint_interval = 10  # Save checkpoints

    # Early stopping configuration
    config.early_stopping_patience = None  # Disable early stopping (run all iterations)
    config.convergence_threshold = 0.001  # Any improvement resets counter
    config.early_stopping_metric = "combined_score"

    return config


def extract_evolution_history(output_dir: str) -> tuple[int, list]:
    """
    Extract evolution history from OpenEvolve output by reading all program files.
    
    Args:
        output_dir: OpenEvolve output directory
        
    Returns:
        Tuple of (num_metric_calls, history_list)
        history_list contains: [{'attempt': i, 'score': s, 'best_score': bs}, ...]
    """
    history = []
    num_metric_calls = 0
    
    # Try to load programs from the programs directory
    try:
        # OpenEvolve saves programs in checkpoint_XX/programs/ directories
        # Find the last checkpoint
        checkpoints_dir = os.path.join(output_dir, "checkpoints")
        if os.path.exists(checkpoints_dir):
            checkpoints = [d for d in os.listdir(checkpoints_dir) if d.startswith('checkpoint_')]
            if checkpoints:
                # Get the last checkpoint (highest iteration number)
                last_checkpoint_num = max([int(d.split('_')[1]) for d in checkpoints if len(d.split('_')) > 1 and d.split('_')[1].isdigit()])
                last_checkpoint_dir = os.path.join(checkpoints_dir, f"checkpoint_{last_checkpoint_num}")
                
                # Load all programs from this checkpoint
                programs_dir = os.path.join(last_checkpoint_dir, "programs")
                if os.path.exists(programs_dir):
                    all_programs = []
                    for program_file in os.listdir(programs_dir):
                        if program_file.endswith('.json'):
                            program_path = os.path.join(programs_dir, program_file)
                            try:
                                with open(program_path, 'r') as f:
                                    prog_data = json.load(f)
                                    all_programs.append(prog_data)
                            except Exception as e:
                                print(f"Warning: Could not load program {program_file}: {e}")
                    
                    # Sort by iteration_found to get chronological order
                    programs_sorted = sorted(all_programs, key=lambda p: p.get('iteration_found', 0))
                    num_metric_calls = len(programs_sorted)
                    
                    # Build history with best score tracking
                    best_score_so_far = 0.0
                    for i, prog in enumerate(programs_sorted, 1):
                        score = prog.get('metrics', {}).get('combined_score', 0.0)
                        is_new_best = score > best_score_so_far
                        if is_new_best:
                            best_score_so_far = score
                        
                        history.append({
                            'attempt': i,
                            'score': score,
                            'best_score': best_score_so_far,
                            'is_new_best': is_new_best,
                            'iteration_found': prog.get('iteration_found', 0)
                        })
                    
                    print(f"Extracted {len(history)} programs from OpenEvolve checkpoint (last iteration: {last_checkpoint_num})")
                    return num_metric_calls, history
                else:
                    print(f"Warning: Programs directory not found in {last_checkpoint_dir}")
    except Exception as e:
        print(f"Warning: Could not extract full history from programs directory: {e}")
    
    # Fallback: count checkpoint directories (approximation)
    try:
        checkpoints_dir = os.path.join(output_dir, "checkpoints")
        if os.path.exists(checkpoints_dir):
            checkpoints = [d for d in os.listdir(checkpoints_dir) if d.startswith('checkpoint_')]
            if checkpoints:
                last_checkpoint = max([int(d.split('_')[1]) for d in checkpoints if len(d.split('_')) > 1 and d.split('_')[1].isdigit()])
                num_metric_calls = last_checkpoint
                print(f"Using fallback: estimated {num_metric_calls} metric calls from checkpoint count")
    except:
        pass
    
    return num_metric_calls, history


def main():
    parser = argparse.ArgumentParser(description='Optimize CUDA kernel using OpenEvolve')
    parser.add_argument('--task_idx', type=int, default=0,
                       help='Task index from KernelBench dataset (0-15)')
    parser.add_argument('--model', type=str, default='claude-3.7-sonnet',
                       help='LLM model name (e.g., claude-3.5-sonnet, gpt-4o)')
    parser.add_argument('--max_iterations', type=int, default=50,
                       help='Maximum number of evolution iterations (default: 10)')
    parser.add_argument('--num_workers', type=int, default=5,
                       help='Number of parallel workers for evaluation (1=sequential, 4+=parallel)')
    parser.add_argument('--num_correct_trials', type=int, default=1,
                       help='Number of correctness trials (default: 1)')
    parser.add_argument('--num_perf_trials', type=int, default=5,
                       help='Number of performance trials (default: 5)')
    parser.add_argument('--verbose', action='store_true', default=False,
                       help='Print detailed logs during evolution')
    parser.add_argument('--run_num', type=int, default=1,
                       help='Run number (default: 1)')

    # Logging parameters
    parser.add_argument('--save_results', action='store_true', default=False,
                       help='Save full results (default: only summary)')

    args = parser.parse_args()

    # Load task from KernelBench dataset
    print(f"\nLoading task {args.task_idx} from KernelBench dataset...")
    ds = create_matrix_multiplication_dataset()
    if args.task_idx >= len(ds):
        raise ValueError(f"Task index {args.task_idx} out of range (0-{len(ds)-1})")

    task_description = ds[args.task_idx]['input']
    ref_arch_src = ds[args.task_idx]['ref_arch_src']

    print(f"Task loaded successfully. Task ID: {args.task_idx}")
    # print(f"\nTask description:")
    # print("=" * 70)
    # print(task_description)
    print("=" * 70)
    
    # Create temporary directory for OpenEvolve files
    temp_dir = tempfile.mkdtemp(prefix=f"openevolve_kernel_task_{args.task_idx}_")
    print(f"\nCreated temporary directory: {temp_dir}")

    try:
        # Write initial CUDA kernel
        print("Creating initial CUDA kernel...")
        initial_program_path = write_initial_cuda_kernel(temp_dir, task_description, args.task_idx)

        # Write evaluator wrapper
        print("Creating evaluator wrapper...")
        evaluator_path = write_evaluator_wrapper(temp_dir, ref_arch_src, args.task_idx,
                                                  args.num_correct_trials, args.num_perf_trials)

        # Create config
        print("Creating OpenEvolve configuration...")
        config = create_openevolve_config(args.model, args.max_iterations, task_description, args.num_workers)

        # Set output directory
        output_dir = os.path.join(temp_dir, "openevolve_output")

        # Run OpenEvolve
        print(f"\nStarting OpenEvolve evolution:")
        print(f"  Model: {args.model}")
        print(f"  Max iterations: {args.max_iterations}")
        print(f"  Parallel workers: {args.num_workers} (evaluations run in parallel)")
        print(f"  Objective: Maximize speedup (combined_score)")
        print(f"  Correctness trials: {args.num_correct_trials}")
        print(f"  Performance trials: {args.num_perf_trials}")
        print(f"  Evolution strategy: Full rewrites (not diff-based)")
        print(f"  Artifact feedback: ENABLED (KernelBench evaluation feedback passed to LLM)")
        print()
        
        start_time = time.time()
        
        result: EvolutionResult = run_evolution(
            initial_program=initial_program_path,
            evaluator=evaluator_path,
            config=config,
            iterations=args.max_iterations,
            output_dir=output_dir,
            cleanup=False  # Don't cleanup yet, need to extract data
        )
        
        duration = time.time() - start_time

        # Extract results
        best_score = result.best_score if result.best_program else 0.0
        best_cuda_code = result.best_code if result.best_program else "// No successful code generated"

        # Extract evolution history with best score tracking
        num_metric_calls, history = extract_evolution_history(output_dir)
        if num_metric_calls == 0:
            num_metric_calls = args.max_iterations  # Fallback approximation

        # Find when best score was achieved
        success_at = None
        if result.best_program:
            success_at = result.best_program.iteration_found
            if success_at is None:
                # Try to find from history
                for entry in history:
                    if entry['score'] == best_score and entry['is_new_best']:
                        success_at = entry['attempt']
                        break
                if success_at is None:
                    success_at = num_metric_calls  # Fallback

        # Always save summary to kernel_openevolve folder
        run_num = args.run_num
        os.makedirs(f'results/kernel_openevolve_{run_num}', exist_ok=True)
        summary_path = f"results/kernel_openevolve_{run_num}/task_{args.task_idx}_summary.json"

        summary_data = {
            'task_idx': args.task_idx,
            'best_speedup': best_score,
            'num_metric_calls': num_metric_calls,
            'best_at_metric_call': success_at,
            'duration_seconds': duration,
            'model': args.model,
            'method': 'openevolve',
            'num_correct_trials': args.num_correct_trials,
            'num_perf_trials': args.num_perf_trials,
            'history': history,  # Include per-step scores for all runs
        }

        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)

        print(f"\nSaved summary with per-step history to {summary_path}")

        # Save full result if requested
        if args.save_results:
            full_result_path = f"results/kernel_openevolve_{run_num}/task_{args.task_idx}_result.json"

            result_data = {
                'task_idx': args.task_idx,
                'best_speedup': best_score,
                'num_metric_calls': num_metric_calls,
                'best_at_metric_call': success_at,
                'best_cuda_code': best_cuda_code,
                'duration_seconds': duration,
                'history': history,  # Include step-by-step history with best scores
                'metrics': result.metrics,
                'settings': {
                    'model': args.model,
                    'max_iterations': args.max_iterations,
                    'num_workers': args.num_workers,
                    'evolution_strategy': 'full_rewrite',
                    'population_size': config.database.population_size,
                    'num_correct_trials': args.num_correct_trials,
                    'num_perf_trials': args.num_perf_trials,
                }
            }

            with open(full_result_path, 'w') as f:
                json.dump(result_data, f, indent=2)

            print(f"Saved full result to {full_result_path}")

        # Clean one-line summary
        print("\n" + "=" * 70)
        print(f"Task {args.task_idx} | Duration: {duration:.1f}s | Model: {args.model} | Method: OpenEvolve")
        print("=" * 70)
        if best_score > 0:
            print(f"✓ COMPLETED: Best speedup = {best_score:.4f}x")
            if success_at:
                print(f"  Achieved at iteration {success_at}/{num_metric_calls}")
        else:
            print(f"✗ FAILED: No valid kernel achieved positive speedup")
            print(f"  Max iterations ({num_metric_calls}) reached")
        print("=" * 70)
        
    finally:
        # Cleanup temporary directory
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"\nCleaned up temporary directory: {temp_dir}")
            except Exception as e:
                print(f"\nWarning: Could not clean up temp directory: {e}")


if __name__ == "__main__":
    main()

