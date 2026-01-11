# Optimize CUDA kernels using DSPy sequential refinement
# Uses DSPy modules to iteratively improve CUDA code based on performance feedback

import sys
import os
import numpy as np
import time
import argparse
import json

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import dspy

import litellm
litellm.drop_params = True
litellm.suppress_debug_info = True

# Add retry logic for rate limiting errors
litellm.num_retries = 3
litellm.request_timeout = 300

# Import dataset utilities
from dataset.utils import create_matrix_multiplication_dataset

# Import evaluation function
from guide.evaluate import evaluate
# def evaluate(ref_arch_src, custom_cuda, num_correct_trials, num_perf_trials):
#     return 1.0, "Dummy feedback"
from opto.optimizers.utils import print_color

# Set random seeds for reproducibility
np.random.seed(10)


class CUDAKernelGenerator(dspy.Signature):
    """Generate a CUDA kernel from task description."""
    task_description = dspy.InputField(desc="Task description and requirements for the CUDA kernel")
    cuda_code = dspy.OutputField(desc="Complete, compilable CUDA kernel implementation")


class CUDAKernelRefiner(dspy.Signature):
    """Refine a CUDA kernel based on performance feedback."""
    task_description = dspy.InputField(desc="Original task description")
    current_cuda_code = dspy.InputField(desc="Current CUDA implementation that needs improvement")
    performance_feedback = dspy.InputField(desc="Performance evaluation feedback and suggestions")
    cuda_code = dspy.OutputField(desc="Improved CUDA kernel implementation")


class DSPyCUDAAgent(dspy.Module):
    """
    DSPy agent that sequentially refines CUDA kernels based on performance feedback.
    """
    
    def __init__(self):
        super().__init__()
        # Initial generator for first attempt
        self.generator = dspy.ChainOfThought(CUDAKernelGenerator)
        # Refiner for subsequent attempts based on feedback
        self.refiner = dspy.ChainOfThought(CUDAKernelRefiner)
    
    def forward(self, task_description: str, feedback: str = None, current_code: str = None):
        """
        Generate or refine CUDA kernel code.
        
        Args:
            task_description: Task description for CUDA kernel
            feedback: Performance feedback (None for first attempt)
            current_code: Current CUDA code (None for first attempt)
            
        Returns:
            dspy.Prediction with cuda_code
        """
        if feedback is None or current_code is None:
            # First attempt - generate from scratch
            result = self.generator(task_description=task_description)
        else:
            # Refinement - improve based on feedback
            result = self.refiner(
                task_description=task_description,
                current_cuda_code=current_code,
                performance_feedback=feedback
            )
        
        return result


def sequential_optimization(
    agent: DSPyCUDAAgent,
    task_description: str,
    ref_arch_src: str,
    max_attempts: int = 10,
    num_correct_trials: int = 1,
    num_perf_trials: int = 5,
    verbose: bool = False
) -> dict:
    """
    Sequentially optimize CUDA kernel using DSPy agent and KernelBench evaluation.
    
    Args:
        agent: DSPy agent for generating/refining CUDA code
        task_description: Task description for CUDA kernel
        ref_arch_src: Reference PyTorch implementation
        max_attempts: Maximum number of refinement attempts
        num_correct_trials: Number of correctness trials for evaluation
        num_perf_trials: Number of performance trials for evaluation
        verbose: Whether to print detailed logs
        
    Returns:
        Dictionary with results
    """
    current_cuda_code = None
    current_feedback = None
    best_score = 0.0
    best_cuda_code = None
    
    history = []
    
    for attempt in range(1, max_attempts + 1):
        if verbose:
            print(f"\n{'='*70}")
            print(f"Attempt {attempt}/{max_attempts}")
            print(f"{'='*70}")
        
        # Generate or refine CUDA code
        try:
            result = agent(
                task_description=task_description,
                feedback=current_feedback,
                current_code=current_cuda_code
            )
            cuda_code = result.cuda_code
            
            if verbose:
                print(f"\nGenerated CUDA code (first 500 chars):")
                print("-" * 70)
                print(cuda_code[:500] + ("..." if len(cuda_code) > 500 else ""))
                print("-" * 70)
                
        except Exception as e:
            if verbose:
                print(f"Error generating code: {e}")
            cuda_code = "// Error generating CUDA code"
        
        # Evaluate with KernelBench
        try:
            if verbose:
                print(f"\nEvaluating CUDA kernel...")
            
            score, feedback = evaluate(
                ref_arch_src=ref_arch_src,
                custom_cuda=cuda_code,
                num_correct_trials=num_correct_trials,
                num_perf_trials=num_perf_trials
            )

            print_color(f"Score: {score}", 'green')
            print_color(f"Feedback: {feedback}", 'yellow')
            if verbose:
                print(f"\n✓ Evaluation complete")
                
        except Exception as e:
            score = 0.0
            feedback = f"Error during evaluation: {str(e)}"
            if verbose:
                print(f"\n✗ Evaluation error: {e}")
        
        # Track history
        history.append({
            'attempt': attempt,
            'score': score,
            'feedback': feedback[:500] if len(feedback) > 500 else feedback,  # Truncate for storage
        })
        
        if verbose:
            print(f"\nScore: {score:.4f}")
            print(f"Feedback: {feedback[:300]}{'...' if len(feedback) > 300 else ''}")
        
        # Update best
        if score > best_score:
            best_score = score
            best_cuda_code = cuda_code
            if verbose:
                print(f"★ New best score: {best_score:.4f}")
        
        
        # Prepare for next iteration
        current_cuda_code = cuda_code
        current_feedback = feedback
    
    # Failed to reach score 1.0
    if verbose:
        print(f"\n{'='*70}")
        print(f"✗ FAILED after {max_attempts} attempts")
        print(f"  Best score achieved: {best_score:.4f}")
        print(f"{'='*70}")
    
    return {
        'success': False,
        'attempts': max_attempts,
        'best_score': best_score,
        'best_cuda_code': best_cuda_code,
        'final_feedback': current_feedback,
        'history': history
    }


def create_single_task_dataset(task_idx: int):
    """
    Create a single task dataset from the matrix multiplication dataset.
    """
    ds = create_matrix_multiplication_dataset()
    return {'inputs': [ds[task_idx]['input']], 'infos': [ds[task_idx]['ref_arch_src']]}


def main():
    parser = argparse.ArgumentParser(description='Optimize CUDA kernel using DSPy sequential refinement')
    
    # Task parameters
    parser.add_argument('--task_idx', type=int, default=0,
                       help='Task index from KernelBench dataset (default: 0)')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='claude-3.7-sonnet',
                       help='LLM model name (e.g., claude-3.7-sonnet, gpt-4o)')
    
    # Optimization parameters
    parser.add_argument('--max_attempts', type=int, default=10,
                       help='Maximum number of refinement attempts (default: 50)')
    parser.add_argument('--num_correct_trials', type=int, default=1,
                       help='Number of correctness trials (default: 1)')
    parser.add_argument('--num_perf_trials', type=int, default=5,
                       help='Number of performance trials (default: 5)')
    
    # Logging parameters
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Print detailed logs during optimization')
    parser.add_argument('--save_results', action='store_true', default=False,
                       help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Prepare model name with provider prefix
    model_name = args.model
    if "/" not in model_name:
        if "claude" in model_name.lower():
            model_name = f"anthropic/{model_name}"
        elif "gpt" in model_name.lower():
            model_name = f"openai/{model_name}"
        elif "gemini" in model_name.lower():
            model_name = f"gemini/{model_name}"
    
    # Configure DSPy
    print(f"Configuring DSPy with {model_name}...")
    lm = dspy.LM(model=model_name, max_tokens=8192, cache=False)
    dspy.configure(lm=lm)
    
    # Load task
    print(f"\nLoading task {args.task_idx} from KernelBench dataset...")
    task = create_single_task_dataset(args.task_idx)
    input_text = task['inputs'][0]
    ref_arch_src = task['infos'][0]
    print(f"Task loaded successfully. Task ID: {args.task_idx}")
    print(f"\nTask description (first 500 chars):")
    print("=" * 70)
    print(input_text)
    print("=" * 70)
    
    # Initialize DSPy agent
    print("\nInitializing DSPy CUDA agent...")
    agent = DSPyCUDAAgent()
    
    # Run sequential optimization
    print(f"\nStarting DSPy sequential optimization:")
    print(f"  Model: {model_name}")
    print(f"  Max attempts: {args.max_attempts}")
    print(f"  Correctness trials: {args.num_correct_trials}")
    print(f"  Performance trials: {args.num_perf_trials}")
    print()
    
    start_time = time.time()
    
    result = sequential_optimization(
        agent=agent,
        task_description=input_text,
        ref_arch_src=ref_arch_src,
        max_attempts=args.max_attempts,
        num_correct_trials=args.num_correct_trials,
        num_perf_trials=args.num_perf_trials,
        verbose=args.verbose
    )
    
    duration = time.time() - start_time
    
    # Save result if requested
    if args.save_results:
        save_dir = os.path.join(project_root, "results", "kernel_dspy")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"dspy_task_{args.task_idx}_result.json")
        
        result_data = {
            'task_idx': args.task_idx,
            'success': result['success'],
            'attempts': result['attempts'],
            'best_score': result['best_score'],
            'best_cuda_code': result['best_cuda_code'],
            'duration_seconds': duration,
            'history': result['history'],
            'settings': {
                'model': model_name,
                'max_attempts': args.max_attempts,
                'num_correct_trials': args.num_correct_trials,
                'num_perf_trials': args.num_perf_trials,
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"\nSaved result to {save_path}")
    
    # Print final summary
    print("\n" + "=" * 70)
    print(f"Task {args.task_idx} | Duration: {duration:.1f}s | Model: {model_name.split('/')[-1]}")
    print("=" * 70)
    if result['success']:
        print(f"✓ SUCCESS: Reached score >= 1.0 at attempt {result['attempts']}/{args.max_attempts}")
        print(f"  Final score: {result['best_score']:.4f}")
    else:
        print(f"✗ FAILED: Max attempts ({result['attempts']}) reached without achieving score >= 1.0")
        print(f"  Best score achieved: {result['best_score']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()

