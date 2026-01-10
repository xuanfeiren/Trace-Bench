"""
Test script for evaluating CUDA kernels using the guide.evaluate function.

This script tests CUDA kernel implementations from .txt files against
PyTorch reference implementations from the KernelBench dataset.

Usage:
    # Run on GPU machine (sequential evaluation - recommended for 1 GPU)
    python test_guide.py

    # Run with asyncio (for testing purposes, not faster on 1 GPU)
    python test_guide.py --async
"""

import os
import sys
import glob
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guide.evaluate import evaluate


def load_cuda_kernels(directory: str = "../my_process_agents", specific_file: str = None, num_repeats: int = 1) -> List[Dict[str, str]]:
    """
    Load CUDA kernel .txt files from the specified directory.

    Parameters
    ----------
    directory : str
        Directory containing .txt files with CUDA kernels
    specific_file : str, optional
        If provided, load only this specific file (can be relative to directory or absolute path)
    num_repeats : int
        Number of times to repeat the evaluation of the file(s)

    Returns
    -------
    List[Dict[str, str]]
        List of dicts with 'filename' and 'code' keys
    """
    # Resolve directory path relative to this script
    script_dir = Path(__file__).parent
    
    kernels = []
    
    if specific_file:
        # Handle specific file loading
        # First try as absolute path
        if os.path.isabs(specific_file):
            filepath = specific_file
        else:
            # Try relative to script directory
            filepath = (script_dir / specific_file).resolve()
            if not os.path.exists(filepath):
                # Try relative to provided directory
                kernel_dir = (script_dir / directory).resolve()
                filepath = kernel_dir / specific_file
        
        print(f"Loading specific CUDA kernel: {filepath}")
        
        if not os.path.exists(filepath):
            print(f"⚠️  File not found: {filepath}")
            return []
        
        try:
            with open(filepath, 'r') as f:
                code = f.read()
            
            # Add the kernel num_repeats times
            for i in range(num_repeats):
                kernels.append({
                    'filename': f"{os.path.basename(filepath)} (Run {i+1}/{num_repeats})",
                    'filepath': str(filepath),
                    'code': code,
                    'run_number': i+1
                })
            print(f"  ✓ Loaded: {os.path.basename(filepath)} x{num_repeats} times")
        except Exception as e:
            print(f"  ✗ Failed to load {filepath}: {e}")
            return []
    else:
        # Original behavior - load all files
        kernel_dir = (script_dir / directory).resolve()
        print(f"Loading CUDA kernels from: {kernel_dir}")

        # Find all .txt files (including error files)
        txt_files = glob.glob(str(kernel_dir / "*.txt"))
        kernel_files = txt_files  # Include all .txt files

        if not kernel_files:
            print(f"⚠️  No kernel files found in {kernel_dir}")
            print(f"    Looking for .txt files")
            return []

        print(f"Found {len(kernel_files)} kernel file(s)")

        for filepath in sorted(kernel_files):
            filename = os.path.basename(filepath)
            try:
                with open(filepath, 'r') as f:
                    code = f.read()
                
                # Add each file num_repeats times
                for i in range(num_repeats):
                    suffix = f" (Run {i+1}/{num_repeats})" if num_repeats > 1 else ""
                    kernels.append({
                        'filename': filename + suffix,
                        'filepath': filepath,
                        'code': code,
                        'run_number': i+1 if num_repeats > 1 else None
                    })
                print(f"  ✓ Loaded: {filename}" + (f" x{num_repeats} times" if num_repeats > 1 else ""))
            except Exception as e:
                print(f"  ✗ Failed to load {filename}: {e}")

    return kernels


def load_reference_implementation(problem_id: int = 1) -> str:
    """
    Load reference PyTorch implementation from KernelBench dataset.

    Parameters
    ----------
    problem_id : int
        Problem ID to load (1-based indexing)

    Returns
    -------
    str
        Reference PyTorch source code
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets library not installed. Run: pip install datasets"
        )

    print(f"\nLoading reference implementation from KernelBench dataset...")
    ds = load_dataset("allenanie/kernelbench_with_prompts")
    examples = [ex for ex in ds['train'] if ex['backend'] == 'cuda']

    if not examples:
        raise ValueError("No CUDA examples found in dataset")

    if problem_id < 1 or problem_id > len(examples):
        raise ValueError(f"Problem ID must be between 1 and {len(examples)}")

    # Use 0-based indexing
    example = examples[problem_id - 1]
    ref_arch_src = example['ref_arch_src']

    problem_name = example.get('problem', f"Problem {problem_id}")
    print(f"✓ Loaded reference for: {problem_name}")
    print(f"  Level: {example.get('level', 'N/A')}")
    print(f"  Backend: {example.get('backend', 'N/A')}")

    return ref_arch_src


def evaluate_sequential(
    kernels: List[Dict[str, str]],
    ref_arch_src: str,
    num_correct_trials: int = 1,
    num_perf_trials: int = 5
) -> List[Dict[str, Any]]:
    """
    Evaluate CUDA kernels sequentially (recommended for single GPU).

    Parameters
    ----------
    kernels : List[Dict[str, str]]
        List of kernel dicts with 'filename' and 'code'
    ref_arch_src : str
        Reference PyTorch implementation
    num_correct_trials : int
        Number of correctness validation trials
    num_perf_trials : int
        Number of performance measurement trials

    Returns
    -------
    List[Dict[str, Any]]
        List of evaluation results
    """
    print("\n" + "="*80)
    print("SEQUENTIAL EVALUATION (Recommended for Single GPU)")
    print("="*80)

    results = []

    for i, kernel in enumerate(kernels, 1):
        print(f"\n[{i}/{len(kernels)}] Evaluating: {kernel['filename']}")
        print("-" * 80)

        start_time = time.time()

        try:
            # Get score and feedback from evaluate
            score, feedback = evaluate(
                ref_arch_src=ref_arch_src,
                custom_cuda=kernel['code'],
                num_correct_trials=num_correct_trials,
                num_perf_trials=num_perf_trials
            )

            elapsed = time.time() - start_time

            # Parse result from feedback
            # Determine compiled and correctness from score and feedback
            compiled = score >= 0  # score is -1 on errors, 0 on failures, >0 on success
            correctness = score > 0  # score > 0 means it passed correctness and has speedup
            
            # Create result dict
            result = {
                'filename': kernel['filename'],
                'score': score,
                'feedback': feedback,
                'compiled': compiled,
                'correctness': correctness,
                'runtime': -1.0,  # Will be parsed from feedback if available
                'evaluation_time': elapsed
            }
            
            results.append(result)

            # Print summary
            status = "✓ PASSED" if correctness else "✗ FAILED"
            print(f"\n{status}")
            print(f"  Compiled:     {compiled}")
            print(f"  Correctness:  {correctness}")
            print(f"  Score:        {score:.4f}")

            # Print feedback (first 500 chars for more context)
            feedback_preview = feedback
            print(f"\n  Feedback:\n    {feedback_preview.replace(chr(10), chr(10) + '    ')}")

            print(f"\n  Eval time:    {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n✗ EXCEPTION")
            print(f"  Error:        {str(e)[:200]}")
            print(f"  Eval time:    {elapsed:.1f}s")

            results.append({
                'filename': kernel['filename'],
                'score': -1,
                'feedback': f"Exception: {str(e)}",
                'compiled': False,
                'correctness': False,
                'runtime': -1.0,
                'evaluation_time': elapsed
            })

    return results


def evaluate_async(
    kernels: List[Dict[str, str]],
    ref_arch_src: str,
    num_correct_trials: int = 1,
    num_perf_trials: int = 5,
    max_workers: int = 5
) -> List[Dict[str, Any]]:
    """
    Evaluate CUDA kernels using asyncio with thread pool.

    WARNING: On a single GPU, this won't be faster than sequential!
    All workers will compete for the same GPU, causing contention.

    This is mainly for demonstration/testing purposes.

    Parameters
    ----------
    kernels : List[Dict[str, str]]
        List of kernel dicts
    ref_arch_src : str
        Reference implementation
    num_correct_trials : int
        Correctness trials
    num_perf_trials : int
        Performance trials
    max_workers : int
        Maximum concurrent workers (threads)

    Returns
    -------
    List[Dict[str, Any]]
        List of evaluation results
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    print("\n" + "="*80)
    print(f"ASYNC EVALUATION (max_workers={max_workers})")
    print("="*80)
    print("⚠️  WARNING: On single GPU, this may be SLOWER than sequential!")
    print("    All workers compete for same GPU resources.")
    print()

    async def evaluate_one(kernel: Dict[str, str], index: int) -> Dict[str, Any]:
        """Async wrapper for single evaluation."""
        print(f"[{index}/{len(kernels)}] Starting: {kernel['filename']}")

        loop = asyncio.get_event_loop()
        start_time = time.time()

        try:
            # Run synchronous evaluate() in thread pool
            # Create wrapper function that evaluate returns tuple
            def evaluate_wrapper():
                return evaluate(
                    ref_arch_src=ref_arch_src,
                    custom_cuda=kernel['code'],
                    num_correct_trials=num_correct_trials,
                    num_perf_trials=num_perf_trials
                )
            
            score, feedback = await loop.run_in_executor(
                None,  # Use default executor
                evaluate_wrapper
            )

            elapsed = time.time() - start_time
            
            # Parse result from score and feedback
            compiled = score >= 0
            correctness = score > 0
            
            result = {
                'filename': kernel['filename'],
                'score': score,
                'feedback': feedback,
                'compiled': compiled,
                'correctness': correctness,
                'runtime': -1.0,
                'evaluation_time': elapsed
            }

            status = "✓ PASSED" if correctness else "✗ FAILED"
            print(f"[{index}/{len(kernels)}] {status}: {kernel['filename']} ({elapsed:.1f}s)")

            return result

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[{index}/{len(kernels)}] ✗ EXCEPTION: {kernel['filename']} - {str(e)[:100]}")

            return {
                'filename': kernel['filename'],
                'score': -1,
                'feedback': f"Exception: {str(e)}",
                'compiled': False,
                'correctness': False,
                'runtime': -1.0,
                'evaluation_time': elapsed
            }

    async def main():
        """Run all evaluations concurrently."""
        # Create tasks for all kernels
        tasks = [
            evaluate_one(kernel, i+1)
            for i, kernel in enumerate(kernels)
        ]

        # Wait for all to complete
        results = await asyncio.gather(*tasks)
        return results

    # Run async event loop
    start_time = time.time()
    results = asyncio.run(main())
    total_time = time.time() - start_time

    print(f"\nTotal async time: {total_time:.1f}s")

    return results


def print_summary(results: List[Dict[str, Any]]):
    """Print summary of evaluation results."""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)

    total = len(results)
    compiled = sum(1 for r in results if r.get('compiled', False))
    correct = sum(1 for r in results if r.get('correctness', False))

    print(f"\nTotal kernels:     {total}")
    print(f"Compiled:          {compiled}/{total} ({compiled/total*100:.1f}%)")
    print(f"Correct:           {correct}/{total} ({correct/total*100:.1f}%)")

    # Find best scoring kernel
    correct_results = [r for r in results if r.get('correctness', False)]
    if correct_results:
        best = max(correct_results, key=lambda r: r.get('score', 0))
        print(f"\nBest scoring kernel:")
        print(f"  File:      {best['filename']}")
        print(f"  Score:     {best['score']:.4f}x")

    # Total evaluation time
    total_eval_time = sum(r.get('evaluation_time', 0) for r in results)
    print(f"\nTotal evaluation time: {total_eval_time:.1f}s")

    print("\n" + "="*80)


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test CUDA kernel evaluation")
    parser.add_argument(
        '--async',
        action='store_true',
        dest='use_async',
        help='Use asyncio for parallel evaluation (not recommended for 1 GPU)'
    )
    parser.add_argument(
        '--kernels-dir',
        default='../my_process_agents',
        help='Directory containing CUDA kernel .txt files (default: ../my_process_agents)'
    )
    parser.add_argument(
        '--file',
        type=str,
        default='level1_prob1_cuda_custom_cuda_gpt5_example.txt',
        help='Specific file to evaluate (default: level1_prob1_cuda_custom_cuda_gpt5_example.txt)'
    )
    parser.add_argument(
        '--repeats',
        type=int,
        default=3,
        help='Number of times to evaluate the file (default: 3)'
    )
    parser.add_argument(
        '--problem-id',
        type=int,
        default=1,
        help='Problem ID from dataset (default: 1)'
    )
    parser.add_argument(
        '--correct-trials',
        type=int,
        default=1,
        help='Number of correctness trials (default: 1)'
    )
    parser.add_argument(
        '--perf-trials',
        type=int,
        default=5,
        help='Number of performance trials (default: 5)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=5,
        help='Max async workers (default: 5, only for --async mode)'
    )

    args = parser.parse_args()

    print("="*80)
    print("CUDA KERNEL EVALUATION TEST")
    print("="*80)

    # Load CUDA kernels
    kernels = load_cuda_kernels(
        directory=args.kernels_dir,
        specific_file=args.file,
        num_repeats=args.repeats
    )

    if not kernels:
        print("\n❌ No kernels to evaluate!")
        print(f"   Check file: {args.file}")
        return

    # Load reference implementation
    try:
        ref_arch_src = load_reference_implementation(args.problem_id)
    except Exception as e:
        print(f"\n❌ Failed to load reference: {e}")
        return

    print(f"\nEvaluation settings:")
    print(f"  Correctness trials: {args.correct_trials}")
    print(f"  Performance trials: {args.perf_trials}")
    print(f"  Mode: {'Async' if args.use_async else 'Sequential'}")
    if args.use_async:
        print(f"  Max workers: {args.max_workers}")

    # Run evaluation
    start_time = time.time()

    if args.use_async:
        results = evaluate_async(
            kernels,
            ref_arch_src,
            num_correct_trials=args.correct_trials,
            num_perf_trials=args.perf_trials,
            max_workers=args.max_workers
        )
    else:
        results = evaluate_sequential(
            kernels,
            ref_arch_src,
            num_correct_trials=args.correct_trials,
            num_perf_trials=args.perf_trials
        )

    total_time = time.time() - start_time

    # Print summary
    print_summary(results)

    print(f"\nTotal wall time: {total_time:.1f}s")
    print("\n✓ Test complete!")


if __name__ == "__main__":
    main()
