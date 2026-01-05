"""
Wrapper for KernelBench evaluation using Modal.

Provides a simple interface for evaluating CUDA kernels on remote GPUs.
Supports both single and batch evaluations with persistent Modal app.

Usage:
    # Single evaluation
    evaluator = KernelEvaluator(gpu="L40S")
    result = evaluator.evaluate(ref_arch_src, custom_cuda)

    # Multiple evaluations
    evaluator = KernelEvaluator(gpu="L40S", verbose=False)
    result1 = evaluator.evaluate(ref_arch_src1, custom_cuda1)
    result2 = evaluator.evaluate(ref_arch_src2, custom_cuda2)

    # Batch evaluation
    results = evaluator.evaluate_batch([
        {"ref_arch_src": ref1, "custom_cuda": cuda1},
        {"ref_arch_src": ref2, "custom_cuda": cuda2},
    ])
"""

import os
from typing import List, Dict, Any, Optional
from evaluate_with_modal import eval_single_sample_modal, gpu_arch_mapping, app


class KernelEvaluator:
    """
    Wrapper class for evaluating CUDA kernels on Modal GPUs.

    Maintains a persistent connection to Modal, allowing multiple evaluations
    without reinitializing the infrastructure.
    """

    def __init__(
        self,
        gpu: str = "L40S",
        verbose: bool = False,
        num_correct_trials: int = 5,
        num_perf_trials: int = 100,
    ):
        """
        Initialize the KernelEvaluator.

        Args:
            gpu: GPU type to use (L40S, H100, A100, L4, T4, A10G)
            verbose: Whether to print detailed logs during evaluation
            num_correct_trials: Number of correctness validation trials
            num_perf_trials: Number of performance measurement trials
        """
        self.gpu = gpu
        self.verbose = verbose
        self.num_correct_trials = num_correct_trials
        self.num_perf_trials = num_perf_trials

        if gpu not in gpu_arch_mapping:
            raise ValueError(
                f"Unknown GPU type: {gpu}. Must be one of {list(gpu_arch_mapping.keys())}"
            )

        self.gpu_arch = gpu_arch_mapping[gpu]

    def evaluate(
        self,
        ref_arch_src: str,
        custom_cuda: str,
        verbose: Optional[bool] = None,
        num_correct_trials: Optional[int] = None,
        num_perf_trials: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single CUDA kernel against the reference implementation.

        Args:
            ref_arch_src: Reference architecture source code
            custom_cuda: Custom CUDA kernel implementation
            verbose: Override default verbosity setting
            num_correct_trials: Override default correctness trials
            num_perf_trials: Override default performance trials

        Returns:
            Dictionary containing evaluation results with keys:
                - compiled: bool, whether the kernel compiled successfully
                - correctness: bool, whether the kernel passed correctness tests
                - metadata: dict with hardware info and trial counts
                - runtime: float, average runtime in milliseconds
                - runtime_stats: dict with mean, std, min, max, num_trials
        """
        # Use instance defaults if not overridden
        verbose = verbose if verbose is not None else self.verbose
        num_correct_trials = num_correct_trials if num_correct_trials is not None else self.num_correct_trials
        num_perf_trials = num_perf_trials if num_perf_trials is not None else self.num_perf_trials

        result = eval_single_sample_modal.remote(
            ref_arch_src,
            custom_cuda,
            verbose,
            self.gpu_arch,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials
        )

        return result

    def evaluate_batch(
        self,
        eval_configs: List[Dict[str, str]],
        verbose: Optional[bool] = None,
        num_correct_trials: Optional[int] = None,
        num_perf_trials: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple CUDA kernels in parallel.

        Args:
            eval_configs: List of evaluation configurations, each containing:
                - ref_arch_src: Reference architecture source code
                - custom_cuda: Custom CUDA kernel implementation
            verbose: Override default verbosity setting
            num_correct_trials: Override default correctness trials
            num_perf_trials: Override default performance trials

        Returns:
            List of evaluation result dictionaries
        """
        verbose = verbose if verbose is not None else self.verbose
        num_correct_trials = num_correct_trials if num_correct_trials is not None else self.num_correct_trials
        num_perf_trials = num_perf_trials if num_perf_trials is not None else self.num_perf_trials

        # Submit all evaluations in parallel
        futures = []
        for config in eval_configs:
            future = eval_single_sample_modal.spawn(
                config["ref_arch_src"],
                config["custom_cuda"],
                verbose,
                self.gpu_arch,
                num_correct_trials=num_correct_trials,
                num_perf_trials=num_perf_trials
            )
            futures.append(future)

        # Collect results
        results = []
        for future in futures:
            result = future.get()
            results.append(result)

        return results

    def evaluate_from_files(
        self,
        ref_arch_file: str,
        custom_cuda_file: str,
        verbose: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate CUDA kernel from file paths.

        Args:
            ref_arch_file: Path to reference architecture file
            custom_cuda_file: Path to custom CUDA kernel file
            verbose: Override default verbosity setting

        Returns:
            Dictionary containing evaluation results
        """
        with open(ref_arch_file, 'r') as f:
            ref_arch_src = f.read()

        with open(custom_cuda_file, 'r') as f:
            custom_cuda = f.read()

        return self.evaluate(ref_arch_src, custom_cuda, verbose=verbose)


def quick_example():
    """Example usage of KernelEvaluator."""
    from datasets import load_dataset

    # Load sample data
    ds = load_dataset("allenanie/kernelbench_with_prompts")
    examples = [ex for ex in ds['train'] if ex['backend'] == 'cuda']
    ex1 = examples[0]

    ref_arch_src = ex1['ref_arch_src']
    custom_cuda = open("level1_prob1_cuda_custom_cuda_gpt5_example.txt").read()

    # Initialize evaluator
    evaluator = KernelEvaluator(gpu="L40S", verbose=False)

    # Single evaluation
    print("Running single evaluation...")
    result = evaluator.evaluate(ref_arch_src, custom_cuda)
    print(f"Result: {result}")

    # Multiple sequential evaluations (reusing the same evaluator)
    print("\nRunning multiple sequential evaluations...")
    for i in range(3):
        result = evaluator.evaluate(ref_arch_src, custom_cuda)
        print(f"Evaluation {i+1}: correctness={result['correctness']}, runtime={result['runtime']}ms")

    # Batch evaluation (parallel)
    print("\nRunning batch evaluation (3 in parallel)...")
    results = evaluator.evaluate_batch([
        {"ref_arch_src": ref_arch_src, "custom_cuda": custom_cuda},
        {"ref_arch_src": ref_arch_src, "custom_cuda": custom_cuda},
        {"ref_arch_src": ref_arch_src, "custom_cuda": custom_cuda},
    ])

    for i, result in enumerate(results):
        print(f"Batch result {i+1}: correctness={result['correctness']}, runtime={result['runtime']}ms")


if __name__ == "__main__":
    quick_example()
