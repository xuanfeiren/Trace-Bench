"""
Simple evaluation wrapper for CUDA kernel evaluation.

Provides a minimal evaluate() function with just 4 parameters.

Usage:
    from guide.evaluate import evaluate

    result = evaluate(
        ref_arch_src=pytorch_code,
        custom_cuda=cuda_kernel,
        num_correct_trials=5,
        num_perf_trials=100
    )
"""

import os
import sys
from typing import Dict, Any

# Add KernelBench to Python path
KERNELBENCH_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../external/KernelBench")
)
if os.path.exists(KERNELBENCH_PATH) and KERNELBENCH_PATH not in sys.path:
    sys.path.insert(0, KERNELBENCH_PATH)


def evaluate(
    ref_arch_src: str,
    custom_cuda: str,
    num_correct_trials: int = 1,
    num_perf_trials: int = 5
) -> Dict[str, Any]:
    """
    Evaluate a CUDA kernel against PyTorch reference implementation.

    This function directly calls KernelBench's eval_kernel_against_ref with
    minimal parameters. Thread-safe for parallel execution.

    Parameters
    ----------
    ref_arch_src : str
        PyTorch reference implementation source code (as string).

        Must contain:
        - `class Model(nn.Module)` - PyTorch model
        - `def get_inputs()` - Returns list of test input tensors
        - `def get_init_inputs()` - Returns model init parameters

        Example:
            import torch
            import torch.nn as nn

            class Model(nn.Module):
                def forward(self, x):
                    return torch.relu(x)

            def get_inputs():
                return [torch.randn(1024, 1024)]

            def get_init_inputs():
                return []

    custom_cuda : str
        Custom CUDA kernel implementation source code (as string).

        Must contain:
        - Raw CUDA C++ code with `__global__` kernels
        - `torch.utils.cpp_extension.load_inline()` compilation
        - `class ModelNew(nn.Module)` - Wrapper for CUDA kernel
        - Same `get_inputs()` and `get_init_inputs()` as reference

        Example:
            import torch
            from torch.utils.cpp_extension import load_inline

            cuda_src = r'''
            #include <torch/extension.h>
            __global__ void relu_kernel(const float* in, float* out, int n) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < n) out[i] = fmaxf(in[i], 0.0f);
            }
            torch::Tensor relu_cuda(torch::Tensor input) {
                auto output = torch::empty_like(input);
                int n = input.numel();
                relu_kernel<<<(n+255)/256, 256>>>(
                    input.data_ptr<float>(),
                    output.data_ptr<float>(), n);
                return output;
            }
            '''

            ext = load_inline(
                name='relu',
                cpp_sources='torch::Tensor relu_cuda(torch::Tensor input);',
                cuda_sources=cuda_src,
                functions=['relu_cuda']
            )

            class ModelNew(nn.Module):
                def forward(self, x):
                    if x.is_cuda:
                        return ext.relu_cuda(x)
                    return torch.relu(x)

            def get_inputs():
                return [torch.randn(1024, 1024)]

            def get_init_inputs():
                return []

    num_correct_trials : int, optional
        Number of correctness validation trials (default: 1).

        Each trial runs with a different random seed and checks if custom
        kernel output matches reference using torch.allclose(rtol=1e-4, atol=1e-8).

        Must pass ALL trials to be considered correct.

        Recommended values:
        - 1: Quick testing during development
        - 5-10: Standard validation
        - 20+: Production/publication quality

    num_perf_trials : int, optional
        Number of performance measurement trials (default: 1).

        Each trial measures kernel runtime using CUDA events. Statistics
        (mean, std, min, max) are computed across all trials.

        Recommended values:
        - 1: Quick correctness-only check
        - 10-50: Fast performance estimate
        - 100-500: Stable benchmark results
        - 1000+: Publication-quality measurements

    Returns
    -------
    Dict[str, Any]
        Evaluation results dictionary:

        {
            'compiled': bool,
                True if CUDA kernel compiled successfully
                False if compilation errors occurred

            'correctness': bool,
                True if kernel passed ALL correctness trials
                False if any trial failed or compilation failed

            'score': float,
                Evaluation score for optimization:
                - 0.0 if not compiled
                - 0.0 if not correct
                - speedup value (ref_runtime / custom_runtime) if successful
                - 0.0 if performance measurement failed

            'feedback': str,
                Human-readable feedback message describing the result:
                - "COMPILATION FAILURE - ..." if not compiled
                - "CORRECTNESS FAILURE - ..." if not correct
                - "SUCCESS - Score (Speedup): X.XXx ..." if successful
                - "PERFORMANCE FAILURE - ..." if perf measurement failed

            'runtime': float,
                Mean runtime of custom kernel in MICROSECONDS (μs)
                -1.0 if kernel failed or didn't run performance tests

            'runtime_stats': {
                'mean': float,      # Mean runtime (microseconds)
                'std': float,       # Standard deviation
                'median': float,    # Median runtime
                'min': float,       # Fastest runtime
                'max': float,       # Slowest runtime
                'num_trials': int   # Number of trials run
            },

            'ref_runtime': float,
                Mean runtime of reference PyTorch implementation (microseconds)

            'ref_runtime_stats': {
                'mean': float,
                'std': float,
                'median': float,
                'min': float,
                'max': float,
                'num_trials': int
            },

            'metadata': {
                'speedup': float,
                    Performance ratio: ref_runtime / custom_runtime
                    >1.0 = custom is faster
                    <1.0 = custom is slower

                'hardware': str,
                    GPU model name (e.g., "NVIDIA A100")

                'backend': str,
                    Backend used ("cuda", "triton", etc.)

                'precision': str,
                    Data type used ("float32", "float16", etc.)

                'num_correctness_trials': int,
                    Number of correctness tests run

                'num_perf_trials': int,
                    Number of performance trials run

                'error': str,
                    Error message if evaluation failed
                    (only present if failed)

                'compilation_error': str,
                    Compilation error details
                    (only present if compilation failed)

                'correctness_error': str,
                    Correctness failure details
                    (only present if correctness failed)
            }
        }

    Raises
    ------
    ImportError
        If KernelBench is not installed or not in Python path
    RuntimeError
        If no GPU is available
    Exception
        Various exceptions from compilation or execution errors

    Thread Safety
    -------------
    This function is thread-safe and can be called from multiple threads:

    >>> from concurrent.futures import ThreadPoolExecutor
    >>> from guide.evaluate import evaluate
    >>>
    >>> with ThreadPoolExecutor(max_workers=10) as executor:
    ...     results = list(executor.map(
    ...         lambda data: evaluate(data['ref'], data['cuda'], 5, 100),
    ...         dataset
    ...     ))

    Examples
    --------
    # Quick correctness check (no performance measurement)
    >>> result = evaluate(ref, cuda, num_correct_trials=5, num_perf_trials=0)
    >>> if result['correctness']:
    ...     print("Kernel is correct!")
    >>> print(result['feedback'])  # Human-readable status message

    # Full evaluation with performance
    >>> result = evaluate(ref, cuda, num_correct_trials=5, num_perf_trials=100)
    >>> print(f"Score: {result['score']:.2f}")  # Speedup value for optimization
    >>> print(f"Compiled: {result['compiled']}")
    >>> print(f"Correct: {result['correctness']}")
    >>> print(result['feedback'])  # Detailed feedback message
    >>>
    >>> # If successful, access speedup and runtimes
    >>> if result['correctness']:
    ...     speedup = result['metadata']['speedup']
    ...     runtime_us = result['runtime']  # microseconds
    ...     runtime_ms = runtime_us / 1000  # convert to milliseconds
    ...     print(f"Speedup: {speedup:.2f}x")
    ...     print(f"Runtime: {runtime_ms:.3f} ms")

    # Parallel evaluation
    >>> from concurrent.futures import ThreadPoolExecutor
    >>> data = [
    ...     {'ref': ref1, 'cuda': cuda1},
    ...     {'ref': ref2, 'cuda': cuda2},
    ... ]
    >>> with ThreadPoolExecutor(max_workers=4) as executor:
    ...     results = list(executor.map(
    ...         lambda d: evaluate(d['ref'], d['cuda'], 5, 100),
    ...         data
    ...     ))
    >>> correct = sum(1 for r in results if r['correctness'])
    >>> print(f"{correct}/{len(results)} kernels passed")

    Notes
    -----
    - Requires KernelBench to be installed in external/KernelBench
    - First evaluation takes longer (~30-60s) due to CUDA compilation
    - Subsequent evaluations are faster due to PyTorch's compilation cache
    - Uses current CUDA device (set via torch.cuda.set_device())
    - All tensors are float32 by default
    - Correctness uses torch.allclose with rtol=1e-4, atol=1e-8
    - Performance uses CUDA events for precise timing
    - GPU memory is synchronized between tests
    """
    try:
        from src.eval import eval_kernel_against_ref
    except ImportError:
        raise ImportError(
            "Cannot import eval_kernel_against_ref from KernelBench. "
            f"Make sure KernelBench is installed at: {KERNELBENCH_PATH}\n"
            "Run: bash install.sh"
        )

    # Call KernelBench evaluation with minimal parameters
    result = eval_kernel_against_ref(
        original_model_src=ref_arch_src,
        custom_model_src=custom_cuda,
        verbose=False,                      # Silent mode
        measure_performance=(num_perf_trials > 0),  # Only if perf trials requested
        num_correct_trials=num_correct_trials,
        num_perf_trials=num_perf_trials
    )

    # Convert to dict if it's a Pydantic model or dataclass
    if hasattr(result, 'model_dump'):
        result_dict = result.model_dump()
    elif hasattr(result, 'dict'):
        result_dict = result.dict()
    elif hasattr(result, '__dict__'):
        result_dict = vars(result)
    else:
        result_dict = result

    # Extract score and feedback (same logic as KernelGuide in kernel_PS_modal.py)
    compiled = result_dict.get('compiled', False)
    correctness = result_dict.get('correctness', False)
    runtime = result_dict.get('runtime', -1.0)
    ref_runtime = result_dict.get('ref_runtime', -1.0)
    runtime_stats = result_dict.get('runtime_stats', {})
    metadata = result_dict.get('metadata', {})

    if not compiled:
        # COMPILATION FAILURE
        score = 0.0
        error_name = metadata.get('compilation_error_name', metadata.get('error_type', 'Unknown'))
        error_msg = metadata.get('compilation_error', metadata.get('error', 'Unknown error'))
        feedback_from_eval = metadata.get('feedback', '')

        feedback = f"COMPILATION FAILURE - Score: {score}\n\n"
        feedback += f"Error Type: {error_name}\n"
        feedback += f"Error Message: {error_msg}\n\n"
        if feedback_from_eval:
            feedback += f"Detailed Feedback:\n{feedback_from_eval}"
        else:
            feedback += f"Full Metadata: {metadata}"

    elif not correctness:
        # CORRECTNESS FAILURE
        score = 0.0
        runtime_error_name = metadata.get('runtime_error_name', 'Unknown')
        runtime_error = metadata.get('runtime_error', 'Unknown error')
        feedback_from_eval = metadata.get('feedback', '')

        feedback = f"CORRECTNESS FAILURE - Score: {score}\n\n"
        feedback += f"The kernel compiled but failed correctness tests.\n"
        feedback += f"Error Type: {runtime_error_name}\n"
        feedback += f"Error Details: {runtime_error}\n\n"
        if feedback_from_eval:
            feedback += f"Detailed Feedback:\n{feedback_from_eval}"
        else:
            feedback += f"Full Metadata: {metadata}"

    else:
        # SUCCESS or PERFORMANCE FAILURE
        speedup = metadata.get('speedup', None)

        if speedup is not None:
            # SUCCESS with speedup
            score = speedup
            feedback = (
                f"SUCCESS - Score (Speedup): {score:.4f}x\n\n"
                f"Custom Kernel Runtime: {runtime:.4f} μs\n"
                f"Reference Runtime: {ref_runtime:.4f} μs\n"
                f"Speedup: {speedup:.4f}x\n"
                f"Runtime Stats: {runtime_stats}\n\n"
            )
        else:
            # Speedup not available
            score = 0.0
            error_during_perf = metadata.get('error_during_performance', None)

            if error_during_perf:
                # Performance measurement failed
                error_type = metadata.get('error_during_performance_name', 'Unknown')
                feedback = (
                    f"PERFORMANCE FAILURE - Score: {score}\n\n"
                    f"The kernel compiled and passed correctness tests, but failed during performance measurement.\n"
                    f"Error Type: {error_type}\n"
                    f"Error: {error_during_perf}\n\n"
                )
            else:
                # Speedup genuinely missing (shouldn't happen)
                feedback = (
                    f"SUCCESS but speedup not available - Score: {score}\n\n"
                    f"Custom Kernel Runtime: {runtime:.4f} μs\n"
                    f"Runtime Stats: {runtime_stats}\n"
                    f"WARNING: Speedup not found in metadata. Check eval settings.\n\n"
                )

        # Add feedback from eval if available
        feedback_from_eval = metadata.get('feedback', '')
        if feedback_from_eval:
            feedback += f"Detailed Feedback:\n{feedback_from_eval}"

    # Add score and feedback to result dict
    result_dict['score'] = score
    result_dict['feedback'] = feedback

    return result_dict


if __name__ == "__main__":
    print("=" * 80)
    print("SIMPLE CUDA KERNEL EVALUATOR")
    print("=" * 80)
    print()
    print("This module provides evaluate() - a simple 4-parameter function for")
    print("evaluating CUDA kernels against PyTorch reference implementations.")
    print()
    print("USAGE:")
    print("  from guide.evaluate import evaluate")
    print()
    print("  result = evaluate(")
    print("      ref_arch_src=pytorch_code,    # PyTorch reference (string)")
    print("      custom_cuda=cuda_kernel,      # CUDA kernel (string)")
    print("      num_correct_trials=5,         # Correctness tests (1-20)")
    print("      num_perf_trials=100           # Performance trials (0-1000)")
    print("  )")
    print()
    print("  print(f\"Compiled: {result['compiled']}\")")
    print("  print(f\"Correct: {result['correctness']}\")")
    print("  print(f\"Speedup: {result['metadata']['speedup']:.2f}x\")")
    print()
    print("PARALLEL USAGE:")
    print("  from concurrent.futures import ThreadPoolExecutor")
    print()
    print("  with ThreadPoolExecutor(max_workers=10) as executor:")
    print("      results = list(executor.map(")
    print("          lambda d: evaluate(d['ref'], d['cuda'], 5, 100),")
    print("          dataset")
    print("      ))")
    print()
    print("PARAMETERS:")
    print("  ref_arch_src       - PyTorch reference implementation (string)")
    print("  custom_cuda        - CUDA kernel implementation (string)")
    print("  num_correct_trials - Number of correctness tests (default: 1)")
    print("  num_perf_trials    - Number of performance trials (default: 1)")
    print()
    print("RETURNS:")
    print("  Dictionary with: compiled, correctness, runtime, speedup, metadata")
    print()
    print("=" * 80)

    # Check if KernelBench is available
    if os.path.exists(KERNELBENCH_PATH):
        print()
        print(f"✓ KernelBench found at: {KERNELBENCH_PATH}")
        try:
            from src.eval import eval_kernel_against_ref
            print("✓ eval_kernel_against_ref imported successfully")
            print()
            print("✓ Ready to evaluate!")
        except ImportError as e:
            print(f"✗ Cannot import eval_kernel_against_ref: {e}")
            print()
            print("  Run: bash install.sh")
    else:
        print()
        print(f"✗ KernelBench not found at: {KERNELBENCH_PATH}")
        print()
        print("  Run: bash install.sh")
