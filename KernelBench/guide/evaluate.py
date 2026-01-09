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
    num_perf_trials: int = 5,
    server_url: str = "http://localhost:6000",
    timeout: int = 300
) -> Dict[str, Any]:
    """
    Evaluate a CUDA kernel against PyTorch reference implementation using remote server.

    This function creates a new client instance for each evaluation call and sends
    the request to a remote CUDA evaluation server. Each call is independent and
    isolated, making it suitable for concurrent evaluations and preventing CUDA
    errors from affecting subsequent evaluations.

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
        Number of performance measurement trials (default: 5).

        Each trial measures kernel runtime using CUDA events. Statistics
        (mean, std, min, max) are computed across all trials.

        Recommended values:
        - 1: Quick correctness-only check
        - 10-50: Fast performance estimate
        - 100-500: Stable benchmark results
        - 1000+: Publication-quality measurements

    server_url : str, optional
        URL of the remote CUDA evaluation server (default: "http://localhost:6000").
        Each call to evaluate() creates a new client instance and sends a request
        to this server. The server must be running and accessible.

    timeout : int, optional
        Maximum time in seconds to wait for evaluation completion (default: 300).
        If the evaluation takes longer than this, a timeout error will be raised.

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
    ConnectionError
        If the remote server is not reachable or not running
    TimeoutError
        If the evaluation takes longer than the specified timeout
    requests.RequestException
        For various network-related errors during communication with server
    Exception
        Various exceptions from server-side compilation or execution errors

    Thread Safety
    -------------
    This function is fully thread-safe since each call creates a new client instance.
    Multiple threads can call evaluate() concurrently without interference:

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

    # Parallel evaluation with multiple server requests
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

    # Custom server URL and timeout
    >>> result = evaluate(
    ...     ref, cuda, 
    ...     server_url="http://gpu-server:8000",
    ...     timeout=600  # 10 minutes
    ... )

    Notes
    -----
    - Requires a running CUDA evaluation server at the specified URL
    - Each call creates a new HTTP client instance for complete isolation
    - Server handles CUDA compilation and execution in isolated processes
    - First evaluation on server may take longer due to CUDA compilation
    - Network latency affects total evaluation time
    - Server must have GPU access and CUDA toolkit installed
    - All tensors are float32 by default
    - Correctness uses torch.allclose with rtol=1e-4, atol=1e-8
    - Performance uses CUDA events for precise timing
    """
    # Import the client (do this inside function to avoid import issues)
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    client_path = os.path.join(parent_dir, 'cuda_eval_client.py')

    if not os.path.exists(client_path):
        raise ImportError(
            f"Cannot find cuda_eval_client.py at {client_path}. "
            "Make sure you're running from the KernelBench directory."
        )

    # Add parent to path and import
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    from cuda_eval_client import CUDAEvalClient

    # Create a new client instance for this evaluation
    # This ensures complete isolation between calls
    client = CUDAEvalClient(server_url=server_url)

    # Check if server is running
    if not client.health_check():
        raise ConnectionError(
            f"CUDA evaluation server not reachable at {server_url}. "
            "Please start the server with: python cuda_eval_server.py"
        )

    try:
        # Submit evaluation synchronously with custom parameters
        # Use dummy IDs since we're providing source code directly
        response = client.evaluate_sync(
            problem_id=1,  # Dummy ID
            sample_id=0,   # Dummy ID
            custom_cuda=custom_cuda,
            ref_arch_src=ref_arch_src,
            timeout=timeout
        )
    except Exception as e:
        # Handle client-side errors (network, timeout, etc.)
        return {
            'compiled': False,
            'correctness': False,
            'runtime': -1.0,
            'score': 0.0,
            'metadata': {'error': f"Client error: {str(e)}"},
            'feedback': f"Client-side error during evaluation: {str(e)}"
        }

    # Process server response
    if response['status'] == 'completed':
        result_dict = response.get('result', {})
    elif response['status'] == 'failed':
        # Server-side failure
        result_dict = {
            'compiled': False,
            'correctness': False,
            'runtime': -1.0,
            'score': 0.0,
            'metadata': {'error': response.get('error', 'Server evaluation failed')},
            'feedback': f"Server evaluation failed: {response.get('error', 'Unknown error')}"
        }
    else:
        # Unexpected status
        result_dict = {
            'compiled': False,
            'correctness': False,
            'runtime': -1.0,
            'score': 0.0,
            'metadata': {'error': f"Unexpected status: {response['status']}"},
            'feedback': f"Unexpected server response status: {response['status']}"
        }

    # Extract and process evaluation results
    compiled = result_dict.get('compiled', False)
    correctness = result_dict.get('correctness', False)
    runtime = result_dict.get('runtime', -1.0)
    ref_runtime = result_dict.get('ref_runtime', -1.0)
    runtime_stats = result_dict.get('runtime_stats', {})
    metadata = result_dict.get('metadata', {})

    # Generate score and feedback based on results
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

    # Add computed score and feedback to result dict
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
    print("      num_perf_trials=100,          # Performance trials (0-1000)")
    print("      server_url='http://localhost:6000',  # Remote server URL")
    print("      timeout=300                   # Timeout in seconds")
    print("  )")
    print()
    print("  print(f\"Compiled: {result['compiled']}\")")
    print("  print(f\"Correct: {result['correctness']}\")")
    print("  print(f\"Speedup: {result['metadata']['speedup']:.2f}x\")")
    print()
    print("PARALLEL USAGE (Each call creates new client):")
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
    print("  num_perf_trials    - Number of performance trials (default: 5)")
    print("  server_url         - Remote CUDA server URL (default: localhost:6000)")
    print("  timeout            - Evaluation timeout in seconds (default: 300)")
    print()
    print("RETURNS:")
    print("  Dictionary with: compiled, correctness, runtime, speedup, metadata")
    print()
    print("SERVER REQUIREMENTS:")
    print("  - Remote CUDA evaluation server must be running")
    print("  - Server must be accessible at the specified URL")
    print("  - Each evaluate() call creates a new client instance")
    print()
    print("=" * 80)

    # Check if client is available and server connectivity
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    client_path = os.path.join(parent_dir, 'cuda_eval_client.py')
    
    if os.path.exists(client_path):
        print()
        print(f"✓ CUDA client found at: {client_path}")
        try:
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            from cuda_eval_client import CUDAEvalClient
            print("✓ CUDAEvalClient imported successfully")
            
            # Test server connectivity
            client = CUDAEvalClient()
            if client.health_check():
                print("✓ Server is running and accessible!")
                try:
                    status = client.get_server_status()
                    print(f"✓ Server status: {status}")
                except:
                    print("✓ Server is running (status details unavailable)")
            else:
                print("✗ Server not reachable at http://localhost:6000")
                print("  Start server with: python cuda_eval_server.py")
            print()
            print("✓ Ready to evaluate with remote server!")
        except ImportError as e:
            print(f"✗ Cannot import CUDAEvalClient: {e}")
            print()
            print("  Make sure cuda_eval_client.py is available")
    else:
        print()
        print(f"✗ CUDA client not found at: {client_path}")
        print()
        print("  Make sure you're running from the KernelBench directory")
