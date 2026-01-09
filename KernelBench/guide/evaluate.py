"""
Simple evaluation wrapper for CUDA kernel evaluation.

Provides a minimal evaluate() function that returns a score and feedback.

Usage:
    from guide.evaluate import evaluate

    score, feedback = evaluate(
        ref_arch_src=pytorch_code,
        custom_cuda=cuda_kernel,
        server_url="http://localhost:6000"
    )
"""

import os
import sys
from typing import Tuple

# Note: This client-server approach doesn't require external KernelBench installation
# The server handles all CUDA evaluation internally


def evaluate(
    ref_arch_src: str,
    custom_cuda: str,
    num_correct_trials: int = 1,
    num_perf_trials: int = 5,
    server_url: str = "http://localhost:6000",
    timeout: int = 300
) -> Tuple[float, str]:
    """
    Evaluate a CUDA kernel against PyTorch reference implementation.

    Parameters
    ----------
    ref_arch_src : str
        PyTorch reference implementation source code
    custom_cuda : str
        Custom CUDA kernel implementation source code
    num_correct_trials : int
        Number of correctness validation trials (default: 5)
    num_perf_trials : int
        Number of performance measurement trials (default: 100)
    server_url : str
        URL of the CUDA evaluation server (default: "http://localhost:6000")
    timeout : int
        Maximum time in seconds to wait for evaluation (default: 300)

    Returns
    -------
    Tuple[float, str]
        score : float
            Speedup value if successful, 0.0 if failed, -1 if evaluation error
        feedback : str
            Detailed feedback message containing all evaluation results
    """
    # Step 1: Import and create client
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    from cuda_eval_client import CUDAEvalClient

    client = CUDAEvalClient(server_url=server_url)

    # Check if server is running
    if not client.health_check():
        score = -1
        feedback = f"Evaluation failed: Server not reachable at {server_url}\n\nPlease start the server with: python cuda_eval_server.py"
        return score, feedback

    # Step 2: Send request to server
    try:
        response = client.evaluate_sync(
            problem_id=1,
            sample_id=0,
            custom_cuda=custom_cuda,
            ref_arch_src=ref_arch_src,
            timeout=timeout,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials
        )
    except Exception as e:
        score = -1
        feedback = f"Evaluation failed with error:\n{str(e)}\n\nPlease check server connection and try again."
        return score, feedback

    # Step 3: Get result
    if response['status'] != 'completed':
        score = -1
        feedback = f"Evaluation failed with status: {response['status']}\n\nError: {response.get('error', 'Unknown error')}"
        return score, feedback

    result = response.get('result', {})

    # Step 4: Use EXACTLY same logic as KernelGuide to compute score and feedback
    compiled = result.get('compiled', False)
    correctness = result.get('correctness', False)
    runtime = result.get('runtime', -1.0)
    ref_runtime = result.get('ref_runtime', -1.0)
    runtime_stats = result.get('runtime_stats', {})
    metadata = result.get('metadata', {})

    if not compiled:
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
        # SUCCESS: Use speedup as score
        speedup = metadata.get('speedup', None)

        if speedup is not None:
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

    return score, feedback


if __name__ == "__main__":
    pass