"""
Persistent Modal GPU evaluator for kernel optimization.
This module provides a reusable Modal container that stays warm across multiple evaluations.
"""

import os
import modal

# Reuse the same app and image configuration from evaluate_with_modal.py
REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

REPO_EXTERNAL_LIB_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../external/",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_EXTERNAL_LIB_PATH, "KernelBench")

gpu_arch_mapping = {
    "L40S": ["Ada"],
    "H100": ["Hopper"],
    "A100": ["Ampere"],
    "L4": ["Ada"],
    "T4": ["Turing"],
    "A10G": ["Ampere"]
}

cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install("git", "gcc-10", "g++-10", "clang")
    .pip_install(
        "anthropic",
        "numpy",
        "openai",
        "packaging",
        "pydra_config",
        "torch==2.5.0",
        "tqdm",
        "datasets",
        "huggingface-hub<1.0",
        "transformers",
        "google-generativeai",
        "together",
        "pytest",
        "ninja",
        "utils",
        "dotenv",
        "triton",
        "litellm",
        "einops"
    ).env({
        "PYTHONPATH": "/root/KernelBench"
    }).add_local_dir(
        KERNEL_BENCH_PATH,
        remote_path="/root/KernelBench"
    ).add_local_file(
        os.path.join(REPO_TOP_PATH, "KernelBench/level1_prob1_cuda_custom_cuda_gpt5_example.txt"),
        remote_path="/root"
    )
)

app = modal.App("trace_bench_kernel_persistent_eval")


# Helper function to process evaluation results
def _process_eval_result(result):
    """Convert evaluation result to plain dict for safe serialization."""
    if hasattr(result, 'model_dump'):
        return result.model_dump()
    elif hasattr(result, 'dict'):
        return result.dict()
    else:
        return {
            'compiled': getattr(result, 'compiled', False),
            'correctness': getattr(result, 'correctness', False),
            'metadata': getattr(result, 'metadata', {}),
            'runtime': getattr(result, 'runtime', -1.0),
            'runtime_stats': getattr(result, 'runtime_stats', {})
        }


# Define separate Modal functions for each GPU type at global scope
@app.function(
    image=image,
    gpu="L40S",
    timeout=3600,
    min_containers=1,  # Keep 1 container warm for faster subsequent calls
)
def evaluate_kernel_L40S(ref_arch_src: str, custom_cuda: str, verbose: bool = False):
    from src.eval import eval_kernel_against_ref
    from src.utils import set_gpu_arch
    set_gpu_arch(gpu_arch_mapping["L40S"])
    result = eval_kernel_against_ref(ref_arch_src, custom_cuda, verbose=verbose,
                                     measure_performance=True, num_correct_trials=5, num_perf_trials=100)
    return _process_eval_result(result)


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    min_containers=1,
)
def evaluate_kernel_H100(ref_arch_src: str, custom_cuda: str, verbose: bool = False):
    from src.eval import eval_kernel_against_ref
    from src.utils import set_gpu_arch
    set_gpu_arch(gpu_arch_mapping["H100"])
    result = eval_kernel_against_ref(ref_arch_src, custom_cuda, verbose=verbose,
                                     measure_performance=True, num_correct_trials=5, num_perf_trials=100)
    return _process_eval_result(result)


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    min_containers=1,
)
def evaluate_kernel_A100(ref_arch_src: str, custom_cuda: str, verbose: bool = False):
    from src.eval import eval_kernel_against_ref
    from src.utils import set_gpu_arch
    set_gpu_arch(gpu_arch_mapping["A100"])
    result = eval_kernel_against_ref(ref_arch_src, custom_cuda, verbose=verbose,
                                     measure_performance=True, num_correct_trials=5, num_perf_trials=100)
    return _process_eval_result(result)


@app.function(
    image=image,
    gpu="L4",
    timeout=3600,
    min_containers=1,
)
def evaluate_kernel_L4(ref_arch_src: str, custom_cuda: str, verbose: bool = False):
    from src.eval import eval_kernel_against_ref
    from src.utils import set_gpu_arch
    set_gpu_arch(gpu_arch_mapping["L4"])
    result = eval_kernel_against_ref(ref_arch_src, custom_cuda, verbose=verbose,
                                     measure_performance=True, num_correct_trials=5, num_perf_trials=100)
    return _process_eval_result(result)


@app.function(
    image=image,
    gpu="T4",
    timeout=3600,
    min_containers=1,
)
def evaluate_kernel_T4(ref_arch_src: str, custom_cuda: str, verbose: bool = False):
    from src.eval import eval_kernel_against_ref
    from src.utils import set_gpu_arch
    set_gpu_arch(gpu_arch_mapping["T4"])
    result = eval_kernel_against_ref(ref_arch_src, custom_cuda, verbose=verbose,
                                     measure_performance=True, num_correct_trials=5, num_perf_trials=100)
    return _process_eval_result(result)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    min_containers=1,
)
def evaluate_kernel_A10G(ref_arch_src: str, custom_cuda: str, verbose: bool = False):
    from src.eval import eval_kernel_against_ref
    from src.utils import set_gpu_arch
    set_gpu_arch(gpu_arch_mapping["A10G"])
    result = eval_kernel_against_ref(ref_arch_src, custom_cuda, verbose=verbose,
                                     measure_performance=True, num_correct_trials=5, num_perf_trials=100)
    return _process_eval_result(result)


# Map GPU types to their evaluator functions
evaluators = {
    "L40S": evaluate_kernel_L40S,
    "H100": evaluate_kernel_H100,
    "A100": evaluate_kernel_A100,
    "L4": evaluate_kernel_L4,
    "T4": evaluate_kernel_T4,
    "A10G": evaluate_kernel_A10G,
}


class LocalKernelEvaluatorWrapper:
    """
    Local wrapper for the persistent Modal evaluator.
    This class is used in the local code to interact with the Modal container.
    Modal will keep containers warm for faster subsequent evaluations.
    """

    def __init__(self, gpu: str = "L40S"):
        """
        Initialize the evaluator wrapper.

        Args:
            gpu: GPU type to use (e.g., "L40S", "H100", "A100")
        """
        if gpu not in evaluators:
            raise ValueError(f"Unsupported GPU type: {gpu}. Choose from: {list(evaluators.keys())}")

        self.gpu = gpu
        self._evaluator_func = evaluators[gpu]
        self._app_context = None

    def __enter__(self):
        """Context manager entry - starts the Modal app and sets up the evaluator."""
        # Start the Modal app context
        self._app_context = app.run()
        self._app_context.__enter__()

        print(f"Modal KernelEvaluator ready with GPU: {self.gpu}")
        print(f"(Modal will keep container warm for faster evaluations)")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes Modal app context."""
        if self._app_context:
            self._app_context.__exit__(exc_type, exc_val, exc_tb)
            self._app_context = None
        print(f"Evaluator context exited (Modal keeps {self.gpu} container warm)")
        return False

    def evaluate(self, ref_arch_src: str, custom_cuda: str, verbose: bool = False):
        """
        Evaluate a kernel using the persistent Modal container.

        Args:
            ref_arch_src: Reference kernel implementation
            custom_cuda: Custom CUDA kernel to evaluate
            verbose: Whether to print verbose output

        Returns:
            Dict with evaluation results
        """
        if self._app_context is None:
            raise RuntimeError("Evaluator not active. Use 'with LocalKernelEvaluatorWrapper() as evaluator:'")

        # Call the Modal function remotely (GPU arch is set in the function itself)
        return self._evaluator_func.remote(
            ref_arch_src,
            custom_cuda,
            verbose
        )


# Convenience function for quick testing
@app.local_entrypoint()
def test_persistent_evaluator():
    """Test the persistent evaluator setup."""
    from datasets import load_dataset

    ds = load_dataset("allenanie/kernelbench_with_prompts")
    examples = [ex for ex in ds['train'] if ex['backend'] == 'cuda']
    ex1 = examples[0]
    ref_arch_src = ex1['ref_arch_src']
    custom_cuda = open("level1_prob1_cuda_custom_cuda_gpt5_example.txt").read()

    print(f"\n{'='*60}")
    print(f"Testing Persistent Modal Evaluator")
    print(f"{'='*60}\n")

    # Get the L40S evaluator function
    evaluate_func = evaluators["L40S"]

    # Run multiple evaluations - Modal will keep the container warm
    for i in range(3):
        print(f"\nEvaluation {i+1}/3...")
        result = evaluate_func.remote(ref_arch_src, custom_cuda, False)
        print(f"Compiled: {result.get('compiled')}, Correctness: {result.get('correctness')}, "
              f"Runtime: {result.get('runtime', -1):.2f} ms")

    print(f"\n{'='*60}")
    print("All evaluations completed! Modal keeps container warm for subsequent calls.")
    print(f"{'='*60}\n")
