"""
This file modifies https://github.com/ScalingIntelligence/KernelBench/blob/main/scripts/generate_and_eval_single_sample_modal.py
Original authors: William Hu, Anne Ouyang, Simon Guo

Note: Triton evaluation is not supported due to KernelBench's original code
"""

import os, sys
import json
import modal

from datasets import load_dataset

app = modal.App("trace_bench_kernel")  # Changed name to force cache invalidation

# NOTE: this assumes you have run `install.sh` inside this repo so that KernelBench is stored and built under `./external/KernelBench`
REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

REPO_EXTERNAL_LIB_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../external/",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_EXTERNAL_LIB_PATH, "KernelBench")

gpu_arch_mapping = {"L40S": ["Ada"], "H100": ["Hopper"], "A100": ["Ampere"], "L4": ["Ada"], "T4": ["Turing"],
                    "A10G": ["Ampere"]}
cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install("git",
                 "gcc-10",
                 "g++-10",
                 "clang"  # note i skip a step
                 )
    .pip_install(  # required to build flash-attn
        "anthropic",
        "numpy",
        "openai",
        "packaging",
        "pydra_config",
        "torch==2.5.0",
        "tqdm",
        "datasets",
        "transformers",
        "google-generativeai",
        "together",
        "pytest",
        "ninja",
        "utils",
        "dotenv",
        "triton",
        "litellm"
    ).env({
        "PYTHONPATH": "/root/KernelBench"  # add KernelBench to python path
    }).add_local_dir(
        KERNEL_BENCH_PATH,
        remote_path="/root/KernelBench",
        # Ignore large/unnecessary files to prevent timeout
        ignore=[
            "**/.git",
            "**/__pycache__",
            "**/*.pyc",
            "**/.pytest_cache",
            "**/build",
            "**/dist",
            "**/.DS_Store",
            "**/node_modules",
            "**/.venv",
            "**/venv",
            "**/*.egg-info"
        ]
    ).add_local_file(
        os.path.join(os.path.dirname(__file__), "level1_prob1_cuda_custom_cuda_gpt5_example.txt"),
        remote_path="/root"
    )
)


def _eval_in_process(ref_arch_src, custom_cuda, verbose, gpu_arch, num_correct_trials, num_perf_trials):
    """
    Wrapper function to run evaluation in an isolated subprocess.
    This prevents CUDA context corruption from affecting future evaluations.
    """
    import sys
    import os
    from io import StringIO
    from src.eval import eval_kernel_against_ref
    from src.utils import set_gpu_arch

    set_gpu_arch(gpu_arch)

    # Suppress output if not verbose
    if not verbose:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        try:
            result = eval_kernel_against_ref(
                ref_arch_src, custom_cuda, verbose=False, measure_performance=True,
                num_correct_trials=num_correct_trials,
                num_perf_trials=num_perf_trials
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    else:
        result = eval_kernel_against_ref(
            ref_arch_src, custom_cuda, verbose=True, measure_performance=True,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials
        )

    # Convert Pydantic model to dict before returning from subprocess
    return result.model_dump() if hasattr(result, 'model_dump') else result.dict()


@app.function(image=image, timeout=3600, gpu="L40S", min_containers=1)
def eval_single_sample_modal(ref_arch_src, custom_cuda, verbose, gpu_arch, num_correct_trials=5, num_perf_trials=100):
    """
    Evaluate kernel in an isolated subprocess to prevent CUDA context corruption.

    Each evaluation runs in a fresh process, so CUDA errors (like illegal memory access,
    XID faults, etc.) from buggy kernels don't contaminate future evaluations.
    """
    import multiprocessing as mp

    # Run evaluation in isolated subprocess with timeout
    # This ensures CUDA context is fully cleaned up after each evaluation
    with mp.Pool(1) as pool:
        try:
            result = pool.apply_async(
                _eval_in_process,
                args=(ref_arch_src, custom_cuda, verbose, gpu_arch, num_correct_trials, num_perf_trials),
            ).get(timeout=3500)  # 3500s timeout (slightly less than function timeout)
        except mp.TimeoutError:
            print(f"[WARNING] Evaluation timed out after 3500 seconds")
            return {
                'compiled': False,
                'correctness': False,
                'runtime': -1.0,
                'runtime_stats': {},
                'metadata': {'error': 'Evaluation timed out'}
            }
        except Exception as e:
            import traceback
            full_traceback = traceback.format_exc()
            print(f"[ERROR] Evaluation failed with exception: {e}")
            print(full_traceback)
            return {
                'compiled': False,
                'correctness': False,
                'runtime': -1.0,
                'runtime_stats': {},
                'metadata': {
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'traceback': full_traceback
                }
            }

    return result


def quick_test_on_setup():
    """
    This allows you to test if your modal AI account setup is fully correct.
    Should take about ~1m to finish.
    :return:
    """
    ds = load_dataset("allenanie/kernelbench_with_prompts")
    examples = [ex for ex in ds['train'] if ex['backend'] == 'cuda']
    ex1 = examples[0]
    ref_arch_src = ex1['ref_arch_src']
    # custom_cuda = open("level1_prob1_cuda_custom_cuda_gpt5_example.txt").read()
    custom_cuda = open("my_process_agents/error_response.txt").read()
    gpu = "L40S"
    verbose = True

    # first try

    kernel_exec_result = eval_single_sample_modal.remote(ref_arch_src, custom_cuda, False, gpu_arch_mapping[gpu])
    print(f"Evaluation result 1:")
    print(kernel_exec_result)

    # second try
    kernel_exec_result = eval_single_sample_modal.remote(ref_arch_src, custom_cuda, False, gpu_arch_mapping[gpu])
    print(f"Evaluation result 2:")
    print(kernel_exec_result)

    kernel_exec_result = eval_single_sample_modal.remote(ref_arch_src, custom_cuda, False, gpu_arch_mapping[gpu])
    print(f"Evaluation result 3:")
    print(kernel_exec_result)




@app.local_entrypoint()
def main():
    quick_test_on_setup()


if __name__ == '__main__':
    pass
