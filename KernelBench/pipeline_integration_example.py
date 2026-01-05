"""
Example showing how to integrate KernelEvaluator into your pipeline.

For Modal to work, you have two options:
1. Use `modal run` on your pipeline script (recommended)
2. Wrap your pipeline in a @app.local_entrypoint() decorator

This example shows option 1.
"""

from evaluate_with_modal import app, eval_single_sample_modal, gpu_arch_mapping
from datasets import load_dataset


@app.local_entrypoint()
def run_pipeline():
    """
    Your pipeline code goes here.
    This decorator tells Modal to run this function locally while
    enabling remote function calls.
    """
    print("Loading test data...")
    ds = load_dataset("allenanie/kernelbench_with_prompts")
    examples = [ex for ex in ds['train'] if ex['backend'] == 'cuda']
    ex1 = examples[0]

    ref_arch_src = ex1['ref_arch_src']
    custom_cuda = open("level1_prob1_cuda_custom_cuda_gpt5_example.txt").read()

    gpu = "L40S"
    gpu_arch = gpu_arch_mapping[gpu]

    # Now you can call the remote function directly in your pipeline
    print("\nRunning evaluation 1...")
    result1 = eval_single_sample_modal.remote(
        ref_arch_src, custom_cuda, False, gpu_arch,
        num_correct_trials=5, num_perf_trials=50
    )

    print(f"Result 1: correctness={result1['correctness']}, runtime={result1['runtime']:.2f}ms")

    # Run another evaluation
    print("\nRunning evaluation 2...")
    result2 = eval_single_sample_modal.remote(
        ref_arch_src, custom_cuda, False, gpu_arch,
        num_correct_trials=5, num_perf_trials=50
    )

    print(f"Result 2: correctness={result2['correctness']}, runtime={result2['runtime']:.2f}ms")

    # You can also run multiple in parallel
    print("\nRunning 3 evaluations in parallel...")
    futures = []
    for i in range(3):
        future = eval_single_sample_modal.spawn(
            ref_arch_src, custom_cuda, False, gpu_arch,
            num_correct_trials=5, num_perf_trials=50
        )
        futures.append(future)

    # Collect results
    results = [f.get() for f in futures]
    print(f"Parallel results: {[r['runtime'] for r in results]}")

    return results


# To run this: modal run pipeline_integration_example.py
