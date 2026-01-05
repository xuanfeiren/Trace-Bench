"""
Demo script showing how to use the KernelEvaluator wrapper.

This demonstrates the recommended pattern:
1. Initialize the evaluator once
2. Run multiple evaluations over time
3. Use batch mode for parallel evaluations
"""

from kernel_evaluator import KernelEvaluator
from datasets import load_dataset
import time


def demo_sequential_evaluations():
    """Demo: Run multiple evaluations sequentially with the same evaluator."""
    print("=" * 60)
    print("Demo 1: Sequential Evaluations")
    print("=" * 60)

    # Load sample data
    ds = load_dataset("allenanie/kernelbench_with_prompts")
    examples = [ex for ex in ds['train'] if ex['backend'] == 'cuda']
    ex1 = examples[0]

    ref_arch_src = ex1['ref_arch_src']
    custom_cuda = open("level1_prob1_cuda_custom_cuda_gpt5_example.txt").read()

    # Initialize evaluator once
    evaluator = KernelEvaluator(
        gpu="L40S",
        verbose=False,
        num_correct_trials=5,
        num_perf_trials=50  # Fewer trials for faster demo
    )

    # Run multiple evaluations over time
    print("\nRunning 3 evaluations sequentially...")
    for i in range(3):
        print(f"\n--- Evaluation {i+1} ---")
        result = evaluator.evaluate(ref_arch_src, custom_cuda)

        print(f"Compiled: {result['compiled']}")
        print(f"Correctness: {result['correctness']}")
        print(f"Runtime: {result['runtime']:.2f} ms")
        print(f"Hardware: {result['metadata']['hardware']}")

        # Simulate doing other work between evaluations
        if i < 2:
            print("Doing other work...")
            time.sleep(1)


def demo_batch_parallel_evaluation():
    """Demo: Run multiple evaluations in parallel using batch mode."""
    print("\n" + "=" * 60)
    print("Demo 2: Batch Parallel Evaluations")
    print("=" * 60)

    # Load sample data
    ds = load_dataset("allenanie/kernelbench_with_prompts")
    examples = [ex for ex in ds['train'] if ex['backend'] == 'cuda']
    ex1 = examples[0]

    ref_arch_src = ex1['ref_arch_src']
    custom_cuda = open("level1_prob1_cuda_custom_cuda_gpt5_example.txt").read()

    # Initialize evaluator
    evaluator = KernelEvaluator(
        gpu="L40S",
        verbose=False,
        num_perf_trials=50
    )

    # Prepare batch of evaluations (could be different kernels)
    eval_configs = [
        {"ref_arch_src": ref_arch_src, "custom_cuda": custom_cuda},
        {"ref_arch_src": ref_arch_src, "custom_cuda": custom_cuda},
        {"ref_arch_src": ref_arch_src, "custom_cuda": custom_cuda},
    ]

    print(f"\nRunning {len(eval_configs)} evaluations in parallel...")
    start_time = time.time()
    results = evaluator.evaluate_batch(eval_configs)
    elapsed = time.time() - start_time

    print(f"\nCompleted in {elapsed:.1f} seconds")
    print(f"Average runtime per eval: {elapsed/len(eval_configs):.1f}s (wall-clock time)\n")

    for i, result in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  Correctness: {result['correctness']}")
        print(f"  Runtime: {result['runtime']:.2f} ms")
        print(f"  Trials: {result['metadata']['correctness_trials']}")


def demo_custom_parameters():
    """Demo: Using custom parameters for different evaluation needs."""
    print("\n" + "=" * 60)
    print("Demo 3: Custom Parameters")
    print("=" * 60)

    # Load sample data
    ds = load_dataset("allenanie/kernelbench_with_prompts")
    examples = [ex for ex in ds['train'] if ex['backend'] == 'cuda']
    ex1 = examples[0]

    ref_arch_src = ex1['ref_arch_src']
    custom_cuda = open("level1_prob1_cuda_custom_cuda_gpt5_example.txt").read()

    # Initialize with default parameters
    evaluator = KernelEvaluator(gpu="L40S", verbose=False)

    # Quick evaluation (fewer trials)
    print("\nQuick evaluation (10 perf trials)...")
    result_quick = evaluator.evaluate(
        ref_arch_src,
        custom_cuda,
        num_perf_trials=10
    )
    print(f"Quick result: {result_quick['runtime']:.2f} ms")

    # Thorough evaluation (more trials)
    print("\nThorough evaluation (200 perf trials)...")
    result_thorough = evaluator.evaluate(
        ref_arch_src,
        custom_cuda,
        num_perf_trials=200
    )
    print(f"Thorough result: {result_thorough['runtime']:.2f} ms ± {result_thorough['runtime_stats']['std']:.2f}")


def demo_persistent_workflow():
    """Demo: Realistic workflow with persistent evaluator."""
    print("\n" + "=" * 60)
    print("Demo 4: Persistent Workflow")
    print("=" * 60)

    # Load sample data
    ds = load_dataset("allenanie/kernelbench_with_prompts")
    examples = [ex for ex in ds['train'] if ex['backend'] == 'cuda']
    ex1 = examples[0]

    ref_arch_src = ex1['ref_arch_src']
    custom_cuda = open("level1_prob1_cuda_custom_cuda_gpt5_example.txt").read()

    # Initialize once at the start of your workflow
    print("\nInitializing evaluator...")
    evaluator = KernelEvaluator(gpu="L40S", verbose=False, num_perf_trials=30)

    # Scenario: You're iteratively developing kernels
    kernel_versions = [custom_cuda, custom_cuda, custom_cuda]  # In reality, these would be different

    for i, kernel_code in enumerate(kernel_versions):
        print(f"\n--- Testing kernel version {i+1} ---")
        result = evaluator.evaluate(ref_arch_src, kernel_code)

        if result['correctness']:
            print(f"✓ Passed correctness tests")
            print(f"  Runtime: {result['runtime']:.2f} ms")
        else:
            print(f"✗ Failed correctness tests")

        # Simulate development time between iterations
        if i < len(kernel_versions) - 1:
            print("  (Optimizing kernel...)")
            time.sleep(1)

    print("\nWorkflow complete!")


if __name__ == "__main__":
    import sys

    # Check if a specific demo is requested
    if len(sys.argv) > 1:
        demo_name = sys.argv[1]
        demos = {
            "sequential": demo_sequential_evaluations,
            "batch": demo_batch_parallel_evaluation,
            "custom": demo_custom_parameters,
            "persistent": demo_persistent_workflow,
        }
        if demo_name in demos:
            demos[demo_name]()
        else:
            print(f"Unknown demo: {demo_name}")
            print(f"Available demos: {', '.join(demos.keys())}")
    else:
        # Run all demos
        demo_sequential_evaluations()
        demo_batch_parallel_evaluation()
        demo_custom_parameters()
        demo_persistent_workflow()

        print("\n" + "=" * 60)
        print("All demos completed!")
        print("=" * 60)
