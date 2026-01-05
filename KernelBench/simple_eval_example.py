"""
Simple example showing how to use KernelEvaluator in your pipeline.
Run with: python simple_eval_example.py
"""

from kernel_evaluator import KernelEvaluator
from datasets import load_dataset


def main():
    print("Loading test data...")
    ds = load_dataset("allenanie/kernelbench_with_prompts")
    examples = [ex for ex in ds['train'] if ex['backend'] == 'cuda']
    ex1 = examples[0]

    ref_arch_src = ex1['ref_arch_src']
    custom_cuda = open("level1_prob1_cuda_custom_cuda_gpt5_example.txt").read()

    # Initialize evaluator - Modal handles app lifecycle automatically
    print("\nInitializing KernelEvaluator...")
    evaluator = KernelEvaluator(
        gpu="L40S",
        verbose=False,
        num_perf_trials=50  # Adjust based on your needs
    )

    # Single evaluation
    print("\nRunning evaluation...")
    result = evaluator.evaluate(ref_arch_src, custom_cuda)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Compiled:    {result['compiled']}")
    print(f"Correctness: {result['correctness']}")
    print(f"Runtime:     {result['runtime']:.2f} ms")
    print(f"Std Dev:     {result['runtime_stats']['std']:.3f} ms")
    print(f"Hardware:    {result['metadata']['hardware']}")
    print(f"Trials:      {result['metadata']['correctness_trials']}")
    print("=" * 60)

    # You can run more evaluations with the same evaluator
    print("\nRunning another evaluation...")
    result2 = evaluator.evaluate(ref_arch_src, custom_cuda)
    print(f"Second run runtime: {result2['runtime']:.2f} ms")

    return result


if __name__ == "__main__":
    result = main()
