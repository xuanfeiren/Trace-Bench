"""
Simplest test: Just test the guide with the known-good example code
"""
import sys
import os

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets import load_dataset

# Import the guide and modal evaluator from kernel_PS
from kernel_PS import KernelGuide
from modal_kernel_evaluator import LocalKernelEvaluatorWrapper


def main():
    # Load the KernelBench dataset
    print("Loading KernelBench dataset...")
    ds = load_dataset("allenanie/kernelbench_with_prompts")
    examples = [ex for ex in ds['train'] if ex['backend'] == 'cuda']

    # Get first task
    task = examples[0]
    task_prompt = task['input']
    ref_arch_src = task['ref_arch_src']

    # Load the known-good example code
    example_code_path = os.path.join(project_root, "level1_prob1_cuda_custom_cuda_gpt5_example.txt")
    print(f"Loading example code from: {example_code_path}")
    with open(example_code_path, 'r') as f:
        custom_cuda = f.read()

    print(f"Example code loaded: {len(custom_cuda)} characters\n")

    # Initialize Modal container and guide
    print("Initializing Modal GPU evaluator (L40S)...")
    with LocalKernelEvaluatorWrapper(gpu='L40S') as modal_evaluator:
        guide = KernelGuide(modal_evaluator=modal_evaluator, verbose=True)
        print("Modal container ready!\n")

        print("="*80)
        print("Testing guide with known-good example code")
        print("="*80)

        # Evaluate using the guide
        print("\nEvaluating kernel code...")
        score, feedback = guide.get_feedback(
            task=task_prompt,
            response=custom_cuda,
            info=ref_arch_src
        )

        # Print results
        print(f"\n{'='*80}")
        print("RESULTS")
        print(f"{'='*80}")
        print(f"Score: {score:.6f}")
        print(f"\nFeedback:\n{feedback}")
        print(f"{'='*80}\n")

    print("Test completed!")


if __name__ == "__main__":
    main()
