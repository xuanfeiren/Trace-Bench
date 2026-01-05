"""
Simple test file to evaluate kernel generation using gemini-2.0-flash with LiteLLM.
Tests the first 5 tasks in the KernelBench dataset.
"""
import sys
import os

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets import load_dataset
import litellm

# Import the guide and modal evaluator from kernel_PS
from kernel_PS import KernelGuide
from modal_kernel_evaluator import LocalKernelEvaluatorWrapper

litellm.drop_params = True
litellm.suppress_debug_info = True


def generate_kernel_code(task_prompt: str, model: str = "gemini/gemini-2.0-flash") -> str:
    """
    Generate kernel code using LiteLLM with the specified model.

    Args:
        task_prompt: The task description
        model: The model to use (default: gemini-2.0-flash)

    Returns:
        Generated kernel code as a string
    """
    messages = [
        {
            "role": "user",
            "content": task_prompt
        }
    ]

    response = litellm.completion(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=4096
    )
    response_text = response.choices[0].message.content

    # Extract code from markdown code blocks if present
    # Try different code block formats
    if "```python" in response_text:
        # Extract everything between ```python and ```
        parts = response_text.split("```python")
        if len(parts) > 1:
            code_section = parts[1].split("```")[0]
            return code_section.strip()
    elif "```" in response_text:
        # Generic code block without language specifier
        parts = response_text.split("```")
        if len(parts) > 1:
            return parts[1].strip()

    # If no code blocks, return the entire response
    # (LLM might have returned raw code without markdown formatting)
    return response_text.strip()


@app.local_entrypoint()
def kernel_PS_train():
    # Load the KernelBench dataset
    print("Loading KernelBench dataset...")
    ds = load_dataset("allenanie/kernelbench_with_prompts")
    examples = [ex for ex in ds['train'] if ex['backend'] == 'cuda']

    # Test only the first 5 tasks
    num_tasks = min(5, len(examples))
    print(f"\nTesting the first {num_tasks} tasks with gemini-2.0-flash\n")

    # Initialize Modal container and guide
    print("Initializing Modal GPU evaluator (L40S)...")
    with LocalKernelEvaluatorWrapper(gpu='L40S') as modal_evaluator:
        guide = KernelGuide(modal_evaluator=modal_evaluator, verbose=True)
        print("Modal container ready!\n")

        # Process each task
        for idx in range(num_tasks):
            task = examples[idx]
            task_prompt = task['input']
            ref_arch_src = task['ref_arch_src']

            print(f"{'='*80}")
            print(f"Task {idx + 1}/{num_tasks}")
            print(f"{'='*80}")
            print(f"Prompt: {task_prompt[:200]}..." if len(task_prompt) > 200 else f"Prompt: {task_prompt}")
            print()

            # Generate kernel code using gemini-2.0-flash
            print("Generating kernel code with gemini-2.0-flash...")
            try:
                generated_code = generate_kernel_code(task_prompt)
                print(f"Generated code length: {len(generated_code)} characters")

                # Save generated code for debugging
                debug_file = f"my_process_agents/generated_task_{idx}_debug.py"
                with open(debug_file, 'w') as f:
                    f.write(generated_code)
                print(f"Saved generated code to: {debug_file}")
                print(f"First 300 chars:\n{generated_code[:300]}...")
                print()

                # Evaluate using the guide
                print("Evaluating kernel code...")
                score, feedback = guide.get_feedback(
                    task=task_prompt,
                    response=generated_code,
                    info=ref_arch_src
                )

                # Print results
                print(f"\n{'='*80}")
                print(f"RESULTS FOR TASK {idx + 1}")
                print(f"{'='*80}")
                print(f"Score: {score:.6f}")
                print(f"\nFeedback:\n{feedback}")
                print(f"{'='*80}\n")

            except Exception as e:
                print(f"\n{'='*80}")
                print(f"ERROR PROCESSING TASK {idx + 1}")
                print(f"{'='*80}")
                print(f"Error: {e}")
                print(f"{'='*80}\n")
                import traceback
                traceback.print_exc()

            print()

    print("Test completed!")


if __name__ == "__main__":
    pass
