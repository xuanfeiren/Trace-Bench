"""
Test the modal evaluator directly with Gemini generated code
"""
import sys
import os

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datasets import load_dataset
from modal_kernel_evaluator import evaluators

# Load dataset
ds = load_dataset("allenanie/kernelbench_with_prompts")
examples = [ex for ex in ds['train'] if ex['backend'] == 'cuda']
ex1 = examples[0]
ref_arch_src = ex1['ref_arch_src']

# Load Gemini generated code
generated_code_path = "my_process_agents/generated_task_0_debug.py"
with open(generated_code_path, 'r') as f:
    custom_cuda = f.read()

print("Testing modal evaluator directly with .remote()")
print(f"Reference code length: {len(ref_arch_src)}")
print(f"Custom code length: {len(custom_cuda)}")
print()

# Get the L40S evaluator function and call it directly
evaluate_func = evaluators["L40S"]

print("Calling evaluate_func.remote()...")
result = evaluate_func.remote(ref_arch_src, custom_cuda, True)

print("\n" + "="*80)
print("RESULT:")
print("="*80)
print(f"Compiled: {result.get('compiled')}")
print(f"Correctness: {result.get('correctness')}")
print(f"Runtime: {result.get('runtime', -1):.2f} ms")
print(f"Metadata: {result.get('metadata')}")
print("="*80)
