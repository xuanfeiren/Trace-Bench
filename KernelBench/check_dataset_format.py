"""Quick script to check the dataset format"""
from datasets import load_dataset

ds = load_dataset("allenanie/kernelbench_with_prompts")
examples = [ex for ex in ds['train'] if ex['backend'] == 'cuda']

# Print first example to see the format
ex = examples[0]
print("="*80)
print("TASK INPUT (prompt):")
print("="*80)
print(ex['input'][:500])
print("\n" + "="*80)
print("REF_ARCH_SRC (reference implementation):")
print("="*80)
print(ex['ref_arch_src'][:1000])
print("\n...")
