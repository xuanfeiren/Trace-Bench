import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guide.evaluate import evaluate
from datasets import load_dataset

# def evaluate(**kwargs):
#     print("Evaluating...")
#     pass

ds = load_dataset("allenanie/kernelbench_with_prompts")
examples = [ex for ex in ds['train'] if ex['backend'] == 'cuda']
ex1 = examples[0]
ref_arch_src = ex1['ref_arch_src']
custom_cuda = open("level1_prob1_cuda_custom_cuda_gpt5_example.txt").read()

result_dict = evaluate(
    ref_arch_src=ref_arch_src,
    custom_cuda=custom_cuda,
    num_correct_trials=5,
    num_perf_trials=100
)
score = result_dict['score']
feedback = result_dict['feedback']
print(f"Score: {score:}")
print(f"Feedback: {feedback}")
