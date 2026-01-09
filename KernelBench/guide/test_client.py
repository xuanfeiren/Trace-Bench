import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cuda_eval_client import CUDAEvalClient

client = CUDAEvalClient("http://localhost:6000")
problem_id = 1 
sample_id = 0
from datasets import load_dataset
ds = load_dataset("allenanie/kernelbench_with_prompts")
examples = [ex for ex in ds['train'] if ex['backend'] == 'cuda']
ex1 = examples[0]
ref_arch_src = ex1['ref_arch_src']
custom_cuda = open("level1_prob1_cuda_custom_cuda_gpt5_example.txt").read()

job_id = client.submit_evaluation(
            problem_id=problem_id, sample_id=sample_id,
            custom_cuda=custom_cuda,
            ref_arch_src=ref_arch_src,
            level=1
)

result = client.wait_for_job(job_id, timeout=300)
kernel_exec_result = result['result']
