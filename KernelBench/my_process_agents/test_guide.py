from kernel_PS_modal import KernelGuide
from evaluate_with_modal import app
from datasets import load_dataset

guide = KernelGuide(gpu="L40S", verbose=False)


@app.local_entrypoint()
def test_guide():
    ds = load_dataset("allenanie/kernelbench_with_prompts")
    examples = [ex for ex in ds['train'] if ex['backend'] == 'cuda']
    ex1 = examples[0]
    ref_arch_src = ex1['ref_arch_src']
    # custom_cuda = open("my_process_agents/level1_prob1_cuda_custom_cuda_gpt5_example.txt").read()
    custom_cuda = open("my_process_agents/error_response.txt").read()
    score, feedback = guide.get_feedback(
        task=ex1['input'],
        response=custom_cuda,
        info=ref_arch_src
    )

    # print(score, feedback)

    # test again with the same input and response
    # score, feedback = guide.get_feedback(
    #     task=ex1['input'],
    #     response="# Dummy kernel code",
    #     info=ref_arch_src
    # )

    # print(score, feedback)

    # test again with the same input and response
    # score, feedback = guide.get_feedback(
    #     task=ex1['input'],
    #     response=custom_cuda,
    #     info=ref_arch_src
    # )

    # print(score, feedback)


