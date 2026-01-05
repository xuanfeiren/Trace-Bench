from datasets import load_dataset


def create_single_task_dataset(task_idx: int):
    ds = load_dataset("allenanie/kernelbench_with_prompts")
    examples = [ex for ex in ds['train'] if ex['backend'] == 'cuda']
    task = examples[task_idx]
    return {
        'inputs': [task['input']],
        'infos': [task['ref_arch_src']]
    }

def load_entire_cuda_dataset():
    ds = load_dataset("allenanie/kernelbench_with_prompts")
    examples = [ex for ex in ds['train'] if ex['backend'] == 'cuda' and ex['level'] == 1]
    # after this filtering, I only keep input and ref_arch_src
    return [{'input': ex['input'], 'ref_arch_src': ex['ref_arch_src']} for ex in examples]

def create_matrix_multiplication_dataset():
    """
    Create a dataset of matrix multiplication tasks. This dataset contains 16 matrix multiplication tasks.
    """
    ds = load_dataset("allenanie/kernelbench_with_prompts")
    examples = [ex for ex in ds['train'] if ex['backend'] == 'cuda' and ex['level'] == 1]\

    # get all the examples that contain 'matrix multiplication' in the ref_arch_src
    examples = [ex for ex in examples if 'matrix multiplication' in ex['ref_arch_src']]

    # after this filtering, I only keep input and ref_arch_src
    return [{'input': ex['input'], 'ref_arch_src': ex['ref_arch_src']} for ex in examples]
    
def main():
    ds = create_matrix_multiplication_dataset()
    print(len(ds))
    # print all ref_arch_src
    # for ex in ds:
    #     print(ex['ref_arch_src'])
    #     print("-"*100)
    # get the 13th example
    print(ds[13]['ref_arch_src'])

if __name__ == "__main__":
    main()