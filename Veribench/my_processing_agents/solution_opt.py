# Train agent using PrioritySearch algorithm on Veribench

# set np and torch seeds
import numpy as np
import torch
np.random.seed(10)
torch.manual_seed(10)

import os
import re
import json
import time
import argparse
from typing import Union, Optional, Dict, Any, List

import opto
from opto import trace
from opto.trace.nodes import GRAPH
from opto.trace.modules import Module
from opto.utils.llm import LLM
from opto.optimizers.utils import print_color
from opto.trainer.guide import Guide
from opto.trainer.loggers import WandbLogger, DefaultLogger
from opto.trainer.utils import async_run
from opto.optimizers import OptoPrimeV2

from lean_interpretor import lean_interpreter
from system_prompts import SYSTEM_PROMPT, EXAMPLES

import litellm
litellm.drop_params = True
litellm.suppress_debug_info = True


os.environ["TRACE_LITELLM_MODEL"] = "gemini/gemini-2.5-flash-lite"
from optimize_veribench_agent import VeribenchGuide


def load_single_task(task_idx: int = 0) -> Dict[str, Any]:
    """
    Load a single task from Veribench dataset.
    
    Args:
        task_idx: Index of the task to load
        
    Returns:
        Dictionary containing 'user_query' and 'task_id'
    """
    from datasets import load_dataset
    
    # Load dataset from HuggingFace: https://huggingface.co/datasets/allenanie/veribench_with_prompts
    dataset = load_dataset("allenanie/veribench_with_prompts")
    task = dataset['train'][task_idx]
    
    return {
        'user_query': task['user_query'],
        'task_id': task_idx
    }


def get_initial_lean_code(user_query: str, model: str = "gemini/gemini-2.5-flash-lite") -> str:
    """
    Call LLM to get initial Lean code from the user query.
    
    Args:
        user_query: The task/user query from the dataset
        model: The LLM model to use
        
    Returns:
        Extracted Lean code from the LLM response
    """
    llm = LLM(model=model)
    
    # Construct the prompt using system prompt and examples
    system_content = SYSTEM_PROMPT + "\n\n" + EXAMPLES
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_query}
    ]
    
    response = llm(messages=messages, max_tokens=65536)
    response_text = response.choices[0].message.content
    
    # Extract the lean code from the response
    lean_pattern = r'```lean\s*\n(.*?)\n```'
    try:
        matches = re.findall(lean_pattern, response_text, re.DOTALL)
        if matches:
            return matches[0].strip()
        else:
            print_color("Warning: Could not extract Lean code from response. Using full response.", "yellow")
            return response_text
    except Exception as e:
        print_color(f"Error extracting lean code: {e}", "red")
        raise e


def main():
    parser = argparse.ArgumentParser(description='Optimize a single Lean solution using feedback loop')
    parser.add_argument('--task_idx', type=int, default=0, help='Task index from the Veribench dataset')
    parser.add_argument('--epoch', type=int, default=10, help='Maximum number of optimization epochs')
    parser.add_argument('--model', type=str, default='gemini/gemini-2.5-flash-lite', help='Model to use for LLM calls')
    args = parser.parse_args()

    epoch = args.epoch
    task_idx = args.task_idx
    model = args.model
    
    # Step 1: Load a task from the Veribench dataset
    # Dataset: https://huggingface.co/datasets/allenanie/veribench_with_prompts/viewer/default/train
    print(f"Loading task {task_idx} from Veribench dataset...")
    task = load_single_task(task_idx)
    user_query = task['user_query']
    print(f"Task loaded successfully. Task ID: {task['task_id']}")
    
    # Step 2: Call LLM to get the initial Lean code
    print(f"Generating initial Lean code using {model}...")
    initial_lean_code = get_initial_lean_code(user_query, model=model)
    print("Initial Lean code generated successfully.")
    print("-" * 50)
    print(initial_lean_code[:500] + "..." if len(initial_lean_code) > 500 else initial_lean_code)
    print("-" * 50)
    
    # Step 3: Initialize the lean code node as a trainable parameter
    lean_code = trace.node(initial_lean_code, trainable=True)
    
    # Step 4: Initialize the optimizer 
    optimizer = OptoPrimeV2([lean_code], max_tokens=25000, initial_var_char_limit=10000)
    optimizer.objective = """You are optimizing Lean 4 code to make it compile without errors.

CONTEXT:
- The variable contains Lean 4 code that needs to compile successfully
- The feedback contains compilation results: either success or error messages

YOUR TASK:
Analyze the compilation errors in the feedback and fix the Lean 4 code.

STRATEGY:
1. If feedback says "correct": The code is done, no changes needed
2. If feedback contains errors:
   - Identify the ROOT CAUSE of each error from the error message
   - Determine what Lean 4 syntax or logic is incorrect
   - Fix the specific issues in the code
   - Ensure type annotations are correct
   - Ensure all imports are valid

OUTPUT: Return the complete fixed Lean 4 code."""

    # Step 5: Initialize the guide
    guide = VeribenchGuide()
    
    # Step 6: Use the optimizer to optimize the lean code until the lean code is correct
    print(f"\nStarting optimization loop (max {epoch} epochs)...")
    for i in range(epoch):
        print(f"\n{'='*50}")
        print(f"Training Epoch {i + 1}/{epoch}")
        print(f"{'='*50}")
        
        # Get feedback from the guide (evaluates the lean code using lean_interpreter)
        score, feedback = guide.get_feedback(task=user_query, response=lean_code.data, info=None)
        
        print(f"Score: {score}")
        if score < 1.0:
            print(f"Feedback (truncated): {feedback[:300]}..." if len(feedback) > 300 else f"Feedback: {feedback}")
        else:
            print(f"Feedback: {feedback}")
        
        # Check if the lean code is correct
        if score == 1.0:
            print(f"\n{'*'*50}")
            print("SUCCESS! Lean code compiled correctly!")
            print(f"{'*'*50}")
            break
        
        # Perform optimization step
        optimizer.zero_feedback()
        optimizer.backward(lean_code, feedback)
        optimizer.step()
        
        print(f"Optimization step completed. Updated lean code.")
    else:
        print(f"\nReached maximum epochs ({epoch}). Final score: {score}")
    
    # Print final result
    print(f"\n{'='*50}")
    print("FINAL LEAN CODE:")
    print(f"{'='*50}")
    print(lean_code.data)


if __name__ == "__main__":
    main()
