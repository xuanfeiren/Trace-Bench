#!/usr/bin/env python3
"""
Simple agent to solve Veribench tasks with one LLM call using Gemini.

max_tokens parameter controls the maximum number of output tokens the model can generate.
Gemini model limits:
- gemini-2.0-flash: 8,192 tokens
- gemini-2.5-flash: 65,535 tokens  
- gemini-2.5-flash-lite: 65,536 tokens
"""

import os
import litellm
from datasets import load_dataset

def solve_veribench_task(task_index=0):
    """Solve a Veribench task with one LLM call."""
    
    # Load dataset and get task
    dataset = load_dataset("allenanie/veribench_with_prompts")
    task = dataset['train'][task_index]
    
    # Build separate system and user prompts
    system_prompt = task.get("system_prompt", "")
    user_query = task.get("user_query", "")
    
    # Create proper message structure
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    
    # Call Gemini with maximum output tokens (65,536 for gemini-2.5-flash-lite)
    response = litellm.completion(
        model="gemini/gemini-2.5-flash-lite",
        messages=messages,
        max_tokens=65536,  # Maximum for gemini-2.5-flash-lite
        temperature=0.1
    )
    
    # Print results
    print(f"Task: {task['filename']}")
    print(f"Category: {task['category']}")
    print(f"System prompt length: {len(system_prompt)} chars")
    print(f"User query length: {len(user_query)} chars")
    print("\nSolution:")
    print("-" * 40)
    print(response.choices[0].message.content)

if __name__ == "__main__":
    solve_veribench_task()