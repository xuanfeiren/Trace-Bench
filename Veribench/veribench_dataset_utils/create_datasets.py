"""
Dataset utilities for VeriBench local dataset loading.

This module provides a function to create datasets from the local dataset directory,
matching the interface from solution_PS.py (lines 76-91).
"""

import json
from pathlib import Path
from typing import Dict, List, Any


def create_single_task_dataset(task_idx: int) -> Dict[str, List[Any]]:
    """
    Create a dataset with a single task for PrioritySearch or similar training.
    
    Loads data from the dataset/task_{task_idx}.json files.
    Matches the interface from solution_PS.py (lines 76-91).
    
    Args:
        task_idx: Index of the task to load (0-139)
        
    Returns:
        Dictionary with:
        - 'inputs': List containing the Python code
        - 'infos': List containing the task index
        
    Example:
        >>> dataset = create_single_task_dataset(0)
        >>> print(f"Python code length: {len(dataset['inputs'][0])}")
        >>> print(f"Task ID: {dataset['infos'][0]}")
    """
    # Get dataset directory path
    script_dir = Path(__file__).resolve().parent
    dataset_dir = script_dir / "dataset"
    task_file = dataset_dir / f"task_{task_idx}.json"
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found at {dataset_dir}")
    
    if not task_file.exists():
        raise FileNotFoundError(f"Task file not found: {task_file}")
    
    if task_idx < 0 or task_idx >= 140:
        raise ValueError(f"Task index {task_idx} out of range (0-139)")
    
    # Load task data from JSON file
    with open(task_file, 'r', encoding='utf-8') as f:
        task_data = json.load(f)
    
    python_code = task_data['python_code']
    
    return {
        'inputs': [python_code],
        'infos': [task_idx]
    }


def load_task_full(task_idx: int) -> Dict[str, Any]:
    """
    Load complete task data including Python code and gold Lean4 code.
    
    Args:
        task_idx: Index of the task to load (0-139)
        
    Returns:
        Dictionary with:
        - 'task_id': int
        - 'python_code': str
        - 'gold_reference_lean4_code': str or None
        
    Example:
        >>> task = load_task_full(0)
        >>> print(f"Task ID: {task['task_id']}")
        >>> print(f"Python code: {len(task['python_code'])} chars")
        >>> print(f"Lean code: {len(task['gold_reference_lean4_code'])} chars")
    """
    script_dir = Path(__file__).resolve().parent
    dataset_dir = script_dir / "dataset"
    task_file = dataset_dir / f"task_{task_idx}.json"
    
    if not task_file.exists():
        raise FileNotFoundError(f"Task file not found: {task_file}")
    
    with open(task_file, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    pass
    # Test the function
    # print("Testing create_single_task_dataset()...")
    
    # for task_id in range(140):
    #     dataset = create_single_task_dataset(task_id)
    #     print(f"Task {task_id} loaded successfully")
    #     print(f"  Python code: {len(dataset['inputs'][0])} chars")
    #     print(f"  Task ID: {dataset['infos'][0]}")
        
    #     # Compare with solution_PS.py if available
        
    #     import sys
    #     sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    #     from my_processing_agents.solution_PS import create_single_task_dataset as create_single_task_dataset_PS
        
    #     dataset_PS = create_single_task_dataset_PS(task_id)
    #     match = dataset_PS['inputs'][0] == dataset['inputs'][0]
    #     print(f"\n✅ Comparison with solution_PS.py: {'MATCH' if match else 'MISMATCH'}")
    



