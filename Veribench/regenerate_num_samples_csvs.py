#!/usr/bin/env python3
"""
Regenerate result_num_samples CSV files for PS algorithms.
For each lean4_tool_call_num from 0 to 50, calculates the mean score
across all 140 tasks using forward fill.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List


def load_results_json(json_path: Path) -> Dict:
    """Load results JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def get_score_at_lean4_calls(task_steps: List[Dict], target_calls: int) -> float:
    """
    Get the score for a task at a given lean4_tool_call_num using forward fill.
    
    Args:
        task_steps: List of step dictionaries for a task
        target_calls: Target lean4_tool_call_num
    
    Returns:
        Score at target_calls (or forward-filled score)
    """
    if target_calls == 0:
        return 0.0
    
    # Find the step with the largest lean4_tool_call_num <= target_calls
    best_score = 0.0
    for step in task_steps:
        calls = step.get('lean4_tool_call_num', 0)
        if calls <= target_calls:
            score = step.get('score', 0.0)
            # Use the maximum score seen so far (best score up to this point)
            best_score = max(best_score, score)
    
    return best_score


def generate_num_samples_csv(folder_path: Path, run_num: int, output_path: Path):
    """
    Generate result_num_samples_{run_num}.csv for a given folder and run.
    
    Args:
        folder_path: Path to folder containing results JSON files
        run_num: Run number (1, 2, or 3)
        output_path: Path where CSV will be written
    """
    json_file = folder_path / f"results_{run_num}.json"
    
    if not json_file.exists():
        print(f"  ⚠️  Warning: {json_file} not found, skipping...")
        return
    
    print(f"\n📊 Processing {folder_path.name}/results_{run_num}.json")
    
    # Load JSON data
    data = load_results_json(json_file)
    
    # Get all task keys (should be task_0 to task_139)
    task_keys = sorted(data.keys(), key=lambda x: int(x.split('_')[1]))
    num_tasks = len(task_keys)
    print(f"  Found {num_tasks} tasks")
    
    # Process each lean4_tool_call_num from 0 to 50
    max_calls = 50
    mean_scores = []
    
    for target_calls in range(max_calls + 1):
        task_scores = []
        
        for task_key in task_keys:
            task_steps = data[task_key]
            score = get_score_at_lean4_calls(task_steps, target_calls)
            task_scores.append(score)
        
        # Calculate mean across all tasks
        mean_score = np.mean(task_scores)
        mean_scores.append(mean_score)
    
    # Write CSV file
    with open(output_path, 'w') as f:
        f.write("num_samples,score\n")
        for calls, score in enumerate(mean_scores):
            f.write(f"{calls},{score}\n")
    
    print(f"  ✅ Saved: {output_path}")


def main():
    """Main function to regenerate all num_samples CSV files."""
    print("=" * 80)
    print("REGENERATING RESULT_NUM_SAMPLES CSV FILES")
    print("=" * 80)
    
    # Base directory
    base_dir = Path(__file__).resolve().parent / "data"
    
    # Folders to process
    folders = ["PS", "PS_epsNet_summarizer", "PS_summarizer", "PS_epsNet"]
    
    # Process each folder
    for folder_name in folders:
        folder_path = base_dir / folder_name
        
        if not folder_path.exists():
            print(f"\n⚠️  Warning: {folder_path} not found, skipping...")
            continue
        
        print(f"\n{'='*80}")
        print(f"Processing folder: {folder_name}")
        print(f"{'='*80}")
        
        # Process each run (1, 2, 3)
        for run_num in range(1, 4):
            output_file = folder_path / f"result_num_samples_{run_num}.csv"
            generate_num_samples_csv(folder_path, run_num, output_file)
    
    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Regenerated result_num_samples CSV files")
    print(f"   - 4 folders × 3 runs = 12 CSV files")
    print("=" * 80)


if __name__ == "__main__":
    main()
