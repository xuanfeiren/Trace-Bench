#!/usr/bin/env python3
"""
Generate CSV files for GEPA data with test scores.
Creates 12 CSV files (3 runs × 4 metrics) for plotting.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np


def find_restart_index(iterations: list) -> int:
    """
    Find the index where iterations restart from 1.
    Returns the index where the restart happens, or len(iterations) if no restart.
    """
    if not iterations:
        return 0
    
    for i in range(1, len(iterations)):
        prev_iter = iterations[i-1].get('iteration', 0)
        curr_iter = iterations[i].get('iteration', 0)
        
        # If iteration goes backwards or restarts
        if curr_iter <= prev_iter:
            return i
    
    return len(iterations)


def load_task_data(run_num: int, task_idx: int, base_dir: Path) -> List[Dict]:
    """
    Load valid iteration data for a specific task.
    Returns list of valid iterations with test_score and num_metric_calls_so_far.
    """
    result_file = base_dir / f"gepa_{run_num}" / f"task_{task_idx}_result.json"
    
    if not result_file.exists():
        print(f"  ⚠️  File not found: {result_file}")
        return []
    
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"  ⚠️  Failed to parse: {result_file}")
        return []
    
    iteration_history = data.get('iteration_history', [])
    if not iteration_history:
        return []
    
    # Find valid iterations (before restart)
    restart_idx = find_restart_index(iteration_history)
    valid_iterations = iteration_history[:restart_idx]
    
    return valid_iterations


def compute_best_scores_by_iteration(valid_iterations: List[Dict]) -> Dict[int, float]:
    """
    Compute the highest test score so far at each iteration.
    Returns dict mapping iteration number to best score so far.
    """
    best_scores = {}
    best_so_far = 0.0
    
    for iter_data in valid_iterations:
        iteration = iter_data.get('iteration')
        test_score = iter_data.get('test_score')
        
        if iteration is None or test_score is None:
            continue
        
        # Update best score so far
        if test_score > best_so_far:
            best_so_far = test_score
        
        best_scores[iteration] = best_so_far
    
    return best_scores


def compute_best_scores_by_metric_calls(valid_iterations: List[Dict]) -> Dict[int, float]:
    """
    Compute the highest test score so far at each num_metric_calls_so_far.
    Returns dict mapping num_metric_calls to best score so far.
    """
    best_scores = {}
    best_so_far = 0.0
    
    for iter_data in valid_iterations:
        num_calls = iter_data.get('num_metric_calls_so_far')
        test_score = iter_data.get('test_score')
        
        if num_calls is None or test_score is None:
            continue
        
        # Update best score so far
        if test_score > best_so_far:
            best_so_far = test_score
        
        best_scores[num_calls] = best_so_far
    
    return best_scores


def generate_prop_step_csv(run_num: int, base_dir: Path, output_dir: Path):
    """
    Generate result_prop_step_{run_num}.csv
    X-axis: prop_step (iteration number) from 0 to 30
    """
    print(f"\n📊 Generating result_prop_step_{run_num}.csv")
    
    max_iterations = 30
    all_task_scores = []
    
    # Process each task (10-50)
    for task_idx in range(10, 51):
        valid_iterations = load_task_data(run_num, task_idx, base_dir)
        if not valid_iterations:
            # If no data, use zeros
            all_task_scores.append([0.0] * (max_iterations + 1))
            continue
        
        # Get best scores by iteration
        best_by_iter = compute_best_scores_by_iteration(valid_iterations)
        
        # Forward fill for iterations 0 to max_iterations
        task_scores = []
        last_score = 0.0
        for i in range(max_iterations + 1):
            if i in best_by_iter:
                last_score = best_by_iter[i]
            task_scores.append(last_score)
        
        all_task_scores.append(task_scores)
    
    # Calculate mean across all tasks for each iteration
    mean_scores = np.mean(all_task_scores, axis=0)
    
    # Write CSV
    output_file = output_dir / f"result_prop_step_{run_num}.csv"
    with open(output_file, 'w') as f:
        f.write("prop_step,score\n")
        for i, score in enumerate(mean_scores):
            f.write(f"{i},{score}\n")
    
    print(f"  ✅ Saved: {output_file}")


def generate_num_proposals_csv(run_num: int, base_dir: Path, output_dir: Path):
    """
    Generate result_num_proposals_{run_num}.csv
    X-axis: num_proposals (same as iteration number) from 0 to 30
    """
    print(f"\n📊 Generating result_num_proposals_{run_num}.csv")
    
    max_proposals = 30
    all_task_scores = []
    
    # Process each task (10-50)
    for task_idx in range(10, 51):
        valid_iterations = load_task_data(run_num, task_idx, base_dir)
        if not valid_iterations:
            # If no data, use zeros
            all_task_scores.append([0.0] * (max_proposals + 1))
            continue
        
        # Get best scores by iteration (same as proposals for GEPA)
        best_by_iter = compute_best_scores_by_iteration(valid_iterations)
        
        # Forward fill for proposals 0 to max_proposals
        task_scores = []
        last_score = 0.0
        for i in range(max_proposals + 1):
            if i in best_by_iter:
                last_score = best_by_iter[i]
            task_scores.append(last_score)
        
        all_task_scores.append(task_scores)
    
    # Calculate mean across all tasks for each proposal
    mean_scores = np.mean(all_task_scores, axis=0)
    
    # Write CSV
    output_file = output_dir / f"result_num_proposals_{run_num}.csv"
    with open(output_file, 'w') as f:
        f.write("num_proposals,score\n")
        for i, score in enumerate(mean_scores):
            f.write(f"{i},{score}\n")
    
    print(f"  ✅ Saved: {output_file}")


def generate_eval_step_csv(run_num: int, base_dir: Path, output_dir: Path):
    """
    Generate result_eval_step_{run_num}.csv
    X-axis: eval_step (num_metric_calls_so_far) from 0 to 50
    """
    print(f"\n📊 Generating result_eval_step_{run_num}.csv")
    
    max_eval_steps = 50
    all_task_scores = []
    
    # Process each task (10-50)
    for task_idx in range(10, 51):
        valid_iterations = load_task_data(run_num, task_idx, base_dir)
        if not valid_iterations:
            # If no data, use zeros
            all_task_scores.append([0.0] * (max_eval_steps + 1))
            continue
        
        # Get best scores by metric calls
        best_by_calls = compute_best_scores_by_metric_calls(valid_iterations)
        
        # Forward fill for eval_steps 0 to max_eval_steps
        task_scores = []
        last_score = 0.0
        for i in range(max_eval_steps + 1):
            if i in best_by_calls:
                last_score = best_by_calls[i]
            task_scores.append(last_score)
        
        all_task_scores.append(task_scores)
    
    # Calculate mean across all tasks for each eval step
    mean_scores = np.mean(all_task_scores, axis=0)
    
    # Write CSV
    output_file = output_dir / f"result_eval_step_{run_num}.csv"
    with open(output_file, 'w') as f:
        f.write("eval_step,score\n")
        for i, score in enumerate(mean_scores):
            f.write(f"{i},{score}\n")
    
    print(f"  ✅ Saved: {output_file}")


def generate_num_samples_csv(run_num: int, base_dir: Path, output_dir: Path):
    """
    Generate result_num_samples_{run_num}.csv
    X-axis: num_samples (same as eval_step) from 0 to 50
    """
    print(f"\n📊 Generating result_num_samples_{run_num}.csv")
    
    max_samples = 50
    all_task_scores = []
    
    # Process each task (10-50)
    for task_idx in range(10, 51):
        valid_iterations = load_task_data(run_num, task_idx, base_dir)
        if not valid_iterations:
            # If no data, use zeros
            all_task_scores.append([0.0] * (max_samples + 1))
            continue
        
        # Get best scores by metric calls
        best_by_calls = compute_best_scores_by_metric_calls(valid_iterations)
        
        # Forward fill for samples 0 to max_samples
        task_scores = []
        last_score = 0.0
        for i in range(max_samples + 1):
            if i in best_by_calls:
                last_score = best_by_calls[i]
            task_scores.append(last_score)
        
        all_task_scores.append(task_scores)
    
    # Calculate mean across all tasks for each sample count
    mean_scores = np.mean(all_task_scores, axis=0)
    
    # Write CSV
    output_file = output_dir / f"result_num_samples_{run_num}.csv"
    with open(output_file, 'w') as f:
        f.write("num_samples,score\n")
        for i, score in enumerate(mean_scores):
            f.write(f"{i},{score}\n")
    
    print(f"  ✅ Saved: {output_file}")


def main():
    """Main function to generate all CSV files."""
    print("=" * 80)
    print("GENERATING GEPA CSV FILES WITH TEST SCORES")
    print("=" * 80)
    
    # Setup paths
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir / "results_llm_judge"
    output_dir = script_dir / "data" / "llm_judge" / "gepa"
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate CSVs for each run (1, 2, 3)
    for run_num in range(1, 4):
        print(f"\n{'='*80}")
        print(f"Processing GEPA Run {run_num}")
        print(f"{'='*80}")
        
        # Generate all 4 metrics for this run
        generate_prop_step_csv(run_num, base_dir, output_dir)
        generate_num_proposals_csv(run_num, base_dir, output_dir)
        generate_eval_step_csv(run_num, base_dir, output_dir)
        generate_num_samples_csv(run_num, base_dir, output_dir)
    
    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Generated 12 CSV files in: {output_dir}")
    print("   - result_prop_step_{1,2,3}.csv")
    print("   - result_num_proposals_{1,2,3}.csv")
    print("   - result_eval_step_{1,2,3}.csv")
    print("   - result_num_samples_{1,2,3}.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
