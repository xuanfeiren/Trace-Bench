#!/usr/bin/env python3
"""
Generate CSV files for OpenEvolve data with test scores.
Creates 12 CSV files (3 runs × 4 metrics) for plotting.
"""

import json
from pathlib import Path
from typing import List, Dict
import numpy as np


def load_task_data(run_num: int, task_id: int, base_dir: Path) -> List[Dict]:
    """
    Load history data for a specific task.
    Returns list of attempts with test_score.
    """
    result_file = base_dir / f"openevolve_{run_num}" / f"openevolve_task_{task_id}_result.json"
    
    if not result_file.exists():
        print(f"  ⚠️  File not found: {result_file}")
        return []
    
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"  ⚠️  Failed to parse: {result_file}")
        return []
    
    history = data.get('history', [])
    return history


def compute_best_scores_by_attempt(history: List[Dict]) -> Dict[int, float]:
    """
    Compute the highest test score so far at each attempt.
    Returns dict mapping attempt number to best score so far.
    """
    best_scores = {}
    best_so_far = 0.0
    
    for attempt_data in history:
        attempt = attempt_data.get('attempt')
        test_score = attempt_data.get('test_score')
        
        if attempt is None or test_score is None:
            continue
        
        # Update best score so far
        if test_score > best_so_far:
            best_so_far = test_score
        
        best_scores[attempt] = best_so_far
    
    return best_scores


def generate_num_samples_csv(run_num: int, base_dir: Path, output_dir: Path):
    """
    Generate result_num_samples_{run_num}.csv
    X-axis: num_samples (attempt number) from 0 to 50
    For OpenEvolve: num_samples = attempt
    """
    print(f"\n📊 Generating result_num_samples_{run_num}.csv")
    
    max_samples = 50
    all_task_scores = []
    
    # Process each task (10-50)
    for task_id in range(10, 51):
        history = load_task_data(run_num, task_id, base_dir)
        if not history:
            # If no data, use zeros
            all_task_scores.append([0.0] * (max_samples + 1))
            continue
        
        # Get best scores by attempt
        best_by_attempt = compute_best_scores_by_attempt(history)
        
        # Forward fill for samples 0 to max_samples
        task_scores = []
        last_score = 0.0
        for i in range(max_samples + 1):
            if i in best_by_attempt:
                last_score = best_by_attempt[i]
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


def generate_num_proposals_csv(run_num: int, base_dir: Path, output_dir: Path):
    """
    Generate result_num_proposals_{run_num}.csv
    X-axis: num_proposals (attempt number) from 0 to 50
    For OpenEvolve: num_proposals = attempt
    """
    print(f"\n📊 Generating result_num_proposals_{run_num}.csv")
    
    max_proposals = 50
    all_task_scores = []
    
    # Process each task (10-50)
    for task_id in range(10, 51):
        history = load_task_data(run_num, task_id, base_dir)
        if not history:
            # If no data, use zeros
            all_task_scores.append([0.0] * (max_proposals + 1))
            continue
        
        # Get best scores by attempt
        best_by_attempt = compute_best_scores_by_attempt(history)
        
        # Forward fill for proposals 0 to max_proposals
        task_scores = []
        last_score = 0.0
        for i in range(max_proposals + 1):
            if i in best_by_attempt:
                last_score = best_by_attempt[i]
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
    X-axis: eval_step from 0 to 10
    For OpenEvolve: eval_step uses 5 parallel evaluations
    - eval_step=1 means look at attempt=5 (1 * 5)
    - eval_step=2 means look at attempt=10 (2 * 5)
    - eval_step=n means look at attempt=n*5
    """
    print(f"\n📊 Generating result_eval_step_{run_num}.csv")
    
    max_eval_steps = 10  # 10 * 5 = 50 attempts
    all_task_scores = []
    
    # Process each task (10-50)
    for task_id in range(10, 51):
        history = load_task_data(run_num, task_id, base_dir)
        if not history:
            # If no data, use zeros
            all_task_scores.append([0.0] * (max_eval_steps + 1))
            continue
        
        # Get best scores by attempt
        best_by_attempt = compute_best_scores_by_attempt(history)
        
        # For each eval_step, look at attempt = eval_step * 5
        task_scores = []
        last_score = 0.0
        
        for eval_step in range(max_eval_steps + 1):
            # eval_step 0 means no evaluations yet
            if eval_step == 0:
                task_scores.append(0.0)
            else:
                # eval_step n corresponds to attempt n*5
                target_attempt = eval_step * 5
                
                # Get best score at or before this attempt
                if target_attempt in best_by_attempt:
                    last_score = best_by_attempt[target_attempt]
                # If exact attempt not found, use forward fill
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


def generate_prop_step_csv(run_num: int, base_dir: Path, output_dir: Path):
    """
    Generate result_prop_step_{run_num}.csv
    X-axis: prop_step from 0 to 10
    For OpenEvolve: prop_step = eval_step (same calculation)
    """
    print(f"\n📊 Generating result_prop_step_{run_num}.csv")
    
    max_prop_steps = 10  # 10 * 5 = 50 attempts
    all_task_scores = []
    
    # Process each task (10-50)
    for task_id in range(10, 51):
        history = load_task_data(run_num, task_id, base_dir)
        if not history:
            # If no data, use zeros
            all_task_scores.append([0.0] * (max_prop_steps + 1))
            continue
        
        # Get best scores by attempt
        best_by_attempt = compute_best_scores_by_attempt(history)
        
        # For each prop_step, look at attempt = prop_step * 5
        task_scores = []
        last_score = 0.0
        
        for prop_step in range(max_prop_steps + 1):
            # prop_step 0 means no proposals yet
            if prop_step == 0:
                task_scores.append(0.0)
            else:
                # prop_step n corresponds to attempt n*5
                target_attempt = prop_step * 5
                
                # Get best score at or before this attempt
                if target_attempt in best_by_attempt:
                    last_score = best_by_attempt[target_attempt]
                # If exact attempt not found, use forward fill
                task_scores.append(last_score)
        
        all_task_scores.append(task_scores)
    
    # Calculate mean across all tasks for each prop step
    mean_scores = np.mean(all_task_scores, axis=0)
    
    # Write CSV
    output_file = output_dir / f"result_prop_step_{run_num}.csv"
    with open(output_file, 'w') as f:
        f.write("prop_step,score\n")
        for i, score in enumerate(mean_scores):
            f.write(f"{i},{score}\n")
    
    print(f"  ✅ Saved: {output_file}")


def main():
    """Main function to generate all CSV files."""
    print("=" * 80)
    print("GENERATING OPENEVOLVE CSV FILES WITH TEST SCORES")
    print("=" * 80)
    
    # Setup paths
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir / "results_llm_judge"
    output_dir = script_dir / "data" / "llm_judge" / "openevolve"
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate CSVs for each run (1, 2, 3)
    for run_num in range(1, 4):
        print(f"\n{'='*80}")
        print(f"Processing OpenEvolve Run {run_num}")
        print(f"{'='*80}")
        
        # Generate all 4 metrics for this run
        generate_num_samples_csv(run_num, base_dir, output_dir)
        generate_num_proposals_csv(run_num, base_dir, output_dir)
        generate_eval_step_csv(run_num, base_dir, output_dir)
        generate_prop_step_csv(run_num, base_dir, output_dir)
    
    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Generated 12 CSV files in: {output_dir}")
    print("   - result_num_samples_{1,2,3}.csv (attempt 0-50)")
    print("   - result_num_proposals_{1,2,3}.csv (attempt 0-50)")
    print("   - result_eval_step_{1,2,3}.csv (eval_step 0-10, 5 parallel)")
    print("   - result_prop_step_{1,2,3}.csv (prop_step 0-10, 5 parallel)")
    print("=" * 80)


if __name__ == "__main__":
    main()
