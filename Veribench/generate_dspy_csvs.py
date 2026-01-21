#!/usr/bin/env python3
"""
Generate CSV files for DSPy data (runs 2 and 3) with scores.
Creates 8 CSV files (2 runs × 4 metrics) for plotting.
Note: For DSPy, all 4 metrics are identical (attempt = num_samples = num_proposals = eval_step = prop_step)
"""

import json
from pathlib import Path
from typing import List, Dict
import numpy as np


def load_task_data(run_num: int, task_id: int, base_dir: Path) -> List[Dict]:
    """
    Load history data for a specific task.
    Returns list of attempts with score.
    """
    result_file = base_dir / f"dspy_{run_num}" / f"dspy_task_{task_id}_result.json"
    
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
    Compute the highest score so far at each attempt.
    Returns dict mapping attempt number to best score so far.
    Note: DSPy uses 'score' field, not 'test_score'
    """
    best_scores = {}
    best_so_far = 0.0
    
    for attempt_data in history:
        attempt = attempt_data.get('attempt')
        score = attempt_data.get('score')  # Use 'score' for DSPy
        
        if attempt is None or score is None:
            continue
        
        # Update best score so far
        if score > best_so_far:
            best_so_far = score
        
        best_scores[attempt] = best_so_far
    
    return best_scores


def generate_csv(run_num: int, base_dir: Path, output_dir: Path, metric_name: str):
    """
    Generate CSV for a specific metric.
    For DSPy: all metrics are identical (attempt-based)
    
    Args:
        run_num: Run number (2 or 3)
        base_dir: Base directory with result files
        output_dir: Output directory for CSV files
        metric_name: One of 'num_samples', 'num_proposals', 'eval_step', 'prop_step'
    """
    print(f"\n📊 Generating result_{metric_name}_{run_num}.csv")
    
    max_attempts = 50
    all_task_scores = []
    
    # Process each task (10-50)
    for task_id in range(10, 51):
        history = load_task_data(run_num, task_id, base_dir)
        if not history:
            # If no data, use zeros
            all_task_scores.append([0.0] * (max_attempts + 1))
            continue
        
        # Get best scores by attempt
        best_by_attempt = compute_best_scores_by_attempt(history)
        
        # Forward fill for attempts 0 to max_attempts
        task_scores = []
        last_score = 0.0
        for i in range(max_attempts + 1):
            if i in best_by_attempt:
                last_score = best_by_attempt[i]
            task_scores.append(last_score)
        
        all_task_scores.append(task_scores)
    
    # Calculate mean across all tasks for each attempt
    mean_scores = np.mean(all_task_scores, axis=0)
    
    # Write CSV
    output_file = output_dir / f"result_{metric_name}_{run_num}.csv"
    with open(output_file, 'w') as f:
        f.write(f"{metric_name},score\n")
        for i, score in enumerate(mean_scores):
            f.write(f"{i},{score}\n")
    
    print(f"  ✅ Saved: {output_file}")


def main():
    """Main function to generate all CSV files."""
    print("=" * 80)
    print("GENERATING DSPY CSV FILES (RUNS 2 AND 3)")
    print("=" * 80)
    print("Note: For DSPy, all 4 metrics are identical (attempt-based)")
    
    # Setup paths
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir / "results_llm_judge"
    output_dir = script_dir / "data" / "llm_judge" / "dspy"
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Metrics to generate (all identical for DSPy)
    metrics = ['num_samples', 'num_proposals', 'eval_step', 'prop_step']
    
    # Generate CSVs for runs 2 and 3
    for run_num in [2, 3]:
        print(f"\n{'='*80}")
        print(f"Processing DSPy Run {run_num}")
        print(f"{'='*80}")
        
        # Generate all 4 metrics for this run
        for metric in metrics:
            generate_csv(run_num, base_dir, output_dir, metric)
    
    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Generated 8 CSV files in: {output_dir}")
    print("   - result_num_samples_{2,3}.csv")
    print("   - result_num_proposals_{2,3}.csv")
    print("   - result_eval_step_{2,3}.csv")
    print("   - result_prop_step_{2,3}.csv")
    print("\nNote: All 4 metrics have identical values for DSPy")
    print("      (each attempt = 1 sample = 1 proposal = 1 eval step = 1 prop step)")
    print("=" * 80)


if __name__ == "__main__":
    main()
