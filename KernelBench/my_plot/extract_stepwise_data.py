#!/usr/bin/env python3
"""
Extract step-by-step results from wandb projects and save to JSON files.
"""
import wandb
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
import sys


def extract_stepwise_data_from_wandb(project_name: str, output_file: str, entity: Optional[str] = None):
    """
    Extract step-by-step data from wandb project.

    Args:
        project_name: Name of the wandb project (e.g., 'kernel-bench-PS')
        output_file: Path to output JSON file
        entity: wandb entity/username (optional)
    """
    print(f"\n{'='*80}")
    print(f"Extracting data from wandb project: {project_name}")
    print(f"{'='*80}")

    # Initialize wandb API
    api = wandb.Api()

    # Get all runs from the project
    if entity:
        runs = api.runs(f"{entity}/{project_name}")
    else:
        runs = api.runs(project_name)

    print(f"Found {len(runs)} runs in project")

    # Store results: {task_idx: {step: score}}
    task_results = {}

    for run in runs:
        # Extract task index from run name or config
        task_idx = None

        # Try to get task_idx from config
        try:
            config = dict(run.config) if run.config else {}
            if 'task_idx' in config:
                task_idx = config['task_idx']
            elif 'task_id' in config:
                task_idx = config['task_id']
            elif 'task' in config:
                task_idx = config['task']
        except:
            config = {}

        # Try to extract from run name if not in config
        if task_idx is None and run.name:
            match = re.search(r'task[_-]?(\d+)', run.name.lower())
            if match:
                task_idx = int(match.group(1))

        if task_idx is None:
            print(f"Warning: Could not determine task_idx for run {run.name}, skipping")
            continue

        print(f"Processing task {task_idx} (run: {run.name})...")

        # Get history data
        try:
            history = run.history()

            # Look for the metric "Test/Highest test score so far"
            metric_name = None
            possible_names = [
                'Test/Highest test score so far',
                'Test/Highest test so far',
                'test/highest_test_score_so_far',
                'highest_test_score_so_far',
            ]

            for name in possible_names:
                if name in history.columns:
                    metric_name = name
                    break

            if metric_name is None:
                print(f"  Warning: Could not find metric for task {task_idx}")
                print(f"  Available columns: {list(history.columns)[:10]}...")
                continue

            # Extract step-by-step scores
            # Assuming the history has a 'step' or '_step' column
            step_col = None
            for col in ['_step', 'step', 'Step']:
                if col in history.columns:
                    step_col = col
                    break

            if step_col is None:
                print(f"  Warning: Could not find step column for task {task_idx}")
                continue

            # Get step-wise scores
            step_scores = {}
            for _, row in history.iterrows():
                step = int(row[step_col])
                score = float(row[metric_name]) if not pd.isna(row[metric_name]) else 0.0
                step_scores[step] = score

            # Store results for this task
            task_results[task_idx] = step_scores
            print(f"  Extracted {len(step_scores)} steps (0-{max(step_scores.keys()) if step_scores else 0})")

        except Exception as e:
            print(f"  Error processing task {task_idx}: {e}")
            continue

    # Save to JSON file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(task_results, f, indent=2)

    print(f"\nSaved step-wise data to {output_path}")
    print(f"Total tasks extracted: {len(task_results)}")

    return task_results


def main():
    """Main function to extract data from all wandb projects."""

    # Import pandas here
    global pd
    import pandas as pd

    # Define wandb projects and output files
    projects = [
        ("kernel-bench-PS", "kernel_ps_stepwise.json"),
        ("kernel-bench-PS-Summarizer", "kernel_ps_summarizer_stepwise.json"),
        ("kernel-bench-PS-eps0.1Net-Summarizer", "kernel_ps_epsnet_summarizer_stepwise.json"),
        ("kernel-bench-PS-eps0.1-onlyforSummarizer", "kernel_ps_epsnet_only_summarizer_stepwise.json"),
    ]

    # Base output directory
    output_dir = Path("/Users/xuanfeiren/Documents/Trace-Bench/KernelBench/my_plot/extracted_data")

    # Extract data from each project
    for project_name, output_file in projects:
        try:
            output_path = output_dir / output_file
            extract_stepwise_data_from_wandb(project_name, str(output_path))
        except Exception as e:
            print(f"Error extracting data from {project_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("DATA EXTRACTION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
