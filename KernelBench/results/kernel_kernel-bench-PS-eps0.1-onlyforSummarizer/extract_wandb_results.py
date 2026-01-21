#!/usr/bin/env python3
"""
Extract results from wandb project 'kernel-bench-PS-eps0.1-onlyforSummarizer' and create summary files.
"""
import wandb
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional
import sys


def extract_task_results(project_name: str, entity: Optional[str] = None) -> List[Dict]:
    """
    Extract results from wandb project.
    
    Args:
        project_name: Name of the wandb project (e.g., 'kernel-bench-PS-eps0.1-onlyforSummarizer')
        entity: wandb entity/username (optional, uses default if not provided)
    
    Returns:
        List of dicts with task_idx and best_score
    """
    print(f"Connecting to wandb project: {project_name}")
    
    # Initialize wandb API
    api = wandb.Api()
    
    # Get all runs from the project
    if entity:
        runs = api.runs(f"{entity}/{project_name}")
    else:
        runs = api.runs(project_name)
    
    results = []
    tasks_found = set()
    
    print(f"Found {len(runs)} runs in project")
    
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
            # Look for patterns like "task_0", "task_1", etc.
            import re
            match = re.search(r'task[_-]?(\d+)', run.name.lower())
            if match:
                task_idx = int(match.group(1))
        
        if task_idx is None:
            print(f"Warning: Could not determine task_idx for run {run.name}, skipping")
            continue
        
        # Get the metric "Test/Highest test so far"
        try:
            history = run.history()
            
            # Try different possible metric names
            metric_names = [
                'Test/Highest test score so far',
                'Test/Highest test so far',
                'test/highest_test_so_far',
                'highest_test_so_far',
                'best_score',
                'max_score'
            ]
            
            best_score = None
            for metric_name in metric_names:
                if metric_name in history.columns:
                    # Get the maximum value (should be the final value for "Highest test so far")
                    metric_values = history[metric_name].dropna()
                    if len(metric_values) > 0:
                        best_score = float(metric_values.max())
                        print(f"Task {task_idx}: Found metric '{metric_name}' = {best_score}")
                        break
            
            if best_score is None:
                print(f"Warning: Could not find metric for task {task_idx} in run {run.name}")
                print(f"Available metrics: {list(history.columns)}")
                best_score = 0.0
            
            # Check if we already have this task (keep the best score)
            if task_idx in tasks_found:
                # Find existing entry and update if this score is better
                for result in results:
                    if result['task_idx'] == task_idx:
                        if best_score > result['best_score']:
                            result['best_score'] = best_score
                            result['run_name'] = run.name
                            result['run_id'] = run.id
                        break
            else:
                # Get duration from summary or runtime
                duration = 0.0
                try:
                    if hasattr(run, 'summary'):
                        summary = dict(run.summary) if run.summary else {}
                        duration = summary.get('_runtime', 0.0)
                except:
                    duration = 0.0
                
                results.append({
                    'task_idx': task_idx,
                    'best_score': best_score,
                    'run_name': run.name,
                    'run_id': run.id,
                    'success': best_score > 0.0,
                    'model': config.get('model', 'unknown'),
                    'duration_seconds': duration
                })
                tasks_found.add(task_idx)
        
        except Exception as e:
            print(f"Error processing run {run.name}: {e}")
            continue
    
    # Sort by task_idx
    results.sort(key=lambda x: x['task_idx'])
    
    return results


def save_summary_csv(results: List[Dict], output_file: Path):
    """Save results to CSV file."""
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['task_idx', 'success', 'best_score', 'duration_seconds', 'model']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            writer.writerow({
                'task_idx': result['task_idx'],
                'success': result['success'],
                'best_score': result['best_score'],
                'duration_seconds': result.get('duration_seconds', 0.0),
                'model': result.get('model', 'unknown')
            })
    
    print(f"Summary saved to: {output_file}")


def save_summary_json(results: List[Dict], output_file: Path):
    """Save results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Detailed summary saved to: {output_file}")


def print_statistics(results: List[Dict]):
    """Print statistics about the results."""
    print("\n" + "=" * 80)
    print("PS eps0.1 onlyforSummarizer Results Summary")
    print("=" * 80)
    
    total_tasks = len(results)
    successful_tasks = sum(1 for r in results if r['success'])
    
    scores = [r['best_score'] for r in results]
    non_zero_scores = [s for s in scores if s > 0]
    scores_above_1 = [s for s in scores if s > 1.0]
    
    print(f"Total Tasks: {total_tasks}")
    print(f"Successful (score > 0): {successful_tasks} ({successful_tasks/total_tasks*100:.1f}%)")
    print(f"Tasks with score > 1.0: {len(scores_above_1)} ({len(scores_above_1)/total_tasks*100:.1f}%)")
    
    if non_zero_scores:
        print(f"\nBest Scores:")
        print(f"  Maximum: {max(scores):.4f}x")
        print(f"  Mean (all): {sum(scores)/len(scores):.4f}x")
        print(f"  Mean (non-zero): {sum(non_zero_scores)/len(non_zero_scores):.4f}x")
        print(f"  Median (all): {sorted(scores)[len(scores)//2]:.4f}x")
    
    print("\n" + "=" * 80)
    print(f"\n{'Task':<6} {'Success':<10} {'Best Score':<15}")
    print("-" * 40)
    for r in results:
        success_str = "✓ Yes" if r['success'] else "✗ No"
        score_str = f"{r['best_score']:.4f}x" if r['best_score'] > 0 else "N/A"
        print(f"{r['task_idx']:<6} {success_str:<10} {score_str:<15}")
    print("-" * 40)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract results from wandb project')
    parser.add_argument('--project', default='kernel-bench-PS-eps0.1-onlyforSummarizer', help='wandb project name')
    parser.add_argument('--entity', default=None, help='wandb entity/username')
    args = parser.parse_args()
    
    # Setup output directory
    script_dir = Path(__file__).parent
    
    try:
        # Extract results from wandb
        results = extract_task_results(args.project, args.entity)
        
        if not results:
            print("Error: No results found!")
            sys.exit(1)
        
        # Print statistics
        print_statistics(results)
        
        # Save to CSV
        csv_path = script_dir / "summary.csv"
        save_summary_csv(results, csv_path)
        
        # Save to JSON
        json_path = script_dir / "summary.json"
        save_summary_json(results, json_path)
        
        print(f"\n✓ Successfully extracted {len(results)} task results")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
