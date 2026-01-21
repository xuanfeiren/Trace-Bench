#!/usr/bin/env python3
"""
Process OpenEvolve kernel benchmark results and extract best scores for each task.
"""
import json
import glob
from pathlib import Path
from typing import Dict, List
import csv


def load_summary_file(filepath: Path) -> Dict:
    """Load a single summary JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def process_results(results_dir: Path) -> List[Dict]:
    """
    Process all summary files in the directory and extract key metrics.
    
    Returns:
        List of dicts with task_idx, best_score, success, attempts, duration
    """
    summary_files = sorted(glob.glob(str(results_dir / "openevolve_task_*_summary.json")))
    
    results = []
    for filepath in summary_files:
        data = load_summary_file(Path(filepath))
        
        # OpenEvolve uses "best_speedup" instead of "best_score"
        best_score = data.get('best_speedup', 0.0)
        if best_score is None:
            best_score = 0.0
        
        results.append({
            'task_idx': data['task_idx'],
            'success': best_score > 0.0,
            'attempts': data.get('num_metric_calls', 0),
            'best_score': best_score,
            'duration_seconds': data.get('duration_seconds', 0.0),
            'model': data.get('model', 'unknown'),
        })
    
    return results


def print_summary(results: List[Dict]):
    """Print a formatted summary of all results."""
    print("=" * 80)
    print("OpenEvolve Kernel Benchmark Results Summary")
    print("=" * 80)
    print()
    
    # Sort by task_idx
    results = sorted(results, key=lambda x: x['task_idx'])
    
    # Header
    print(f"{'Task':<6} {'Success':<10} {'Attempts':<10} {'Best Score':<15} {'Duration (s)':<15}")
    print("-" * 80)
    
    # Data rows
    total_success = 0
    total_tasks = len(results)
    best_scores = []
    
    for r in results:
        success_str = "✓ Yes" if r['success'] else "✗ No"
        score_str = f"{r['best_score']:.4f}x" if r['best_score'] > 0 else "N/A"
        duration_str = f"{r['duration_seconds']:.1f}"
        
        print(f"{r['task_idx']:<6} {success_str:<10} {r['attempts']:<10} {score_str:<15} {duration_str:<15}")
        
        if r['success']:
            total_success += 1
            best_scores.append(r['best_score'])
    
    print("-" * 80)
    print()
    
    # Statistics
    print("Statistics:")
    print(f"  Total Tasks: {total_tasks}")
    print(f"  Successful: {total_success} ({total_success/total_tasks*100:.1f}%)")
    print(f"  Failed: {total_tasks - total_success} ({(total_tasks - total_success)/total_tasks*100:.1f}%)")
    
    if best_scores:
        print()
        print(f"  Best Scores (speedup):")
        print(f"    Maximum: {max(best_scores):.4f}x")
        print(f"    Minimum: {min(best_scores):.4f}x")
        print(f"    Average: {sum(best_scores)/len(best_scores):.4f}x")
        print(f"    Median: {sorted(best_scores)[len(best_scores)//2]:.4f}x")
    else:
        print()
        print(f"  No successful tasks - all scores are 0")
    
    print()
    print("=" * 80)


def save_summary_csv(results: List[Dict], output_file: Path):
    """Save results to a CSV file."""
    import csv
    
    results = sorted(results, key=lambda x: x['task_idx'])
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['task_idx', 'success', 'attempts', 'best_score', 'duration_seconds', 'model'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Summary saved to: {output_file}")


def main():
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    
    # Process all result files
    results = process_results(script_dir)
    
    # Print summary to console
    print_summary(results)
    
    # Save to CSV
    csv_output = script_dir / "summary.csv"
    save_summary_csv(results, csv_output)
    
    # Save detailed JSON summary
    json_output = script_dir / "summary.json"
    with open(json_output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed summary saved to: {json_output}")


if __name__ == "__main__":
    main()
