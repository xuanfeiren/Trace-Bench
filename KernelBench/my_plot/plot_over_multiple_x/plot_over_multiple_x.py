#!/usr/bin/env python3
"""
Plot KernelBench learning curves over multiple x values (eval_step, prop_step, num_samples, num_proposals).
Creates fast_0.5 and fast_1.0 plots for different algorithms.

Data is loaded from CSV files in plot_over_multiple_x/data/
Each CSV has columns: task_id, iteration, eval_step, prop_step, num_samples, num_proposals, score
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_csv_data(csv_file, x_column='eval_step'):
    """
    Load data from CSV file and organize by task_id and x_column.
    
    Args:
        csv_file: Path to CSV file
        x_column: Column to use as x-axis ('eval_step', 'prop_step', 'num_samples', or 'num_proposals')
    
    Returns:
        Dictionary {task_id: {x_value: score}}
    """
    df = pd.read_csv(csv_file)
    
    task_data = {}
    
    for task_id in df['task_id'].unique():
        task_df = df[df['task_id'] == task_id].copy()
        
        # Sort by x_column to ensure proper ordering
        task_df = task_df.sort_values(x_column)
        
        # Create mapping from x_value to score
        x_to_score = {}
        for _, row in task_df.iterrows():
            x_val = int(row[x_column])
            score = float(row['score'])
            x_to_score[x_val] = score
        
        task_data[int(task_id)] = x_to_score
    
    return task_data


def calculate_fast_metric(task_data, threshold, max_x_value=None):
    """
    Calculate fast_p metric: ratio of tasks with score > threshold at each x value.
    
    Args:
        task_data: Dictionary {task_id: {x_value: score}}
        threshold: Threshold value (0.5 or 1.0)
        max_x_value: Maximum x value to consider (None = use max from data)
    
    Returns:
        (x_values, fast_values): Arrays of x values and fast_p values
    """
    num_tasks = len(task_data)
    
    if num_tasks == 0:
        return np.array([]), np.array([])
    
    # Find all unique x values across all tasks
    all_x_values = set()
    for task_idx, x_scores in task_data.items():
        all_x_values.update(x_scores.keys())
    
    if not all_x_values:
        return np.array([]), np.array([])
    
    # Determine max x value
    if max_x_value is None:
        max_x_value = max(all_x_values)
    
    # Get sorted x values up to max_x_value
    x_values = sorted([x for x in all_x_values if x <= max_x_value])
    
    # Calculate fast_p at each x value
    fast_values = []
    
    for x_val in x_values:
        count = 0
        for task_idx, x_scores in task_data.items():
            # Get score at this x value (or last available score if forward fill)
            if x_val in x_scores:
                score = x_scores[x_val]
            else:
                # Forward fill: use last available score <= x_val
                available_x = [x for x in x_scores.keys() if x <= x_val]
                if available_x:
                    score = x_scores[max(available_x)]
                else:
                    score = 0.0
            
            if score > threshold:
                count += 1
        
        fast_p = count / num_tasks
        fast_values.append(fast_p)
    
    return np.array(x_values), np.array(fast_values)


def create_plot(algorithms_data, output_prefix, threshold, y_label, x_label='Step'):
    """
    Create a plot for fast_p metric.
    
    Args:
        algorithms_data: List of (x_values, fast_values, display_name, color, linestyle, marker)
        output_prefix: Prefix for output files
        threshold: Threshold value (for title)
        y_label: Y-axis label
        x_label: X-axis label
    """
    # Line width and marker size
    linewidth = 3
    markersize = 8
    
    # Figure size
    figsize = (14, 8)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    for x_values, fast_values, display_name, color, linestyle, marker in algorithms_data:
        plt.plot(x_values, fast_values,
                marker=marker,
                label=display_name,
                linewidth=linewidth,
                markersize=markersize,
                color=color,
                linestyle=linestyle)
    
    # Styling with larger fonts
    plt.xlabel(x_label, fontsize=28, fontweight='bold')
    plt.ylabel(y_label, fontsize=28, fontweight='bold')
    plt.title(f'KernelBench fast_{threshold} Over {x_label}', fontsize=30, fontweight='bold', pad=20)
    plt.legend(fontsize=22, loc='best', frameon=True, shadow=True)
    plt.grid(True, alpha=0.3, linewidth=1)
    
    # Set axis limits
    # Get max x value from all algorithms
    max_x = 0
    for x_values, _, _, _, _, _ in algorithms_data:
        if len(x_values) > 0:
            max_x = max(max_x, max(x_values))
    
    plt.xlim(0, max_x)
    plt.ylim(-0.05, 1.05)
    
    # Increase tick label sizes
    plt.tick_params(axis='both', which='major', labelsize=22)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path("/Users/xuanfeiren/Documents/Trace-Bench/KernelBench/my_plot/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file_pdf = output_dir / f"{output_prefix}.pdf"
    
    # Save PDF
    plt.savefig(output_file_pdf, format='pdf', bbox_inches='tight', pad_inches=0.05)
    print(f"PDF plot saved to {output_file_pdf}")
    
    # Show plot
    plt.show()


def main():
    """Main function to create plots."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Base path for data
    data_dir = Path("/Users/xuanfeiren/Documents/Trace-Bench/KernelBench/my_plot/plot_over_multiple_x/data")
    
    # Algorithm configurations: (directory_name, display_name)
    algorithms = [
        ("gepa", "GEPA"),
        ("PS_summarizer", "PS"),
        ("openevolve", "OpenEvolve"),
        ("dspy", "DSPy"),
    ]
    
    # Define colors and styles
    colors = ['#8c564b', '#d62728', '#2ca02c', '#ff7f0e']  # GEPA, PS, OpenEvolve, DSPy
    linestyles = ['--', '-', '--', '--']
    markers = ['D', 'o', 's', 'o']
    
    # X-axis configurations: (column_name, display_name)
    x_axes = [
        ('eval_step', 'Eval Step'),
        ('prop_step', 'Prop Step'),
        ('num_samples', 'Num Samples'),
        ('num_proposals', 'Num Proposals'),
    ]
    
    # Load data for each algorithm
    all_algorithm_data = {}
    
    for algo_dir, display_name in algorithms:
        csv_file = data_dir / algo_dir / "data.csv"
        if csv_file.exists():
            print(f"\nLoading {display_name}...")
            # Load data for all x axes
            algo_data = {}
            for x_col, _ in x_axes:
                algo_data[x_col] = load_csv_data(csv_file, x_column=x_col)
            all_algorithm_data[display_name] = algo_data
            print(f"  Loaded {len(algo_data['eval_step'])} tasks")
        else:
            print(f"Warning: {csv_file} not found. Skipping {display_name}.")
    
    # ========================================================================
    # Create plots for each x-axis type
    # ========================================================================
    for x_col, x_label in x_axes:
        print("\n" + "="*80)
        print(f"CREATING PLOTS FOR {x_label.upper()}")
        print("="*80)
        
        # ========================================================================
        # PLOT 1: fast_0.5
        # ========================================================================
        print(f"\nCreating fast_0.5 plot for {x_label}...")
        
        fast_05_data = []
        for idx, (algo_dir, display_name) in enumerate(algorithms):
            if display_name in all_algorithm_data:
                task_data = all_algorithm_data[display_name][x_col]
                x_values, fast_values = calculate_fast_metric(task_data, threshold=0.5)
                if len(x_values) > 0:
                    print(f"  {display_name}: fast_0.5 at max {x_label} = {fast_values[-1]:.3f}")
                    fast_05_data.append((
                        x_values, fast_values, display_name,
                        colors[idx], linestyles[idx], markers[idx]
                    ))
        
        if fast_05_data:
            output_prefix = f"fast_0.5_{x_col}"
            create_plot(fast_05_data, output_prefix, "0.5", "fast_0.5", x_label=x_label)
        
        # ========================================================================
        # PLOT 2: fast_1.0
        # ========================================================================
        print(f"\nCreating fast_1.0 plot for {x_label}...")
        
        fast_10_data = []
        for idx, (algo_dir, display_name) in enumerate(algorithms):
            if display_name in all_algorithm_data:
                task_data = all_algorithm_data[display_name][x_col]
                x_values, fast_values = calculate_fast_metric(task_data, threshold=1.0)
                if len(x_values) > 0:
                    print(f"  {display_name}: fast_1.0 at max {x_label} = {fast_values[-1]:.3f}")
                    fast_10_data.append((
                        x_values, fast_values, display_name,
                        colors[idx], linestyles[idx], markers[idx]
                    ))
        
        if fast_10_data:
            output_prefix = f"fast_1.0_{x_col}"
            create_plot(fast_10_data, output_prefix, "1.0", "fast_1.0", x_label=x_label)
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == "__main__":
    main()
