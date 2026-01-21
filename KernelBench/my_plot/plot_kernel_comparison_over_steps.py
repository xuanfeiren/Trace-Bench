#!/usr/bin/env python3
"""
Plot KernelBench learning curves (fast_0.5 and fast_1) over steps for different algorithms.

For PS algorithms: Extract from wandb (Test/Highest test score so far)
For DSPy: Calculate highest score so far from attempt history
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_ps_algorithm_data(json_file):
    """
    Load PS algorithm step-wise data from extracted JSON file.

    Args:
        json_file: Path to JSON file with format {task_idx: {step: score}}

    Returns:
        Dictionary {task_idx: {step: score}}
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Convert string keys to integers
    task_data = {}
    for task_idx_str, step_scores in data.items():
        task_idx = int(task_idx_str)
        task_data[task_idx] = {int(step): float(score) for step, score in step_scores.items()}

    return task_data


def load_dspy_data(results_dir):
    """
    Load DSPy data and calculate highest score so far at each step.

    Args:
        results_dir: Path to results/kernel_dspy directory

    Returns:
        Dictionary {task_idx: {step: highest_score_so_far}}
    """
    results_dir = Path(results_dir)
    task_data = {}

    for task_idx in range(16):  # Tasks 0-15
        task_file = results_dir / f"dspy_task_{task_idx}_result.json"

        if not task_file.exists():
            print(f"Warning: {task_file} not found")
            continue

        with open(task_file, 'r') as f:
            data = json.load(f)

        # Calculate highest score so far at each step
        # Step 0 = score 0.0
        # Step n = attempt n
        step_scores = {0: 0.0}

        if 'history' in data and data['history']:
            highest_so_far = 0.0
            for entry in data['history']:
                attempt = entry['attempt']
                score = entry['score']

                # Update highest score so far
                highest_so_far = max(highest_so_far, score)
                step_scores[attempt] = highest_so_far

        task_data[task_idx] = step_scores

    return task_data


def load_openevolve_data(results_dir):
    """
    Load OpenEvolve data and extract best_score at each attempt.

    Args:
        results_dir: Path to results/kernel_openevolve directory

    Returns:
        Dictionary {task_idx: {step: best_score_so_far}}
    """
    results_dir = Path(results_dir)
    task_data = {}

    for task_idx in range(16):  # Tasks 0-15
        task_file = results_dir / f"openevolve_task_{task_idx}_result.json"

        if not task_file.exists():
            print(f"Warning: {task_file} not found")
            continue

        with open(task_file, 'r') as f:
            data = json.load(f)

        # Extract best_score at each attempt from history
        # Step 0 = score 0.0
        # Step n = attempt n, using best_score field
        step_scores = {0: 0.0}

        if 'history' in data and data['history']:
            for entry in data['history']:
                attempt = entry['attempt']
                best_score = entry.get('best_score', 0.0)
                step_scores[attempt] = best_score

        task_data[task_idx] = step_scores

    return task_data


def load_gepa_data(results_dir):
    """
    Load GEPA data and extract best_score_so_far at each iteration.

    Args:
        results_dir: Path to results/kernel_gepa directory

    Returns:
        Dictionary {task_idx: {step: best_score_so_far}}
    """
    results_dir = Path(results_dir)
    task_data = {}

    for task_idx in range(16):  # Tasks 0-15
        task_file = results_dir / f"gepa_task_{task_idx}_result.json"

        if not task_file.exists():
            print(f"Warning: {task_file} not found")
            continue

        with open(task_file, 'r') as f:
            data = json.load(f)

        # Extract best_score_so_far at each iteration from iteration_history
        # Step 0 = 0.0 (no work done yet, for consistency with other algorithms)
        # Step n = iteration n, using best_score_so_far
        step_scores = {0: 0.0}

        # Extract from iteration_history (iterations 1 to max_iterations)
        if 'iteration_history' in data and data['iteration_history']:
            for entry in data['iteration_history']:
                iteration = entry['iteration']
                best_score = entry.get('best_score_so_far', 0.0)
                step_scores[iteration] = best_score

        task_data[task_idx] = step_scores

    return task_data


def calculate_fast_metric(task_data, threshold, max_step=10):
    """
    Calculate fast_p metric: ratio of tasks with score > threshold at each step.

    Args:
        task_data: Dictionary {task_idx: {step: score}}
        threshold: Threshold value (0.5 or 1.0)
        max_step: Maximum step to consider

    Returns:
        (x_values, fast_values): Arrays of steps and fast_p values
    """
    # Get all tasks
    num_tasks = len(task_data)

    if num_tasks == 0:
        return np.array([]), np.array([])

    # Calculate fast_p at each step
    fast_values = []
    steps = list(range(max_step + 1))

    for step in steps:
        count = 0
        for task_idx, step_scores in task_data.items():
            # Get score at this step (or last available score if forward fill)
            if step in step_scores:
                score = step_scores[step]
            else:
                # Forward fill: use last available score
                available_steps = [s for s in step_scores.keys() if s <= step]
                if available_steps:
                    score = step_scores[max(available_steps)]
                else:
                    score = 0.0

            if score > threshold:
                count += 1

        fast_p = count / num_tasks
        fast_values.append(fast_p)

    return np.array(steps), np.array(fast_values)


def create_plot(algorithms_data, output_prefix, threshold, y_label):
    """
    Create a plot for fast_p metric.

    Args:
        algorithms_data: List of (x_values, fast_values, display_name, color, linestyle, marker)
        output_prefix: Prefix for output files
        threshold: Threshold value (for title)
        y_label: Y-axis label
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
    plt.xlabel('Step', fontsize=28, fontweight='bold')
    plt.ylabel(y_label, fontsize=28, fontweight='bold')
    plt.title(f'KernelBench fast_{threshold} Over Steps', fontsize=30, fontweight='bold', pad=20)
    plt.legend(fontsize=22, loc='best', frameon=True, shadow=True)
    plt.grid(True, alpha=0.3, linewidth=1)

    # Set axis limits
    plt.xlim(0, 10)
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
    """Main function to create both plots."""
    # Set random seed for reproducibility
    np.random.seed(42)

    print("="*80)
    print("LOADING DATA")
    print("="*80)

    # Base paths
    extracted_data_dir = Path("/Users/xuanfeiren/Documents/Trace-Bench/KernelBench/my_plot/extracted_data")
    dspy_results_dir = Path("/Users/xuanfeiren/Documents/Trace-Bench/KernelBench/results/kernel_dspy")
    openevolve_results_dir = Path("/Users/xuanfeiren/Documents/Trace-Bench/KernelBench/results/kernel_openevolve")
    gepa_results_dir = Path("/Users/xuanfeiren/Documents/Trace-Bench/KernelBench/results/kernel_gepa")

    # Load PS algorithm data
    ps_algorithms = [
        ("kernel_ps_stepwise.json", "Vanilla PS"),
        ("kernel_ps_summarizer_stepwise.json", "PS+Summarizer"),
        ("kernel_ps_epsnet_summarizer_stepwise.json", r"PS+$\varepsilon$-Net+Summarizer"),
        ("kernel_ps_epsnet_only_summarizer_stepwise.json", r"PS+$\varepsilon$-Net only Summarizer"),
    ]

    all_algorithm_data = {}

    # Load PS algorithms
    for json_file, display_name in ps_algorithms:
        file_path = extracted_data_dir / json_file
        if file_path.exists():
            print(f"\nLoading {display_name}...")
            data = load_ps_algorithm_data(file_path)
            print(f"  Loaded {len(data)} tasks")
            all_algorithm_data[display_name] = data
        else:
            print(f"Warning: {file_path} not found. Please run extract_stepwise_data.py first.")

    # Load DSPy data
    print(f"\nLoading DSPy ChainOfThought...")
    dspy_data = load_dspy_data(dspy_results_dir)
    print(f"  Loaded {len(dspy_data)} tasks")
    all_algorithm_data["DSPy ChainOfThought"] = dspy_data

    # Load OpenEvolve data
    print(f"\nLoading OpenEvolve...")
    openevolve_data = load_openevolve_data(openevolve_results_dir)
    print(f"  Loaded {len(openevolve_data)} tasks")
    all_algorithm_data["OpenEvolve"] = openevolve_data

    # Load GEPA data (only if all 16 tasks are available)
    print(f"\nLoading GEPA...")
    gepa_data = load_gepa_data(gepa_results_dir)
    print(f"  Loaded {len(gepa_data)} tasks")
    if len(gepa_data) == 16:
        all_algorithm_data["GEPA"] = gepa_data
        print("  ✓ GEPA data complete (16/16 tasks)")
    else:
        print(f"  ✗ GEPA data incomplete ({len(gepa_data)}/16 tasks) - skipping GEPA from plots")

    # Define colors and styles
    # Order: [DSPy, OpenEvolve, GEPA, PS, PS+Summarizer, PS+epsNet+Summarizer, PS+epsNet only Summarizer]
    colors = ['#ff7f0e', '#2ca02c', '#8c564b', '#b0b0b0', '#d62728', '#9467bd', '#1f77b4']
    linestyles = ['--', '--', '--', '-', '-', '-', '-']
    markers = ['o', 's', 'D', 'o', 'o', 'o', 'o']

    algorithm_order = [
        "DSPy ChainOfThought",
        "OpenEvolve",
        "GEPA",
        "Vanilla PS",
        "PS+Summarizer",
        r"PS+$\varepsilon$-Net+Summarizer",
        r"PS+$\varepsilon$-Net only Summarizer",
    ]

    # ========================================================================
    # PLOT 1: fast_0.5
    # ========================================================================
    print("\n" + "="*80)
    print("CREATING PLOT 1: fast_0.5")
    print("="*80)

    fast_05_data = []
    for idx, algo_name in enumerate(algorithm_order):
        if algo_name in all_algorithm_data:
            x_values, fast_values = calculate_fast_metric(all_algorithm_data[algo_name], threshold=0.5, max_step=10)
            print(f"{algo_name}: fast_0.5 at step 10 = {fast_values[-1]:.3f}")
            fast_05_data.append((
                x_values, fast_values, algo_name,
                colors[idx], linestyles[idx], markers[idx]
            ))

    create_plot(fast_05_data, "kernel_fast_0.5", "0.5", "fast_0.5")

    # ========================================================================
    # PLOT 2: fast_1.0
    # ========================================================================
    print("\n" + "="*80)
    print("CREATING PLOT 2: fast_1.0")
    print("="*80)

    fast_10_data = []
    for idx, algo_name in enumerate(algorithm_order):
        if algo_name in all_algorithm_data:
            x_values, fast_values = calculate_fast_metric(all_algorithm_data[algo_name], threshold=1.0, max_step=10)
            print(f"{algo_name}: fast_1.0 at step 10 = {fast_values[-1]:.3f}")
            fast_10_data.append((
                x_values, fast_values, algo_name,
                colors[idx], linestyles[idx], markers[idx]
            ))

    create_plot(fast_10_data, "kernel_fast_1.0", "1.0", "fast_1.0")

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == "__main__":
    main()
