#!/usr/bin/env python3
"""
Plot fast_p scores for all algorithms in separate figures.

Figure 1: GEPA, DSPy, OpenEvolve, PS, PS_Summarizer, PS-eps0.1-onlyforSummarizer
Figure 2: Four PS variants

fast_p = the ratio of tasks that got a score > p. 
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import glob


def load_algorithm_results(results_dir: Path, algorithm_name: str) -> List[float]:
    """
    Load all result files for a specific algorithm and extract best scores.
    
    Args:
        results_dir: Directory containing result files
        algorithm_name: Name of the algorithm (e.g., 'dspy', 'PS')
    
    Returns:
        List of best scores for all tasks (indexed by task_idx)
    """
    # First try to load from summary.csv (most reliable)
    summary_csv = results_dir / "summary.csv"
    if summary_csv.exists():
        import csv
        scores_dict = {}
        with open(summary_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                task_idx = int(row['task_idx'])
                score = float(row['best_score'])
                scores_dict[task_idx] = score
        
        # Sort by task_idx and return as list
        max_idx = max(scores_dict.keys()) if scores_dict else -1
        scores = [scores_dict.get(i, 0.0) for i in range(max_idx + 1)]
        return scores
    
    # Fall back to individual JSON files
    pattern = results_dir / f"{algorithm_name}_task_*_result.json"
    result_files = sorted(glob.glob(str(pattern)))
    
    if result_files:
        scores = []
        for filepath in result_files:
            with open(filepath, 'r') as f:
                data = json.load(f)
                # Try different keys for score
                score = data.get('best_score')
                if score is None:
                    score = data.get('best_speedup', 0.0)
                # Handle None values (failed tasks)
                if score is None:
                    score = 0.0
                scores.append(score)
        return scores
    
    return []


def calculate_fast_p(scores: List[float], p_values: np.ndarray) -> np.ndarray:
    """
    Calculate fast_p for a range of p values.
    
    fast_p(p) = (number of tasks with score > p) / (total number of tasks)
    
    Args:
        scores: List of scores for all tasks
        p_values: Array of p values to evaluate
    
    Returns:
        Array of fast_p values corresponding to p_values
    """
    total_tasks = len(scores)
    fast_p_values = []
    
    for p in p_values:
        # Count tasks with score > p
        num_tasks_above_p = sum(1 for score in scores if score > p)
        fast_p = num_tasks_above_p / total_tasks if total_tasks > 0 else 0.0
        fast_p_values.append(fast_p)
    
    return np.array(fast_p_values)


def plot_fast_p_curves(
    algorithm_data: Dict[str, List[float]], 
    p_values: np.ndarray,
    output_path: Path = None,
    title: str = "Fast-p Performance Comparison",
    use_distinct_styles: bool = False
):
    """
    Plot fast_p curves for multiple algorithms.
    
    Args:
        algorithm_data: Dict mapping algorithm names to their score lists
        p_values: Array of p values for x-axis
        output_path: Optional path to save the plot
        title: Plot title
        use_distinct_styles: If True, use different line styles and markers for better distinction
    """
    plt.figure(figsize=(10, 6))
    
    # Define colors for different algorithms
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']
    
    # Define different line styles and markers for better distinction
    linestyles = ['-', '--', '-.', ':', '-', '--']
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    
    # Define specific p values where we want to show data points
    marker_p_values = np.array([0, 0.5, 0.8, 1, 1.5, 2, 3])
    
    for idx, (algorithm_name, scores) in enumerate(algorithm_data.items()):
        # Calculate fast_p values for all p_values
        fast_p_values = calculate_fast_p(scores, p_values)
        
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)] if use_distinct_styles else '-'
        marker = markers[idx % len(markers)] if use_distinct_styles else 'o'
        
        # Find indices of marker_p_values in p_values
        marker_indices = [np.argmin(np.abs(p_values - p)) for p in marker_p_values if p <= p_values[-1]]
        marker_p = p_values[marker_indices]
        marker_fast_p = fast_p_values[marker_indices]
        
        # Plot only the markers with connecting lines
        plt.plot(
            marker_p,
            marker_fast_p,
            marker=marker,
            markersize=8,
            linewidth=2.5,
            linestyle=linestyle,
            label=algorithm_name,
            color=color
        )
    
    plt.xlabel('Speedup Threshold (p)', fontsize=12, fontweight='bold')
    plt.ylabel('Fast-p Score (Ratio of Tasks > p)', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # Set axis limits
    plt.xlim(p_values[0] - 0.05, p_values[-1] + 0.05)
    plt.ylim(-0.05, 1.05)
    
    # Add reference lines
    plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    plt.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2.5, label='1x speedup')
    
    plt.tight_layout()
    
    if output_path:
        # Save as PNG
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
        
        # Also save as PDF
        pdf_path = output_path.with_suffix('.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"PDF saved to: {pdf_path}")
    else:
        plt.show()


def print_statistics(algorithm_data: Dict[str, List[float]]):
    """Print statistics for each algorithm."""
    print("=" * 80)
    print("Algorithm Statistics")
    print("=" * 80)
    
    for algorithm_name, scores in algorithm_data.items():
        print(f"\n{algorithm_name}:")
        print(f"  Total tasks: {len(scores)}")
        
        # Filter out zero scores for statistics
        non_zero_scores = [s for s in scores if s > 0]
        
        if non_zero_scores:
            print(f"  Tasks with score > 0: {len(non_zero_scores)} ({len(non_zero_scores)/len(scores)*100:.1f}%)")
            print(f"  Tasks with score > 1.0: {sum(1 for s in scores if s > 1.0)} ({sum(1 for s in scores if s > 1.0)/len(scores)*100:.1f}%)")
            print(f"  Max score: {max(scores):.4f}x")
            print(f"  Mean score (all): {np.mean(scores):.4f}x")
            print(f"  Mean score (non-zero): {np.mean(non_zero_scores):.4f}x")
            print(f"  Median score (all): {np.median(scores):.4f}x")
        else:
            print(f"  No successful tasks (all scores are 0)")
    
    print("\n" + "=" * 80)


def load_all_algorithms(results_dir: Path) -> Dict[str, List[float]]:
    """Load all available algorithm results."""
    all_data = {}
    
    # DSPy
    dspy_dir = results_dir / "kernel_dspy"
    if dspy_dir.exists():
        try:
            scores = load_algorithm_results(dspy_dir, "dspy")
            if scores:
                all_data["DSPy"] = scores
                print(f"Loaded {len(scores)} DSPy results")
        except Exception as e:
            print(f"Warning: Could not load DSPy results: {e}")
    
    # OpenEvolve
    openevolve_dir = results_dir / "kernel_openevolve"
    if openevolve_dir.exists():
        try:
            scores = load_algorithm_results(openevolve_dir, "openevolve")
            if scores:
                all_data["OpenEvolve"] = scores
                print(f"Loaded {len(scores)} OpenEvolve results")
        except Exception as e:
            print(f"Warning: Could not load OpenEvolve results: {e}")
    
    # Priority Search (PS)
    for ps_name in ["kernel_PS", "kernel_ps"]:
        ps_dir = results_dir / ps_name
        if ps_dir.exists():
            try:
                scores = load_algorithm_results(ps_dir, "ps")
                if scores:
                    all_data["PS"] = scores
                    print(f"Loaded {len(scores)} PS results")
                    break
            except Exception as e:
                print(f"Warning: Could not load PS results from {ps_name}: {e}")
    
    # PS eps0.1 Summarizer (Net-Sum)
    ps_eps_summarizer_dir = results_dir / "kernel_ps_eps0.1_summarizer"
    if ps_eps_summarizer_dir.exists():
        try:
            scores = load_algorithm_results(ps_eps_summarizer_dir, "ps_eps0.1_summarizer")
            if scores:
                all_data["PS (ε=0.1, Net-Sum)"] = scores
                print(f"Loaded {len(scores)} PS (ε=0.1, Net-Sum) results")
        except Exception as e:
            print(f"Warning: Could not load PS eps0.1 Summarizer results: {e}")
    
    # PS Summarizer
    ps_summarizer_dir = results_dir / "kernel_ps_summarizer"
    if ps_summarizer_dir.exists():
        try:
            scores = load_algorithm_results(ps_summarizer_dir, "ps_summarizer")
            if scores:
                all_data["PS_Summarizer"] = scores
                print(f"Loaded {len(scores)} PS_Summarizer results")
        except Exception as e:
            print(f"Warning: Could not load PS_Summarizer results: {e}")
    
    # GEPA
    gepa_dir = results_dir / "kernel_gepa"
    if gepa_dir.exists():
        try:
            scores = load_algorithm_results(gepa_dir, "gepa")
            if scores:
                all_data["GEPA"] = scores
                print(f"Loaded {len(scores)} GEPA results")
        except Exception as e:
            print(f"Warning: Could not load GEPA results: {e}")
    
    # PS eps0.1 onlyforSummarizer (only-Sum)
    ps_eps01_only_sum_dir = results_dir / "kernel_kernel-bench-PS-eps0.1-onlyforSummarizer"
    if ps_eps01_only_sum_dir.exists():
        try:
            scores = load_algorithm_results(ps_eps01_only_sum_dir, "ps_eps0.1_only_summarizer")
            if scores:
                all_data["PS (ε=0.1, only-Sum)"] = scores
                print(f"Loaded {len(scores)} PS (ε=0.1, only-Sum) results")
        except Exception as e:
            print(f"Warning: Could not load PS eps0.1 onlyforSummarizer results: {e}")
    
    return all_data


def plot_figure_1(all_data: Dict[str, List[float]], p_values: np.ndarray, output_dir: Path):
    """
    Figure 1: GEPA, DSPy, OpenEvolve, PS, PS_Summarizer, PS-eps0.1-onlyforSummarizer
    """
    print("\n" + "="*60)
    print("Generating Figure 1: All Methods Comparison")
    print("="*60)
    
    # Select algorithms for Figure 1
    figure1_algorithms = [
        "GEPA",
        "DSPy", 
        "OpenEvolve",
        "PS",
        "PS_Summarizer",
        "PS (ε=0.1, only-Sum)"
    ]
    
    figure1_data = {name: all_data[name] for name in figure1_algorithms if name in all_data}
    
    if not figure1_data:
        print("Warning: No data available for Figure 1")
        return
    
    print(f"Plotting {len(figure1_data)} algorithms: {list(figure1_data.keys())}")
    
    # Print statistics for Figure 1
    print_statistics(figure1_data)
    
    # Plot
    output_path = output_dir / "fast_p_comparison_figure1.png"
    plot_fast_p_curves(
        figure1_data,
        p_values,
        output_path=output_path,
        title="KernelBench: Fast-p Performance Comparison (p ∈ [0, 3])",
        use_distinct_styles=True  # Use distinct styles to avoid overlapping lines
    )


def plot_figure_2(all_data: Dict[str, List[float]], p_values: np.ndarray, output_dir: Path):
    """
    Figure 2: Four PS variants
    """
    print("\n" + "="*60)
    print("Generating Figure 2: PS Variants Comparison")
    print("="*60)
    
    # Select PS variants for Figure 2
    figure2_algorithms = [
        "PS",
        "PS_Summarizer",
        "PS (ε=0.1, Net-Sum)",
        "PS (ε=0.1, only-Sum)"
    ]
    
    figure2_data = {name: all_data[name] for name in figure2_algorithms if name in all_data}
    
    if not figure2_data:
        print("Warning: No data available for Figure 2")
        return
    
    print(f"Plotting {len(figure2_data)} PS variants: {list(figure2_data.keys())}")
    
    # Print statistics for Figure 2
    print_statistics(figure2_data)
    
    # Plot
    output_path = output_dir / "fast_p_comparison_figure2.png"
    plot_fast_p_curves(
        figure2_data,
        p_values,
        output_path=output_path,
        title="KernelBench: Fast-p Performance Comparison (p ∈ [0, 3])",
        use_distinct_styles=True  # Use distinct styles to avoid overlapping lines
    )


def main():
    # Setup paths
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results"
    output_dir = script_dir / "figures"
    output_dir.mkdir(exist_ok=True)
    
    # Define p values: 0.0, 0.1, 0.2, ..., 3.0
    p_values = np.arange(0.0, 3.1, 0.1)
    
    # Load all algorithm results
    print("Loading algorithm results...")
    all_data = load_all_algorithms(results_dir)
    
    if not all_data:
        print("Error: No algorithm results found!")
        print(f"Checked directory: {results_dir}")
        return
    
    # Generate Figure 1
    plot_figure_1(all_data, p_values, output_dir)
    
    # Generate Figure 2
    plot_figure_2(all_data, p_values, output_dir)
    
    print("\n" + "="*60)
    print("All figures generated successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
