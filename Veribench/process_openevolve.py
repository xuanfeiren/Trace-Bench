import json
import os
from pathlib import Path

def process_openevolve_results(input_dir, output_file):
    """
    Process OpenEvolve results from input_dir and save to output_file.
    For each task, creates two entries:
    - step 0: lean4_tool_call_num = 0, score = 0.0
    - step 1: lean4_tool_call_num = success_at_metric_call, score = final_score
    """
    results = {}

    # Find all summary files
    summary_files = sorted(Path(input_dir).glob("*_summary.json"))

    for summary_file in summary_files:
        with open(summary_file, 'r') as f:
            data = json.load(f)

        task_idx = data['task_idx']
        task_key = f"task_{task_idx}"

        # If task failed, use 50 for lean4_tool_call_num, otherwise use success_at_metric_call
        lean4_calls = data['success_at_metric_call'] if data['success_at_metric_call'] is not None else 50

        # Create two entries for each task
        results[task_key] = [
            {
                "step": 0,
                "lean4_tool_call_num": 0,
                "score": 0.0
            },
            {
                "step": 1,
                "lean4_tool_call_num": lean4_calls,
                "score": data['final_score']
            }
        ]

    # Sort by task number
    sorted_results = {}
    for key in sorted(results.keys(), key=lambda x: int(x.split('_')[1])):
        sorted_results[key] = results[key]

    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to output file
    with open(output_file, 'w') as f:
        json.dump(sorted_results, f, indent=2)

    print(f"Processed {len(sorted_results)} tasks from {input_dir}")
    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    base_dir = "/Users/xuanfeiren/Documents/Trace-Bench/Veribench"

    # Process OpenEvolve_2
    process_openevolve_results(
        input_dir=f"{base_dir}/results/OpenEvolve_2",
        output_file=f"{base_dir}/data/openevolve/results_2.json"
    )

    # Process OpenEvolve_3
    process_openevolve_results(
        input_dir=f"{base_dir}/results/OpenEvolve_3",
        output_file=f"{base_dir}/data/openevolve/results_3.json"
    )
