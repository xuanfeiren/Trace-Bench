# load wandb projects veribench-per-task (vanilla PS) and veribench-per-task_PS_Summarizer
# each run is a task, first do a forward complete until step 9. So the x axis is step 0 to 9, y axis is the score. 1 is success, 0 is failure.
# plot the average score for each step, over 140 tasks. y axis is the pass rate.

import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

# Initialize wandb API
api = wandb.Api()

entity = None  # Set to your wandb entity/username if needed, or None to use default

# Define the projects to compare
projects = {
    "vanilla_PS": "veribench-per-task",
    "PS_Summarizer": "veribench-per-task_PS_Summarizer"
}

# Store results for both projects
results = {}

# Process each project
for project_label, project_name in projects.items():
    print(f"\n{'='*60}")
    print(f"Processing project: {project_name} ({project_label})")
    print(f"{'='*60}")
    
    # Store all task scores (each task has scores for steps 0-9 with forward fill)
    all_tasks_scores = []
    
    num_tasks = 140
    tasks_processed = 0
    
    for task_id in range(num_tasks):
        run_name = f"task_{task_id}"
        
        try:
            # Get all runs and filter by name
            runs = api.runs(project_name, filters={"display_name": run_name})
            
            if len(runs) == 0:
                print(f"Warning: Run {run_name} not found, skipping...")
                continue
            
            run = runs[0]  # Get the first matching run
            
            # Get the history for this run
            history = run.history()
            
            if history.empty:
                print(f"Warning: No history data for {run_name}")
                continue
            
            # Initialize task scores for steps 0-9 with None
            task_scores = [None] * 10
            
            # Check if 'Test/test_score' column exists
            if 'Test/test_score' in history.columns:
                # Each row in history corresponds to a step
                # The _step column tells us which step it is
                if '_step' in history.columns:
                    for _, row in history.iterrows():
                        step = int(row['_step'])
                        if step <= 9:  # Only process steps 0-9
                            score = row['Test/test_score']
                            task_scores[step] = score
                else:
                    # If no _step column, assume rows are in order (step 0, 1, 2, ...)
                    for idx, row in enumerate(history.iterrows()):
                        if idx < 10:
                            score = row[1]['Test/test_score']
                            task_scores[idx] = score
            
            # Apply forward fill: carry forward the last known score
            # This handles cases where experiments stop early (e.g., when score reaches 1)
            for step in range(10):
                if task_scores[step] is None:
                    if step > 0 and task_scores[step - 1] is not None:
                        # Forward fill from previous step
                        task_scores[step] = task_scores[step - 1]
                    else:
                        # If no previous score, assume 0 (failure)
                        task_scores[step] = 0.0
            
            # Add this task's scores to the collection
            all_tasks_scores.append(task_scores)
            
            tasks_processed += 1
            if tasks_processed % 20 == 0:
                print(f"Processed {tasks_processed}/{num_tasks} tasks...")
                
        except Exception as e:
            print(f"Error processing {run_name}: {e}")
            continue
    
    print(f"\nTotal tasks processed for {project_label}: {tasks_processed}/{num_tasks}")
    
    # Store results for this project
    results[project_label] = {
        'scores': np.array(all_tasks_scores),
        'num_tasks': tasks_processed
    }

# Save the test scores for both projects
print(f"\n{'='*60}")
print("Saving test scores...")
print(f"{'='*60}")

for project_label, data in results.items():
    all_tasks_scores = data['scores']
    
    if len(all_tasks_scores) > 0:
        # Save as numpy file
        filename_npy = f'veribench_test_scores_{project_label}.npy'
        np.save(filename_npy, all_tasks_scores)
        print(f"\nSaved {project_label} test scores to '{filename_npy}' (shape: {all_tasks_scores.shape})")
        
        # Save as CSV for human readability
        df = pd.DataFrame(
            all_tasks_scores,
            columns=[f'step_{i}' for i in range(10)],
            index=[f'task_{i}' for i in range(len(all_tasks_scores))]
        )
        filename_csv = f'veribench_test_scores_{project_label}.csv'
        df.to_csv(filename_csv)
        print(f"Saved {project_label} test scores to '{filename_csv}' ({len(all_tasks_scores)} tasks × 10 steps)")

# Calculate pass rates for both projects
steps = list(range(10))
pass_rates_dict = {}

print(f"\n{'='*60}")
print("Calculating pass rates...")
print(f"{'='*60}")

for project_label, data in results.items():
    all_tasks_scores = data['scores']
    pass_rates = []
    
    print(f"\n{project_label}:")
    if len(all_tasks_scores) > 0:
        # Calculate mean score for each step across all tasks
        for step in steps:
            pass_rate = np.mean(all_tasks_scores[:, step])
            pass_rates.append(pass_rate)
            num_success = np.sum(all_tasks_scores[:, step] == 1.0)
            print(f"  Step {step}: Pass Rate: {pass_rate:.2%} ({num_success}/{len(all_tasks_scores)} tasks succeeded)")
    else:
        print("  No tasks processed!")
        pass_rates = [0] * 10
    
    pass_rates_dict[project_label] = pass_rates

# Create the comparison plot
print(f"\n{'='*60}")
print("Creating comparison plot...")
print(f"{'='*60}")

plt.figure(figsize=(14, 7))

# Define colors and markers for each line
styles = {
    "vanilla_PS": {"color": "#1f77b4", "marker": "o", "label": "Vanilla PS"},
    "PS_Summarizer": {"color": "#ff7f0e", "marker": "s", "label": "PS+Summarizer"}
}

# Plot both lines
for project_label, pass_rates in pass_rates_dict.items():
    style = styles[project_label]
    plt.plot(steps, pass_rates, 
             marker=style["marker"], 
             linewidth=2.5, 
             markersize=8,
             color=style["color"],
             label=f"{style['label']} (n={results[project_label]['num_tasks']} tasks)",
             alpha=0.8)
    
    # Add value labels on each point
    for step, rate in zip(steps, pass_rates):
        plt.text(step, rate + 0.02, f'{rate:.1%}', 
                ha='center', va='bottom', fontsize=8, 
                color=style["color"], alpha=0.7)

plt.xlabel('Step', fontsize=13)
plt.ylabel('Pass Rate', fontsize=13)
plt.title('Veribench Pass Rate Comparison: Vanilla PS vs PS+Summarizer', fontsize=15, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim([0, 1.05])
plt.xticks(steps)
plt.yticks(np.arange(0, 1.1, 0.1))

# Format y-axis as percentages
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

# Add legend
plt.legend(loc='best', fontsize=11, framealpha=0.9)

plt.tight_layout()
plt.savefig('veribench_pass_rate_comparison.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved as 'veribench_pass_rate_comparison.png'")
plt.show()