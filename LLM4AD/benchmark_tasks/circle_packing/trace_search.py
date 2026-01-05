"""
Circle Packing Optimization using PrioritySearch Algorithm

Problem:
    Pack 26 circles within a unit square [0,1]×[0,1] to maximize the sum of their radii.
    
Constraints:
    - All circles must be fully contained within the square boundary
    - No circles may overlap (must be pairwise disjoint)
    
Challenge:
    This constrained optimization combines discrete placement decisions with continuous 
    radius optimization. The problem exhibits multiple local optima and requires 
    sophisticated search strategies to avoid suboptimal configurations.

Agent:
    CircleAgent contains circle parameters (x, y, radius) that are optimized via 
    PrioritySearch until a valid, high-scoring solution is found.

Dataset:
    Single-task benchmark (dummy dataset with one optimization task).

Optimization Process:
    1. Initialize agent with initial circle parameters
    2. Sample candidates and evaluate using CirclePackingGuide
    3. Optimizer proposes new parameters based on feedback from selected candidates
    4. Iterate until maximum score is achieved or step limit is reached
"""

import sys
import os

import numpy as np
import torch,secrets_local,itertools , time, argparse,litellm
np.random.seed(10)
torch.manual_seed(10)

from opto import trace
from opto.trainer.loggers import DefaultLogger, WandbLogger
from opto.optimizers import OptoPrimeV2
from opto.optimizers.utils import print_color

from opto.trainer.guide import Guide

litellm.drop_params = True
litellm.suppress_debug_info = True


@trace.model
class CircleAgent:
    """
    An agent that is a container for Circles. 
    The circles array is stored as a string internally (for hashability) 
    but converted to numpy array on output.
    """

    def __init__(self):
        """
        Initialize the agent with initial circles array.
        """
        initial_circles = np.zeros((26, 3))
        # Convert to string representation for hashability in PrioritySearch
        circles_str = str(initial_circles.tolist())
        self.circles = trace.node(circles_str, trainable=True)
        # self.circles = trace.node(initial_circles, trainable=True)

    @trace.bundle()
    def get_circles(self, dummy_input: str, circles: str) -> np.ndarray:
        """
        Get the circles array. Converts from string to numpy array.
        
        Args:
            dummy_input: Dummy input (not used)
            circles_str: The current circles array as string (trainable parameter)
            
        Returns:
            The circles array as numpy array
        """
        import ast
        try:
            circles_list = ast.literal_eval(circles)
            return np.array(circles_list, dtype=np.float64)
        except:
            print_color(f"Error parsing circles string: {circles}", 'red')
            # If parsing fails, return zeros
            return np.zeros((26, 3))
        # return circles

    def forward(self, dummy_input: str) -> np.ndarray:
        """
        Forward pass that returns the current circles as numpy array.
        
        Args:
            dummy_input: The dummy input for the task
            
        Returns:
            The circles array as numpy array
        """
        return self.get_circles(dummy_input, self.circles)



class CirclePackingGuide(Guide):
    """Guide for evaluating and providing feedback on circle packing solutions."""
    
    def __init__(self, num_circles: int = 26):
        """
        Initialize the CirclePackingGuide.
        
        Args:
            num_circles: Number of circles to pack (default: 26)
        """
        super().__init__()
        self.n = num_circles
    
    def verify_circles(self, circles: np.ndarray) -> tuple[bool, str, dict]:
        """
        Checks that the circles are disjoint and lie inside a unit square.

        Args:
            circles: A numpy array of shape (num_circles, 3), where each row is
                of the form (x, y, radius), specifying a circle.

        Returns:
            tuple: (is_valid, error_message, details_dict) with specific problem details
        """
        details = {'overlap_pairs': [], 'boundary_violations': []}
        
        try:
            if len(circles) != self.n:
                return False, f"Expected {self.n} circles, but got {len(circles)}", details
            
            # Check for invalid values (NaN, inf, negative radii)
            if np.any(~np.isfinite(circles)):
                return False, "Contains NaN or infinite values", details
            
            neg_radii_indices = np.where(circles[:, 2] < 0)[0]
            if len(neg_radii_indices) > 0:
                details['negative_radii'] = neg_radii_indices.tolist()
                return False, f"Circles {neg_radii_indices.tolist()} have negative radii", details
            
            # Check pairwise disjointness
            circle_indices = list(range(len(circles)))
            for i, j in itertools.combinations(circle_indices, 2):
                circle1, circle2 = circles[i], circles[j]
                center_distance = np.sqrt((circle1[0] - circle2[0]) ** 2 + (circle1[1] - circle2[1]) ** 2)
                radii_sum = circle1[2] + circle2[2]
                overlap = radii_sum - center_distance
                if overlap > 0:
                    details['overlap_pairs'].append({
                        'circles': (i, j),
                        'overlap_amount': float(overlap),
                        'centers': ((float(circle1[0]), float(circle1[1])), 
                                   (float(circle2[0]), float(circle2[1]))),
                        'radii': (float(circle1[2]), float(circle2[2]))
                    })
            
            if details['overlap_pairs']:
                return False, f"{len(details['overlap_pairs'])} pairs overlap", details

            # Check all circles lie inside the unit square [0,1]x[0,1]
            for i, circle in enumerate(circles):
                x, y, r = circle
                violations = []
                if x - r < 0:
                    violations.append(('left', float(-(x - r))))
                if x + r > 1:
                    violations.append(('right', float(x + r - 1)))
                if y - r < 0:
                    violations.append(('bottom', float(-(y - r))))
                if y + r > 1:
                    violations.append(('top', float(y + r - 1)))
                
                if violations:
                    details['boundary_violations'].append({
                        'circle': i,
                        'position': (float(x), float(y)),
                        'radius': float(r),
                        'violations': violations
                    })
            
            if details['boundary_violations']:
                return False, f"{len(details['boundary_violations'])} circles violate boundary", details
            
            return True, "Valid solution", details
        except Exception as e:
            return False, f"Error during verification: {str(e)}", details

    def get_feedback(self, dummy_input, circles, info=None, **kwargs):
        """
        Get feedback from the agent's circle packing solution.
        
        Args:
            dummy_input: Dummy input (not used)
            circles: The circle parameters array
            info: Additional info (optional)
            
        Returns:
            tuple: (score, feedback_message)
        """
        # Ensure circles is a numpy array
        circles_array = np.array(circles, dtype=np.float64)
        # check the dimension of the circles_array
        if circles_array.shape != (26, 3):
            return 0.0, f"Invalid circles array shape: {circles_array.shape}. Expected (26, 3)."
        
        # Verify the solution
        is_valid, error_msg, details = self.verify_circles(circles_array)
        
        if not is_valid:
            # Provide specific feedback about violations
            feedback_parts = [f"INVALID: {error_msg}"]
            
            # Detail overlap violations
            if details.get('overlap_pairs'):
                feedback_parts.append(f"\nOverlapping pairs ({len(details['overlap_pairs'])} total):")
                for overlap_info in details['overlap_pairs']:
                    i, j = overlap_info['circles']
                    overlap_amt = overlap_info['overlap_amount']
                    r1, r2 = overlap_info['radii']
                    (x1, y1), (x2, y2) = overlap_info['centers']
                    # Show full precision for overlap to reveal floating point errors
                    feedback_parts.append(
                        f"  • Circle {i} ({x1:.4f}, {y1:.4f}, {r1:.4f}) and "
                        f"Circle {j} ({x2:.4f}, {y2:.4f}, {r2:.4f}): overlap by {overlap_amt:.15f}"
                    )
                feedback_parts.append(f"\nFix: Reduce radii or increase distance between these circles.")
            
            # Detail boundary violations
            if details.get('boundary_violations'):
                feedback_parts.append(f"\nBoundary violations ({len(details['boundary_violations'])} total):")
                for viol_info in details['boundary_violations']:
                    i = viol_info['circle']
                    x, y = viol_info['position']
                    r = viol_info['radius']
                    sides = [v[0] for v in viol_info['violations']]
                    feedback_parts.append(f"  • Circle {i} at ({x:.4f}, {y:.4f}) r={r:.4f}: exceeds {', '.join(sides)} boundary")
                feedback_parts.append(f"\nFix: Move circles inward or reduce their radii.")
            
            return 0.0, '\n'.join(feedback_parts)
        
        # Calculate score (sum of radii)
        score = np.sum(circles_array[:, 2])
        
        # Simple feedback for valid solutions
        feedback = f"VALID ✓ Score: {score:.4f}\nTo increase score: try increasing radii or repositioning circles for better packing."
        return score, feedback

    def metric(self, task, response, info=None, **kwargs):
        """
        Calculate the metric score for the agent's performance.
        
        Args:
            task: The task being evaluated
            response: The circle packing solution
            info: Additional info (optional)
            
        Returns:
            float: The score (sum of radii for valid solutions, -inf otherwise)
        """
        score, feedback = self.get_feedback(task, response, info, **kwargs)

        # print_color(f"Score: {score:.4f}", 'green')
        # print_color(feedback, 'yellow')
        return score


def create_single_task_dataset():
    """Create a dummy dataset for the task"""
    return {
        'inputs': ["dummy_input"],
        'infos': ["dummy_info"]
    }

def main():
    parser = argparse.ArgumentParser(description='Optimize a single circle packing solution using PrioritySearch')
    parser.add_argument('--num_steps', type=int, default=10, help='Maximum number of optimization steps')
    parser.add_argument('--num_candidates', type=int, default=1, help='Number of candidates for exploration')
    parser.add_argument('--num_threads', type=int, default=20, help='Number of threads for parallel processing')
    parser.add_argument('--num_proposals', type=int, default=1, help='Number of proposals for each candidate')
    parser.add_argument('--log_frequency', type=int, default=1, help='How often to log results')
    parser.add_argument('--test_frequency', type=int, default=1, help='How often to run evaluation')
    parser.add_argument('--use_best_candidate_to_explore', action='store_true', default=False, help='Use the best candidate to explore')

    parser.add_argument('--algorithm', type=str, default='PS',choices=['PS','PS_Summarizer','PS_epsNet_Summarizer','PS_epsNet'], help='Algorithm to use')

    parser.add_argument('--use_wandb', action='store_true', default=False,
                       help='Whether to use Weights & Biases for logging')
    parser.add_argument('--project_name', type=str, default='circle-packing',
                       help='Name of the W&B project')
    parser.add_argument('--run_name', type=str, default='circle-packing-run',
                       help='Name of the W&B run')
    
    args = parser.parse_args()

    num_steps = args.num_steps
    num_threads = args.num_threads
    log_frequency = args.log_frequency
    test_frequency = args.test_frequency
    num_proposals = args.num_proposals
    # Step 1: Load the task
    task = create_single_task_dataset()
    
    # Step 2: Initialize the agent with a dummy circles string
    agent = CircleAgent()
    
    # Step 4: Initialize the optimizer with a clear objective
    if os.environ.get("TRACE_LITELLM_MODEL") == "gemini/gemini-2.5-flash-lite":
        max_tokens = 65536
    else:
        max_tokens = 8192
    optimizer = OptoPrimeV2(agent.parameters(), max_tokens=max_tokens, initial_var_char_limit=10000)
    optimizer.objective = f"""Your task is to pack 26 circles in a unit square [0,1]×[0,1] to MAXIMIZE the sum of their radii.

CONSTRAINTS (must satisfy ALL):
1. Boundary constraint: Each circle must be fully inside the unit square
   - For circle at (x, y) with radius r: 0 ≤ x-r AND x+r ≤ 1 AND 0 ≤ y-r AND y+r ≤ 1

2. Non-overlap constraint: No two circles may overlap
   - For any two circles: distance_between_centers ≥ sum_of_radii
   - Distance formula: sqrt((x1-x2)² + (y1-y2)²)

3. Exactly 26 circles required

GOAL: MAXIMIZE sum of all radii while satisfying all constraints.

FORMAT: Provide a (26, 3) array where each row is [x_center, y_center, radius].

You will see:
- Current circles in # Variables
- Evaluation feedback in # Feedback (tells you what's wrong or how to improve)

Strategy: Start with valid small circles, then iteratively increase radii and optimize positions to maximize the sum.
"""

    # Step 5: Initialize guide and logger
    guide = CirclePackingGuide(num_circles=26)
    
    # Create config dictionary for logging
    config_dict = {
        'num_steps': num_steps,
        'num_candidates': args.num_candidates,
        'num_threads': num_threads,
        'num_proposals': num_proposals,
        'log_frequency': log_frequency,
        'test_frequency': test_frequency,
    }
    
    # Set run name if not provided
    run_name = args.run_name
    
    # Initialize logger based on wandb flag
    if args.use_wandb:
        logger = WandbLogger(project=args.project_name, verbose=True, name=run_name, config=config_dict)
    else:
        logger = DefaultLogger(verbose=True)
    
    # Step 6: Use the single-task dataset
    train_dataset = task
    
    # Step 7: Create PrioritySearch algorithm
    print("\nCreating PrioritySearch algorithm...")
    if args.use_wandb:
        print(f"Using Weights & Biases logging: project='{args.project_name}', run='{run_name}'")
    else:
        print("Using DefaultLogger (no W&B logging)")

    # Algorithm selection
    if args.algorithm == 'PS':
        from opto.features.priority_search.priority_search import PrioritySearch
        algorithm = PrioritySearch(agent=agent, optimizer=optimizer, logger=logger, num_threads=num_threads)
    # elif args.algorithm == 'PS_Summarizer':
    #     from opto.features.priority_search.priority_search_ablation import PS_veribench
    #     algorithm = PS_veribench(
    #         epsilon=0.0,
    #         use_summarizer=True,
    #         summarizer_model_name="claude-3-5-sonnet",
    #         agent=agent,
    #         optimizer=optimizer,
    #         logger=logger,
    #         num_threads=num_threads
    #     )
    # elif args.algorithm == 'PS_epsNet_Summarizer':
    #     from opto.features.priority_search.priority_search_ablation import PS_veribench
    #     algorithm = PS_veribench(
    #         epsilon=0.1,
    #         use_summarizer=True,
    #         summarizer_model_name="claude-3-5-sonnet",
    #         agent=agent,
    #         optimizer=optimizer,
    #         logger=logger)
    # elif args.algorithm == 'PS_epsNet':
    #     from opto.features.priority_search.priority_search_ablation import PS_veribench
    #     algorithm = PS_veribench(
    #         epsilon=0.1,
    #         agent=agent,
    #         optimizer=optimizer,
    #         logger=logger)
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
    
    # Step 8: Run PrioritySearch training
    print(f"\nStarting PrioritySearch optimization (max {num_steps} steps)...")
    print(f"Target: Maximize sum of circle radii (valid packing)")
    
    start_time = time.time()
    
    algorithm.train(
        guide=guide,
        train_dataset=train_dataset,
        validate_dataset=train_dataset,  # Same task for validation
        test_dataset=train_dataset,       # Same task for test
        batch_size=1,
        num_batches=1,
        num_steps=num_steps,
        num_threads=num_threads,
        num_eval_samples=1,
        validate_exploration_candidates=False,
        num_candidates=args.num_candidates,
        num_proposals=num_proposals,
        score_function='mean',
        log_frequency=log_frequency,
        test_frequency=test_frequency,
        use_best_candidate_to_explore=args.use_best_candidate_to_explore,
    )
    
    duration = time.time() - start_time
    print(f"\nOptimization completed in {duration:.2f} seconds")
    
    # Step 9: Print final result
    dummy_input = task['inputs'][0]
    # Agent returns numpy array directly via forward()
    final_circles_array = agent.forward(dummy_input).data
    final_score, feedback = guide.get_feedback(dummy_input=dummy_input, circles=final_circles_array)
    
    print(f"\n{'='*70}")
    print(f"FINAL RESULT - Score: {final_score}")
    print(f"{'='*70}")
    print(feedback)
    print(f"\nFinal circles (shape: {final_circles_array.shape}):")
    print("-" * 50)
    print(final_circles_array)
    print("-" * 50)


if __name__ == "__main__":
    main()
