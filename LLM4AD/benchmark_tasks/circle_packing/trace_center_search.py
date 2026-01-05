"""
Circle Packing 

In this file, we only use Trace to search for the center positions of the circles.
The radii are computed using a linear programming solver.

"""

import sys
import os

import numpy as np
import torch,secrets_local,itertools , time, argparse,litellm
from scipy.optimize import linprog
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
    An agent that searches for optimal circle centers.
    The centers are stored as a string internally (for hashability in PrioritySearch).
    The get_circles method uses linear programming to find optimal radii given the centers.
    """

    def __init__(self):
        """
        Initialize the agent with initial circle centers.
        """
        initial_centers = np.random.uniform(0.1, 0.9, (26, 2))
        # initial_centers = np.zeros((26, 2))
        # Convert to string representation for hashability in PrioritySearch
        centers_str = str(initial_centers.tolist())
        self.circle_centers = trace.node(centers_str, trainable=True)

    @trace.bundle()
    def get_circles(self, dummy_input: str, centers: str) -> np.ndarray:
        """
        Get the circles array by solving a linear program for optimal radii.
        
        Given fixed centers, solves:
            maximize sum(r_i)
            subject to:
                r_i >= 0
                r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i (boundary)
                r_i + r_j <= dist_ij for all i<j (non-overlap)
        
        Args:
            dummy_input: Dummy input (not used)
            centers: The circle centers as string (trainable parameter)
            
        Returns:
            The circles array as numpy array with shape (26, 3): [x, y, radius]
        """
        import ast
        try:
            # Parse centers string - only accepts literal values
            centers_list = ast.literal_eval(centers)
            # Use high precision float64 (double precision)
            centers_array = np.array(centers_list, dtype=np.float64)
            
            if centers_array.shape != (26, 2):
                print_color(f"Agent forward error: Invalid centers shape: {centers_array.shape}, expected (26, 2)", 'red')
                return np.zeros((26, 3))
            
            # check if the centers are inside [0, 1]×[0, 1]
            if np.any(centers_array[:, 0] < 0) or np.any(centers_array[:, 0] > 1) or np.any(centers_array[:, 1] < 0) or np.any(centers_array[:, 1] > 1):
                print_color(f"Agent forward error: Invalid centers: {centers_array}. Centers must be inside [0, 1]×[0, 1].", 'red')
                return np.zeros((26, 3))
            
            # Solve LP for optimal radii given these centers
            radii = self._solve_lp_for_radii(centers_array)
            
            # Combine centers and radii
            circles = np.column_stack([centers_array, radii])
            return circles
            
        except SyntaxError as e:
            print_color(f"Agent forward error: [SYNTAX ERROR] Invalid Python syntax in centers string", 'red')
            print_color(f"Error: {str(e)}", 'red')
            print_color(f"Centers string:\n{centers}", 'yellow')
            return np.zeros((26, 3))
        except ValueError as e:
            print_color(f"Agent forward error: [VALUE ERROR] Centers contain expressions/comprehensions (not allowed)", 'red')
            print_color(f"Error: {str(e)}", 'red')
            print_color(f"Centers must be literal values only - compute and write actual numbers!", 'yellow')
            print_color(f"String preview: {centers[:300]}...", 'cyan')
            return np.zeros((26, 3))
        except Exception as e:
            print_color(f"Agent forward error: [ERROR] in get_circles: {type(e).__name__}: {str(e)}", 'red')
            print_color(f"Centers string (first 300 chars): {centers[:300]}", 'yellow')
            return np.zeros((26, 3))

    def _solve_lp_for_radii(self, centers: np.ndarray) -> np.ndarray:
        """
        Solve linear program to find optimal radii given fixed centers.
        
        Args:
            centers: Array of shape (26, 2) with circle centers
            
        Returns:
            Array of shape (26,) with optimal radii
        """
        n = len(centers)
        
        # Objective: maximize sum(r_i) => minimize -sum(r_i)
        c = -np.ones(n, dtype=np.float64)
        
        # Inequality constraints: A_ub @ r <= b_ub
        A_ub = []
        b_ub = []
        
        # Boundary constraints for each circle
        for i in range(n):
            x_i, y_i = np.float64(centers[i, 0]), np.float64(centers[i, 1])
            
            # r_i <= x_i (left boundary)
            constraint = np.zeros(n, dtype=np.float64)
            constraint[i] = np.float64(1.0)
            A_ub.append(constraint)
            b_ub.append(np.float64(x_i))
            
            # r_i <= 1 - x_i (right boundary)
            constraint = np.zeros(n, dtype=np.float64)
            constraint[i] = np.float64(1.0)
            A_ub.append(constraint)
            b_ub.append(np.float64(np.float64(1.0) - x_i))
            
            # r_i <= y_i (bottom boundary)
            constraint = np.zeros(n, dtype=np.float64)
            constraint[i] = np.float64(1.0)
            A_ub.append(constraint)
            b_ub.append(np.float64(y_i))
            
            # r_i <= 1 - y_i (top boundary)
            constraint = np.zeros(n, dtype=np.float64)
            constraint[i] = np.float64(1.0)
            A_ub.append(constraint)
            b_ub.append(np.float64(np.float64(1.0) - y_i))
        
        # Non-overlap constraints for each pair
        for i in range(n):
            for j in range(i + 1, n):
                # Calculate distance with high precision float64
                diff = centers[i].astype(np.float64) - centers[j].astype(np.float64)
                dist_ij = np.float64(np.linalg.norm(diff, ord=2))
                
                # r_i + r_j <= dist_ij (non-overlap constraint)
                constraint = np.zeros(n, dtype=np.float64)
                constraint[i] = np.float64(1.0)
                constraint[j] = np.float64(1.0)
                A_ub.append(constraint)
                b_ub.append(np.float64(dist_ij))
        
        # Convert to high precision arrays
        A_ub = np.array(A_ub, dtype=np.float64)
        b_ub = np.array(b_ub, dtype=np.float64)
        
        # Bounds: r_i >= 0
        bounds = [(0, None) for _ in range(n)]
        
        # Solve LP with HiGHS solver options
        options = {
            'presolve': True,    # Enable presolve for better conditioning
            'disp': False,       # Suppress solver output
            'maxiter': 10000     # Maximum iterations
        }
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options)
        
        if result.success:
            # Clip to ensure non-negative radii and convert to float64
            radii = np.maximum(result.x.astype(np.float64), np.float64(0.0))
            return radii
        else:
            # If LP fails, return small positive radii
            print_color(f"LP solver failed: {result.message}", 'yellow')
            # print centers
            print_color(f"Centers: {centers}", 'yellow')
            breakpoint()
            return np.full(n, np.float64(0), dtype=np.float64)

    def forward(self, dummy_input: str) -> np.ndarray:
        """
        Forward pass that returns the current circles as numpy array.
        
        Args:
            dummy_input: The dummy input for the task
            
        Returns:
            The circles array as numpy array with shape (26, 3)
        """
        return self.get_circles(dummy_input, self.circle_centers)




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
            
            # Check pairwise disjointness with numerical tolerance
            # Use tolerance to account for floating-point precision errors
            tolerance = 1e-9
            circle_indices = list(range(len(circles)))
            for i, j in itertools.combinations(circle_indices, 2):
                circle1, circle2 = circles[i], circles[j]
                center_distance = np.sqrt((circle1[0] - circle2[0]) ** 2 + (circle1[1] - circle2[1]) ** 2)
                radii_sum = circle1[2] + circle2[2]
                overlap = radii_sum - center_distance
                if overlap > tolerance:
                    details['overlap_pairs'].append({
                        'circles': (i, j),
                        'overlap_amount': float(overlap),
                        'centers': ((float(circle1[0]), float(circle1[1])), 
                                   (float(circle2[0]), float(circle2[1]))),
                        'radii': (float(circle1[2]), float(circle2[2]))
                    })
            
            if details['overlap_pairs']:
                return False, f"{len(details['overlap_pairs'])} pairs overlap", details

            # Check all circles lie inside the unit square [0,1]x[0,1] with tolerance
            tolerance = 1e-9
            for i, circle in enumerate(circles):
                x, y, r = circle
                violations = []
                if x - r < -tolerance:
                    violations.append(('left', float(-(x - r))))
                if x + r > 1 + tolerance:
                    violations.append(('right', float(x + r - 1)))
                if y - r < -tolerance:
                    violations.append(('bottom', float(-(y - r))))
                if y + r > 1 + tolerance:
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

        # check if the centers are inside [0, 1]×[0, 1]
        if np.any(circles_array[:, 0] < 0) or np.any(circles_array[:, 0] > 1) or np.any(circles_array[:, 1] < 0) or np.any(circles_array[:, 1] > 1):
            return 0.0, f"Invalid centers: {circles_array}. Centers must be inside [0, 1]×[0, 1]."
        
        # # Verify the solution
        # is_valid, error_msg, details = self.verify_circles(circles_array)

        # assert is_valid, f"Invalid circles array: {circles_array}"
        
        
        # Calculate score (sum of radii)
        score = np.sum(circles_array[:, 2])
        
        # Simple feedback for valid solutions
        feedback = f"Score: {score:.5f}. Adjust centers to improve packing. CURRENT SOTA SCORE: 2.64 (this is the target to beat! scores around 2.54 is far from the target)"

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

    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon for the epsilon-net algorithm')

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
    optimizer.objective = f"""PROBLEM: Pack 26 circles in a unit square [0,1]×[0,1] to MAXIMIZE the sum of their radii.

CONSTRAINTS:
1. All circles fully inside [0,1]×[0,1]
2. No overlapping circles (pairwise disjoint)

YOUR TASK: Propose ONLY the CENTER POSITIONS (x, y) for 26 circles.
- A linear programming solver will automatically compute optimal radii for your centers
- Do NOT set radii yourself

OUTPUT FORMAT (CRITICAL):
- EXACTLY 26 centers: [[x1, y1], [x2, y2], ..., [x26, y26]]
- Each center MUST be inside [0, 1]×[0, 1]: 0 ≤ x ≤ 1 and 0 ≤ y ≤ 1
- MUST write LITERAL NUMERIC VALUES only - compute all values yourself!
  ✓ CORRECT: [[0.1, 0.273205], [0.3, 0.446410], [0.5, 0.619615], ...]
- Use MANY DECIMAL DIGITS for precision (e.g., 0.15384615384615, not just 0.15)

OPTIMIZATION APPROACH:
- You can make minor adjustments to the current circle centers to improve the score
- Or propose a completely new configuration with a different pattern
- Write your final answer with high precision (8-12 decimal digits)

You will see:
- Current centers in # Variables
- Resulting circles (with LP-optimized radii) in # Outputs
- Score in # Feedback
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

    from opto.features.priority_search.priority_search import PrioritySearch
    from opto.features.priority_search.epsNetPS_plus_summarizer import EpsilonNetPS_plus_Summarizer

    # Algorithm selection
    if args.algorithm == 'PS':
        algorithm = PrioritySearch(agent=agent, optimizer=optimizer, logger=logger, num_threads=num_threads)
    elif args.algorithm == 'PS_Summarizer':
        algorithm = EpsilonNetPS_plus_Summarizer(
            epsilon=0.0,
            use_summarizer=True,
            summarizer_model_name="claude-3.5-sonnet",
            agent=agent,
            optimizer=optimizer,
            logger=logger,
            num_threads=num_threads
        )
    elif args.algorithm == 'PS_epsNet_Summarizer':
        algorithm = EpsilonNetPS_plus_Summarizer(
            epsilon=args.epsilon,
            use_summarizer=True,
            summarizer_model_name="claude-3.5-sonnet",
            agent=agent,
            optimizer=optimizer,
            logger=logger,
            num_threads=num_threads
            )
    elif args.algorithm == 'PS_epsNet':
        algorithm = EpsilonNetPS_plus_Summarizer(
            epsilon=args.epsilon,
            agent=agent,
            optimizer=optimizer,
            logger=logger,
            num_threads=num_threads
            )
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
