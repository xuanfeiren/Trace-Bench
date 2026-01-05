"""
Demo script to test the CirclePackingGuide on a specific construction.
Standalone version without opto dependencies.
"""

import numpy as np
import itertools


class CirclePackingGuide:
    """Guide for evaluating and providing feedback on circle packing solutions."""
    
    def __init__(self, num_circles: int = 26):
        """
        Initialize the CirclePackingGuide.
        
        Args:
            num_circles: Number of circles to pack (default: 26)
        """
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
            circles: The circle parameters array or callable
            info: Additional info (optional)
            
        Returns:
            tuple: (score, feedback_message)
        """
        # Ensure circles is a numpy array
        circles_array = np.array(circles, dtype=np.float64)
        
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
                    feedback_parts.append(f"  • Circles {i} and {j}: overlap by {overlap_amt:.4f} (radii: {r1:.4f}, {r2:.4f})")
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

def demo_guide():
    """Test the guide on a specific construction."""
    construction_1 = np.array([
        [0.68180423, 0.90401948, 0.09598051040194801],
        [0.90598057, 0.49972596, 0.094019420598057],
        [0.08464495, 0.08464502, 0.084644941535505],
        [0.4818405, 0.1034156, 0.10341558965844],
        [0.48234279, 0.89652324, 0.10347674965232401],
        [0.88928295, 0.11071646, 0.11071644892835401],
        [0.07852351, 0.50033803, 0.07852350214764901],
        [0.27330428, 0.1051224, 0.10512238948776001],
        [0.38153556, 0.70211016, 0.115517618448237],
        [0.13252625, 0.70386646, 0.132047406795258],
        [0.59610341, 0.72720176, 0.10051153994884501],
        [0.59576502, 0.2725969, 0.10054788994521],
        [0.88902242, 0.88902318, 0.11097680890231801],
        [0.68087256, 0.09573004, 0.095730030426996],
        [0.53098174, 0.49996661, 0.13587084641291403],
        [0.90742967, 0.68631006, 0.09257029074297],
        [0.74197859, 0.40386139, 0.095943250405674],
        [0.08471273, 0.91528735, 0.08471263152873601],
        [0.76300349, 0.7598366, 0.069494603050539],
        [0.27358888, 0.89472069, 0.105279299472069],
        [0.13242993, 0.29639184, 0.132426596757339],
        [0.90745313, 0.31316575, 0.092546580745341],
        [0.76284098, 0.23948486, 0.06975189302481001],
        [0.38095848, 0.29797266, 0.115741008425898],
        [0.74207757, 0.59573774, 0.09593309040669],
        [0.27141024, 0.50032568, 0.114361818563817]
    ])
    
    print("Testing CirclePackingGuide on construction_1")
    print("=" * 70)
    print(f"Construction shape: {construction_1.shape}")
    print(f"Number of circles: {len(construction_1)}")
    print()
    
    # Create guide
    guide = CirclePackingGuide(num_circles=26)
    
    # Get feedback
    score, feedback = guide.get_feedback(dummy_input="dummy", circles=construction_1)
    
    # Print results
    print(f"Score: {score}")
    print("\nFeedback:")
    print(feedback)
    print("\n" + "=" * 70)
    
    return score, feedback


if __name__ == "__main__":
    demo_guide()

