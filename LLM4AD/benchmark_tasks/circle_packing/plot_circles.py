"""
Plot circle packing solution
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import linprog


def solve_lp_for_radii(centers: np.ndarray) -> np.ndarray:
    """
    Solve linear program to find optimal radii given fixed centers.
    
    Args:
        centers: Array of shape (n, 2) with circle centers
        
    Returns:
        Array of shape (n,) with optimal radii
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
        'presolve': True,
        'disp': False,
        'maxiter': 10000
    }
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options)
    
    if result.success:
        radii = np.maximum(result.x.astype(np.float64), np.float64(0.0))
        return radii
    else:
        print(f"LP solver failed: {result.message}")
        return np.zeros(n, dtype=np.float64)


def plot_circles(circles: np.ndarray, save_path: str = None):
    """
    Plots the circles.
    
    Args:
        circles: Array of shape (n, 3) where each row is [x, y, radius]
        save_path: Optional path to save the plot
    """
    fig, ax = plt.subplots(1, figsize=(7, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')  # Make axes scaled equally.

    # Draw unit square boundary.
    rect = patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    # Calculate total score
    total_radii = np.sum(circles[:, 2])
    
    # Draw the circles.
    for i, circle in enumerate(circles):
        circ = patches.Circle((circle[0], circle[1]), circle[2], 
                              edgecolor='blue', facecolor='skyblue', alpha=0.5, linewidth=1)
        ax.add_patch(circ)
        # Optionally add circle number
        # ax.text(circle[0], circle[1], str(i), ha='center', va='center', fontsize=8)

    plt.title(f'Circle Packing: {len(circles)} circles, Sum of radii = {total_radii:.6f}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def main():
    # Centers from terminal output
    centers = np.array([
        [0.1, 0.1], [0.3, 0.1], [0.5, 0.1], [0.7, 0.1], [0.9, 0.1],
        [0.1, 0.3], [0.3, 0.3], [0.5, 0.3], [0.7, 0.3], [0.9, 0.3],
        [0.1, 0.5], [0.3, 0.5], [0.5, 0.5], [0.7, 0.5], [0.9, 0.5],
        [0.1, 0.7], [0.3, 0.7], [0.5, 0.7], [0.7, 0.7], [0.9, 0.7],
        [0.1, 0.9], [0.3, 0.9], [0.5, 0.9], [0.7, 0.9], [0.9, 0.9],
        [0.2, 0.2]
    ])
    
    print(f"Computing optimal radii for {len(centers)} circle centers...")
    
    # Solve for optimal radii
    radii = solve_lp_for_radii(centers)
    
    # Combine centers and radii
    circles = np.column_stack([centers, radii])
    
    # Print results
    print(f"\nCircle Packing Solution:")
    print(f"Number of circles: {len(circles)}")
    print(f"Sum of radii: {np.sum(radii):.6f}")
    print(f"\nCircles (x, y, radius):")
    for i, circle in enumerate(circles):
        print(f"  {i+1:2d}. ({circle[0]:.3f}, {circle[1]:.3f}), r={circle[2]:.6f}")
    
    # Plot the circles
    plot_circles(circles, save_path="circle_packing_solution.png")


if __name__ == "__main__":
    main()

