"""
Plot the current circle packing solution from terminal
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_circles(circles: np.ndarray, save_path: str = None):
    """
    Plots the circles.
    
    Args:
        circles: Array of shape (n, 3) where each row is [x, y, radius]
        save_path: Optional path to save the plot
    """
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    # Draw unit square boundary
    rect = patches.Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    # Calculate total score
    total_radii = np.sum(circles[:, 2])
    
    # Draw the circles
    for i, circle in enumerate(circles):
        if circle[2] > 0:  # Only draw circles with positive radius
            circ = patches.Circle((circle[0], circle[1]), circle[2], 
                                  edgecolor='blue', facecolor='skyblue', alpha=0.5, linewidth=1.5)
            ax.add_patch(circ)
        else:
            # Mark zero-radius circles with a red dot
            ax.plot(circle[0], circle[1], 'ro', markersize=8, label='Zero radius' if i == 0 else '')

    plt.title(f'Circle Packing: {len(circles)} circles, Sum of radii = {total_radii:.6f}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()


# Circles from terminal output [x, y, radius]
circles = np.array([
    [0.1, 0.1, 0.1],
    [0.1, 0.3, 0.1],
    [0.1, 0.5, 0.1],
    [0.1, 0.7, 0.1],
    [0.1, 0.9, 0.1],
    [0.3, 0.1, 0.1],
    [0.3, 0.3, 0.1],
    [0.3, 0.5, 0.1],
    [0.3, 0.7, 0.1],
    [0.3, 0.9, 0.1],
    [0.5, 0.1, 0.1],
    [0.5, 0.3, 0.1],
    [0.5, 0.5, 0.1],
    [0.5, 0.7, 0.1],
    [0.5, 0.9, 0.1],
    [0.7, 0.1, 0.08586366],
    [0.7, 0.3, 0.1],
    [0.7, 0.5, 0.1],
    [0.7, 0.7, 0.1],
    [0.7, 0.9, 0.1],
    [0.9, 0.1, 0.1],
    [0.9, 0.3, 0.1],
    [0.9, 0.5, 0.1],
    [0.9, 0.7, 0.1],
    [0.9, 0.9, 0.1],
    [0.77132064, 0.02075195, 0.02075195]
])

print(f"Number of circles: {len(circles)}")
print(f"Sum of radii: {np.sum(circles[:, 2]):.6f}")
print(f"Circles with zero radius: {np.sum(circles[:, 2] == 0)}")
print(f"Circles with positive radius: {np.sum(circles[:, 2] > 0)}")

# Plot
plot_circles(circles, save_path="circle_packing_current.png")

