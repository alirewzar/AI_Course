import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Set the plot style to dark background
plt.style.use('dark_background')

# Hypothesis: h_w(x) = w_0 + w_1 * x_1
def h_w(x, w):
    return w[0] + w[1] * x  # equivalent to w_0 + w_1 * x

def generate_data(n=50, noise=5.0):
    np.random.seed(42)
    X = np.linspace(-10, 10, n)
    # Ground truth line: y = 3x + 8
    true_slope = 3
    true_intercept = 8
    noise = np.random.randn(n) * noise
    Y = true_slope * X + true_intercept + noise
    return X, Y

# MSE cost function
def cost_function(X, Y, w):
    return np.sum((h_w(X, w) - Y)**2) / len(X)

# Gradient descent
def gradient_descent(X, Y, w, alpha, num_iters):
    m = len(X)
    cost_history = []
    w_history = [w.copy()]

    for i in range(num_iters):
        # updates
        gradient_w0 = np.sum(h_w(X, w) - Y) / m
        gradient_w1 = np.sum((h_w(X, w) - Y) * X) / m
        w[0] -= alpha * gradient_w0
        w[1] -= alpha * gradient_w1

        cost_history.append(cost_function(X, Y, w))
        w_history.append(w.copy())  # Store a copy of w, not the reference

    return w, cost_history, w_history

# Create cost function surface for visualization
def create_cost_surface(X, Y, w0_range, w1_range):
    w0_vals = np.linspace(w0_range[0], w0_range[1], 50)
    w1_vals = np.linspace(w1_range[0], w1_range[1], 50)
    w0_mesh, w1_mesh = np.meshgrid(w0_vals, w1_vals)
    
    J_vals = np.zeros(w0_mesh.shape)
    for i in range(w0_mesh.shape[0]):
        for j in range(w0_mesh.shape[1]):
            J_vals[i, j] = cost_function(X, Y, [w0_mesh[i, j], w1_mesh[i, j]])
    
    # Apply logarithmic transformation to cost values
    log_J_vals = np.log10(J_vals + 1e-10)  # Add small constant to avoid log(0)
    
    return w0_mesh, w1_mesh, log_J_vals

# Main function
def main():
    # Generate data
    X, Y = generate_data(n=50, noise=5.0)
    w_initial = [0, 0]  # Start with w0 = 0, w1 = 0
    eta = 0.05  # Learning rate
    num_iters = 100  # Reduced for animation clarity
    
    # Run gradient descent
    w_final, cost_history, w_history = gradient_descent(X, Y, w_initial.copy(), eta, num_iters)
    
    # Create cost surface for visualization
    w0_range = [-10, 20]
    w1_range = [-1, 5]
    w0_mesh, w1_mesh, log_J_vals = create_cost_surface(X, Y, w0_range, w1_range)
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(15, 6))
    fig.patch.set_facecolor('black')  # Set figure background to black
    
    ax1 = fig.add_subplot(121)  # Data and regression line
    ax2 = fig.add_subplot(122, projection='3d')  # Cost surface
    
    # Set both subplot backgrounds to black
    ax1.set_facecolor('black')
    ax2.set_facecolor('black')
    
    # Set the color of the pane in 3D subplot
    ax2.xaxis.set_pane_color((0, 0, 0, 0.8))
    ax2.yaxis.set_pane_color((0, 0, 0, 0.8))
    ax2.zaxis.set_pane_color((0, 0, 0, 0.8))
    
    # Set grid color in 3D subplot
    ax2.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0.2)
    ax2.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0.2)
    ax2.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0.2)
    
    # Set up data subplot
    ax1.scatter(X, Y, color='cyan', alpha=0.8, label='Data Points')
    line, = ax1.plot([], [], color='yellow', lw=2, label='Regression Line')
    title1 = ax1.set_title('Iteration: 0', color='white', fontsize=12)
    ax1.set_xlim(min(X), max(X))
    ax1.set_ylim(min(Y) - 5, max(Y) + 5)
    ax1.set_xlabel('X', color='white')
    ax1.set_ylabel('Y', color='white')
    ax1.tick_params(colors='white')
    ax1.legend()
    
    # Set up cost surface subplot
    surf = ax2.plot_surface(w0_mesh, w1_mesh, log_J_vals, cmap='plasma', alpha=0.8)
    
    # Initialize the path line and current point
    path_line, = ax2.plot([], [], [], color='lime', lw=2, label='GD Path')
    current_point, = ax2.plot([], [], [], 'o', color='yellow', markersize=10, label='Current Position')
    
    title2 = ax2.set_title('Log10(Cost) Surface with GD Path', color='white', fontsize=12)
    ax2.set_xlabel('w0 (Intercept)', color='white')
    ax2.set_ylabel('w1 (Slope)', color='white')
    ax2.set_zlabel('Log10(Cost)', color='white')
    ax2.tick_params(colors='white')
    
    # Set the view angle for better visualization
    ax2.view_init(elev=30, azim=45)
    
    # Legend for 3D plot
    ax2.legend()
    
    # Animation update function
    def update(frame):
        # Update regression line
        w = w_history[frame]
        line.set_data(X, h_w(X, w))
        title1.set_text(f'Iteration: {frame}, Cost: {cost_history[frame]:.2f}')
        
        # Update path on cost surface - show the path so far
        w0_path = [w_history[i][0] for i in range(frame + 1)]
        w1_path = [w_history[i][1] for i in range(frame + 1)]
        
        # Apply logarithmic transformation to the cost path values
        log_cost_path = [np.log10(cost_function(X, Y, w_history[i]) + 1e-10) for i in range(frame + 1)]
        
        path_line.set_data(w0_path, w1_path)
        path_line.set_3d_properties(log_cost_path)
        
        # Update current point with logarithmic cost
        current_point.set_data([w[0]], [w[1]])
        current_point.set_3d_properties([np.log10(cost_history[frame] + 1e-10)])
        
        return line, path_line, current_point, title1, title2
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=num_iters, interval=100, blit=False)
    
    plt.tight_layout()
    plt.show()
    
    print("Final weights:", w_final)
    print("Final cost:", cost_history[-1])
    
    # To save the animation (optional)
    # ani.save('gradient_descent.mp4', writer='ffmpeg')

if __name__ == "__main__":
    main()