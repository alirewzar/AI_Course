from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

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

# SSE cost function
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

# SSE cost function
def cost_function(X, Y, w):
    return np.sum((h_w(X, w) - Y)**2) / len(X)


# Visualize cost function (log of J(w))
w0_vals = np.linspace(-10, 20, 100)
w1_vals = np.linspace(-1, 5, 100)

X, Y = generate_data(n=50, noise=5.0)
w_initial = [0, 0]  # Start with w0 = 0, w1 = 0
eta = 0.05  # Learning rate
num_iters = 500

# Run Gradient Descent
w_final, cost_history, w_history = gradient_descent(X, Y, w_initial, eta, num_iters)

J_vals = np.zeros((len(w0_vals), len(w1_vals)))
for i in range(len(w0_vals)):
    for j in range(len(w1_vals)):
        w = [w0_vals[i], w1_vals[j]]
        J_vals[i, j] = cost_function(X, Y, w)

# Create a single figure with two subplots (1 row, 2 columns)
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 7))  # Wider figure for side-by-side plots

# Convert history to arrays
w_history_array = np.array(w_history)  
w0_history = w_history_array[:, 0]
w1_history = w_history_array[:, 1]
cost_history_log = np.log(np.array(cost_history))  # Log of cost history

### **Left Subplot: 3D Surface Plot**
ax3d = fig.add_subplot(1, 2, 1, projection='3d')  # Add 3D subplot
W0, W1 = np.meshgrid(w0_vals, w1_vals)
ax3d.plot_surface(W0, W1, np.log(J_vals.T), cmap='viridis', alpha=0.25)

# Plot gradient descent path in 3D
ax3d.plot(w0_history[:num_iters], w1_history[:num_iters], cost_history_log, marker='o', color='r', markersize=3, label='GD Path')

ax3d.set_xlabel('w0')
ax3d.set_ylabel('w1')
ax3d.set_zlabel('log(J(w))')
ax3d.set_title("Cost Function Surface (Log Scale)")
ax3d.legend()

### **Right Subplot: 2D Gradient Descent Path**
axs[1].plot(w0_history, w1_history, marker='o', linestyle='-', color='b', label="GD Path")
axs[1].set_xlabel('w0')
axs[1].set_ylabel('w1')
axs[1].set_title("Gradient Descent Path on w0-w1 Plane")
axs[1].legend()
axs[1].grid(True)

# Adjust layout for better spacing
plt.tight_layout()
plt.show()