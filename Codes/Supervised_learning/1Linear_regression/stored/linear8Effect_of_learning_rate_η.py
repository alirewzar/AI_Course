from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

# Hypothesis: h_w(x) = w_0 + w_1 * x_1
def h_w(x, w):
    return w[0] + w[1] * x  # equivalent to w_0 + w_1 * x

# SSE cost function
def cost_function(X, Y, w):
    return np.sum((h_w(X, w) - Y)**2) / len(X)

def generate_data(n=50, noise=5.0):
    np.random.seed(42)
    X = np.linspace(-10, 10, n)
    # Ground truth line: y = 3x + 8
    true_slope = 3
    true_intercept = 8
    noise = np.random.randn(n) * noise
    Y = true_slope * X + true_intercept + noise
    return X, Y

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

X, Y = generate_data(n=50, noise=5.0)

learning_rates = [0.1, 0.02, 0.001]
num_iters = 100
w_initial = [0, 0]

colors = ['purple', 'green', 'orange']

w0_vals = np.linspace(-10, 20, 100)
w1_vals = np.linspace(-1, 5, 100)
J_vals = np.zeros((len(w0_vals), len(w1_vals)))


for i in range(len(w0_vals)):
    for j in range(len(w1_vals)):
        w = [w0_vals[i], w1_vals[j]]
        J_vals[i, j] = cost_function(X, Y, w)

cost_histories = []

# GD for each eta
for idx, eta in enumerate(learning_rates):
    w_final, cost_history, w_history = gradient_descent(X, Y, w_initial.copy(), eta, num_iters)
    cost_histories.append(cost_history)

    plt.figure(figsize=(10, 6))
    for step_idx, w in enumerate(w_history[::num_iters // 100]):
        alpha_val = 0.15 + 0.85*(idx) / 100
        plt.plot(X, h_w(X, w), color=colors[idx], alpha=alpha_val)

    plt.plot(X, h_w(X, w_final), lw=2, label=f'Final Line (eta={eta})', color=colors[idx])
    plt.title(f"Lines during Gradient Descent (Learning Rate {eta})")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()
    plt.scatter(X, Y, color='blue', label='Actual Data')
    plt.show()


plt.title("Cost Function (log scale) over Iterations for Different Learning Rates")
plt.xlabel("Iteration")
plt.ylabel("log(J(w))")
for idx in range(len(cost_histories)):
  plt.plot(np.log(cost_histories[idx]), label=f'eta={learning_rates[idx]}', color=colors[idx])
plt.ylim(bottom=2, top=10)
plt.legend()
plt.show()
