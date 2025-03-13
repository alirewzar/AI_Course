import numpy as np
import matplotlib.pyplot as plt

# Function to generate synthetic data for the first two classifications
def generate_ab_class(n_points=100):
    class_A = []
    class_B = []
    while len(class_A) < n_points or len(class_B) < n_points:
        x = np.random.uniform(0, 10)
        y = np.random.uniform(0, 10)
        if y > x and len(class_A) < n_points:
            class_A.append([x, y])
        elif y < x and len(class_B) < n_points:
            class_B.append([x, y])
    return np.array(class_A), np.array(class_B)

# Function to generate synthetic XOR data
def generate_xor_data(n_points=200, seed=42):
    np.random.seed(seed)
    class_A = []
    class_B = []
    while len(class_A) < n_points or len(class_B) < n_points:
        x = np.random.uniform(0, 1)
        y = np.random.uniform(0, 1)
        if (x > 0.5 and y > 0.5) or (x < 0.5 and y < 0.5):
            if len(class_A) < n_points:
                class_A.append([x, y])
        else:
            if len(class_B) < n_points:
                class_B.append([x, y])
    return np.array(class_A), np.array(class_B)

# Function to generate circle and annulus data for classification
def generate_data(n_points=200, seed=42):
    np.random.seed(seed)

    # Class 0: points inside a circle with radius 5
    radius_0 = 5
    theta_0 = np.random.uniform(0, 2 * np.pi, n_points)
    r_0 = radius_0 * np.sqrt(np.random.uniform(0, 1, n_points))  # sqrt for uniform distribution
    x0 = r_0 * np.cos(theta_0)
    y0 = r_0 * np.sin(theta_0)
    class_0 = np.vstack((x0, y0)).T

    # Class 1: points in an annulus between radius 8 and 10
    inner_radius_1 = 8
    outer_radius_1 = 10
    theta_1 = np.random.uniform(0, 2 * np.pi, n_points)
    r_1 = np.sqrt(np.random.uniform(inner_radius_1**2, outer_radius_1**2, n_points))
    x1 = r_1 * np.cos(theta_1)
    y1 = r_1 * np.sin(theta_1)
    class_1 = np.vstack((x1, y1)).T

    return class_0, class_1

# Generate data for all three plots
class_A, class_B = generate_ab_class()
class_A_xor, class_B_xor = generate_xor_data()
class_0, class_1 = generate_data()


# Create a 2x2 subplot layout
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Classification Based on y > x and y < x
axs[0].scatter(class_A[:, 0], class_A[:, 1], color='green', label='Class A (y > x)')
axs[0].scatter(class_B[:, 0], class_B[:, 1], color='orange', label='Class B (y < x)')
axs[0].plot([0, 10], [0, 10], color='black', linestyle='--', label='Decision Boundary (y = x)')
axs[0].set_xlabel('X-axis')
axs[0].set_ylabel('Y-axis')
axs[0].set_title('Classification Based on y > x and y < x')
axs[0].legend()
axs[0].grid(True)

# Plot 2: XOR Classification
axs[1].scatter(class_A_xor[:, 0], class_A_xor[:, 1], color='green', label='Class A (XOR)')
axs[1].scatter(class_B_xor[:, 0], class_B_xor[:, 1], color='orange', label='Class B (XOR)')
axs[1].plot([0, 1], [0, 1], color='black', linestyle='--', label='Attempted Decision Boundary (y = x)')
axs[1].set_xlabel('X-axis')
axs[1].set_ylabel('Y-axis')
axs[1].set_title('XOR Classification and Linear Decision Boundary')
axs[1].legend()
axs[1].grid(True)

# Plot 3: Class 0 and Class 1 Distribution (Circle and Annulus)
axs[2].scatter(class_0[:, 0], class_0[:, 1], color='blue', label='Class 0 (Circle)')
axs[2].scatter(class_1[:, 0], class_1[:, 1], color='red', label='Class 1 (Annulus)')
axs[2].set_xlabel('X-axis')
axs[2].set_ylabel('Y-axis')
axs[2].set_title('Class 0 and Class 1 Distribution')
axs[2].legend()
axs[2].axis('equal')
axs[2].grid(True)

# Adjust layout
plt.tight_layout()
plt.show()
