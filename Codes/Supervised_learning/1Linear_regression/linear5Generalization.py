# Import required libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
import matplotlib.pyplot as plt

# Function to compute Root Mean Square Error
def compute_rms_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Function to generate synthetic data following y = x^2 - 2x + noise
def generate_data(n=100, noise=10.0):
    np.random.seed(42)  # Set seed for reproducibility
    X = np.random.uniform(-10, 10, n)  # Generate random X values
    y = X**2 - 2 * X + np.random.randn(n) * noise  # Generate y with quadratic relationship and noise
    return X, y

# Generate dataset with 15 samples
X, y = generate_data(n=15)
# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define polynomial degrees and regularization strengths (lambda values)
degrees = [2, 4, 6, 8]  # Different polynomial degrees to test
ln_lambdas = [10, 5, 0, -5, -10]  # ln(lambda) values
lambdas = np.round(np.exp(ln_lambdas), decimals=3)  # Convert to lambda values with 3 decimal places

# Print the relationship between ln(lambda) and lambda
print("\nRelationship between ln(lambda) and lambda values:")
for ln_lambda, lambda_val in zip(ln_lambdas, lambdas):
    print(f"ln(lambda) = {ln_lambda:6.2f} → lambda = {lambda_val:10.3f}")

print("\nFormulas with actual lambda values:")
print("Ridge Loss = Σ(yi - ŷi)² + λΣwj²")
for lambda_val in lambdas:
    print(f"When λ = {lambda_val:10.8f}: Loss = Σ(yi - ŷi)² + {lambda_val:10.3f}Σwj²")

print("\nLasso Loss = Σ(yi - ŷi)² + λΣ|wj|")
for lambda_val in lambdas:
    print(f"When λ = {lambda_val:10.8f}: Loss = Σ(yi - ŷi)² + {lambda_val:10.3f}Σ|wj|")

# Initialize arrays to store RMSE values
ridge_rmse_train = np.zeros((len(degrees), len(lambdas)))
ridge_rmse_test = np.zeros((len(degrees), len(lambdas)))
lasso_rmse_train = np.zeros((len(degrees), len(lambdas)))
lasso_rmse_test = np.zeros((len(degrees), len(lambdas)))


first_place = 0.05

# Loop through each polynomial degree
for degree_idx, degree in enumerate(degrees):
    # Create subplots for each lambda value
    fig, axs = plt.subplots(1, 5, figsize=(18, 10))
    
    # Loop through each lambda value
    for lambda_idx, lambda_val in enumerate(lambdas):
        # Transform features to polynomial features
        poly_features = PolynomialFeatures(degree=degree)
        X_train_poly = poly_features.fit_transform(X_train[:, np.newaxis])
        X_test_poly = poly_features.transform(X_test[:, np.newaxis])

        # Fit Ridge Regression model
        ridge_model = Ridge(alpha=lambda_val)
        ridge_model.fit(X_train_poly, y_train)
        y_train_pred_ridge = ridge_model.predict(X_train_poly)
        y_test_pred_ridge = ridge_model.predict(X_test_poly)

        # Fit Lasso Regression model
        lasso_model = Lasso(alpha=lambda_val, max_iter=10000)
        lasso_model.fit(X_train_poly, y_train)
        y_train_pred_lasso = lasso_model.predict(X_train_poly)
        y_test_pred_lasso = lasso_model.predict(X_test_poly)

        # Calculate and store RMSE values
        ridge_rmse_train[degree_idx, lambda_idx] = compute_rms_error(y_train, y_train_pred_ridge)
        ridge_rmse_test[degree_idx, lambda_idx] = compute_rms_error(y_test, y_test_pred_ridge)
        lasso_rmse_train[degree_idx, lambda_idx] = compute_rms_error(y_train, y_train_pred_lasso)
        lasso_rmse_test[degree_idx, lambda_idx] = compute_rms_error(y_test, y_test_pred_lasso)

        # Generate points for plotting smooth curves
        X_plot = np.linspace(-10, 10, 100)
        X_plot_poly = poly_features.transform(X_plot[:, np.newaxis])
        y_plot_ridge = ridge_model.predict(X_plot_poly)
        y_plot_lasso = lasso_model.predict(X_plot_poly)

        # Plot the results
        ax = axs[lambda_idx]
        ax.scatter(X_train, y_train, color='blue', label='Train Data')
        ax.scatter(X_test, y_test, color='green', label='Test Data')
        ax.plot(X_plot, y_plot_ridge, color='red', label=f'Ridge (λ={lambda_val})')
        ax.plot(X_plot, y_plot_lasso, color='orange', linestyle='--', label=f'Lasso (λ={lambda_val})')
        
        # Format weights for display
        ridge_weights = np.round(ridge_model.coef_, 3)
        lasso_weights = np.round(lasso_model.coef_, 3)
        
        # Create weight strings with W0, W1, etc.
        ridge_weight_str = "Ridge weights:\n" + "\n".join([f"W{i} = {w}" for i, w in enumerate(ridge_weights)])
        lasso_weight_str = "Lasso weights:\n" + "\n".join([f"W{i} = {w}" for i, w in enumerate(lasso_weights)])
        
        ax.set_title(f'Polynomial Degree {degree}, λ={lambda_val}\nRidge Train RMSE={ridge_rmse_train[degree_idx, lambda_idx]:.2f}, \nTest RMSE={ridge_rmse_test[degree_idx, lambda_idx]:.2f}\nLasso Train RMSE={lasso_rmse_train[degree_idx, lambda_idx]:.2f}, \nTest RMSE={lasso_rmse_test[degree_idx, lambda_idx]:.2f}')
        
        # Add text below the plot
        plt.figtext(first_place, 0.1, ridge_weight_str, fontsize=6, ha='left')
        plt.figtext(first_place + 0.04, 0.1, lasso_weight_str, fontsize=6, ha='left')
        first_place += 0.2

        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.legend()

    plt.suptitle(f'Polynomial Degree {degree} - Effect of Lambda (Regularization Strength)')
    plt.tight_layout(rect=[0, 0, 1, 0.85])  # Adjust rect to make room for weights at bottom
    plt.show()
    first_place = 0.05
    
# Create summary plots comparing RMSE across different lambdas
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 10))

# Plot Ridge RMSE for training data
for degree_idx, degree in enumerate(degrees):
    axes[0][0].plot(ln_lambdas, ridge_rmse_train[degree_idx], marker='x', label=f'Degree {degree}')
axes[0][0].set_xlabel('ln(Lambda) (Regularization Strength)')
axes[0][0].set_ylabel('RMSE')
axes[0][0].set_title('Ridge Regression: Training RMSE vs ln(Lambda)')
axes[0][0].legend()
axes[0][0].grid(True)

# Plot Ridge RMSE for test data
for degree_idx, degree in enumerate(degrees):
    axes[0][1].plot(ln_lambdas, ridge_rmse_test[degree_idx], marker='x', label=f'Degree {degree}')
axes[0][1].set_xlabel('ln(Lambda) (Regularization Strength)')
axes[0][1].set_ylabel('RMSE')
axes[0][1].set_title('Ridge Regression: Test RMSE vs ln(Lambda)')
axes[0][1].legend()
axes[0][1].grid(True)

# Plot Lasso RMSE for training data
for degree_idx, degree in enumerate(degrees):
    axes[1][0].plot(ln_lambdas, lasso_rmse_train[degree_idx], marker='x', label=f'Degree {degree}')
axes[1][0].set_xlabel('ln(Lambda) (Regularization Strength)')
axes[1][0].set_ylabel('RMSE')
axes[1][0].set_title('Lasso Regression: Training RMSE vs ln(Lambda)')
axes[1][0].legend()
axes[1][0].grid(True)

# Plot Lasso RMSE for test data
for degree_idx, degree in enumerate(degrees):
    axes[1][1].plot(ln_lambdas, lasso_rmse_test[degree_idx], marker='x', label=f'Degree {degree}')
axes[1][1].set_xlabel('ln(Lambda) (Regularization Strength)')
axes[1][1].set_ylabel('RMSE')
axes[1][1].set_title('Lasso Regression: Test RMSE vs ln(Lambda)')
axes[1][1].legend()
axes[1][1].grid(True)

plt.tight_layout()
plt.show()
