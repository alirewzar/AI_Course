import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
import matplotlib.pyplot as plt

def compute_rms_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def generate_data(n=100, noise=10.0):
    np.random.seed(42)
    X = np.random.uniform(-10, 10, n)
    y = X**2 - 2 * X + np.random.randn(n) * noise  # x**2 - 2*x + noise
    return X, y

X, y = generate_data(n=15)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

degrees = [2, 4, 6, 8]
lambdas = [1e4, 1, 1e-4, 1e-8]

ridge_rmse_train = np.zeros((len(degrees), len(lambdas)))
ridge_rmse_test = np.zeros((len(degrees), len(lambdas)))
lasso_rmse_train = np.zeros((len(degrees), len(lambdas)))
lasso_rmse_test = np.zeros((len(degrees), len(lambdas)))

for degree_idx, degree in enumerate(degrees):
    fig, axs = plt.subplots(1, 4, figsize=(18, 6))  # Create a 1x4 grid of subplots
    for lambda_idx, lambda_val in enumerate(lambdas):
        poly_features = PolynomialFeatures(degree=degree)
        X_train_poly = poly_features.fit_transform(X_train[:, np.newaxis])
        X_test_poly = poly_features.transform(X_test[:, np.newaxis])

        # Ridge Regression using scikit-learn
        ridge_model = Ridge(alpha=lambda_val)
        ridge_model.fit(X_train_poly, y_train)
        y_train_pred_ridge = ridge_model.predict(X_train_poly)
        y_test_pred_ridge = ridge_model.predict(X_test_poly)

        # Lasso Regression using scikit-learn
        lasso_model = Lasso(alpha=lambda_val, max_iter=10000)
        lasso_model.fit(X_train_poly, y_train)
        y_train_pred_lasso = lasso_model.predict(X_train_poly)
        y_test_pred_lasso = lasso_model.predict(X_test_poly)

        ridge_rmse_train[degree_idx, lambda_idx] = compute_rms_error(y_train, y_train_pred_ridge)
        ridge_rmse_test[degree_idx, lambda_idx] = compute_rms_error(y_test, y_test_pred_ridge)
        lasso_rmse_train[degree_idx, lambda_idx] = compute_rms_error(y_train, y_train_pred_lasso)
        lasso_rmse_test[degree_idx, lambda_idx] = compute_rms_error(y_test, y_test_pred_lasso)

        # Plot the fitted curves for both Ridge and Lasso
        X_plot = np.linspace(-10, 10, 100)
        X_plot_poly = poly_features.transform(X_plot[:, np.newaxis])

        y_plot_ridge = ridge_model.predict(X_plot_poly)
        y_plot_lasso = lasso_model.predict(X_plot_poly)

        ax = axs[lambda_idx]
        ax.scatter(X_train, y_train, color='blue', label='Train Data')
        ax.scatter(X_test, y_test, color='green', label='Test Data')
        ax.plot(X_plot, y_plot_ridge, color='red', label=f'Ridge (λ={lambda_val})')
        ax.plot(X_plot, y_plot_lasso, color='orange', linestyle='--', label=f'Lasso (λ={lambda_val})')
        ax.set_title(f'Polynomial Degree {degree}    ,   λ={lambda_val} \n ridge_rmse_train = {compute_rms_error(y_train, y_train_pred_ridge)} \nridge_rmse_test = {compute_rms_error(y_test, y_test_pred_ridge)}\nlasso_rmse_train = {compute_rms_error(y_train, y_train_pred_lasso)}\nlasso_rmse_test = {compute_rms_error(y_test, y_test_pred_lasso)}')
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.legend()

    plt.suptitle(f'Polynomial Degree {degree} - Regularization Comparison')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Create a single figure with two subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 10))

# Plot Ridge RMSE for train datas
for degree_idx, degree in enumerate(degrees):
    axes[0][0].plot(lambdas, ridge_rmse_train[degree_idx], marker='x', label=f'Ridge - Degree {degree}')
axes[0][0].set_xscale('log')
axes[0][0].set_xlabel('Regularization Parameter (λ)')
axes[0][0].set_ylabel('RMSE')
axes[0][0].set_title('RMSE for Ridge at Different Polynomial Degrees for train datas')
axes[0][0].legend()
axes[0][0].grid(True)

# Plot Ridge RMSE for test datas
for degree_idx, degree in enumerate(degrees):
    axes[0][1].plot(lambdas, ridge_rmse_test[degree_idx], marker='x', label=f'Ridge - Degree {degree}')
axes[0][1].set_xscale('log')
axes[0][1].set_xlabel('Regularization Parameter (λ)')
axes[0][1].set_ylabel('RMSE')
axes[0][1].set_title('RMSE for Ridge at Different Polynomial Degrees for test datas')
axes[0][1].legend()
axes[0][1].grid(True)


# Plot Lasso RMSE
for degree_idx, degree in enumerate(degrees):
    axes[1][0].plot(lambdas, lasso_rmse_train[degree_idx], marker='x', label=f'Lasso - Degree {degree}')
axes[1][0].set_xscale('log')
axes[1][0].set_xlabel('Regularization Parameter (λ)')
axes[1][0].set_ylabel('RMSE')
axes[1][0].set_title('RMSE for Lasso at Different Polynomial Degrees for train datas')
axes[1][0].legend()
axes[1][0].grid(True)

# Plot Lasso RMSE
for degree_idx, degree in enumerate(degrees):
    axes[1][1].plot(lambdas, lasso_rmse_test[degree_idx], marker='x', label=f'Lasso - Degree {degree}')
axes[1][1].set_xscale('log')
axes[1][1].set_xlabel('Regularization Parameter (λ)')
axes[1][1].set_ylabel('RMSE')
axes[1][1].set_title('RMSE for Lasso at Different Polynomial Degrees for test datas')
axes[1][1].legend()
axes[1][1].grid(True)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
