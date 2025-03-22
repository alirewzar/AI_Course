import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from IPython.display import HTML
import matplotlib as mpl

# Set style for better visualization
plt.style.use('ggplot')
mpl.rcParams['font.family'] = 'DejaVu Sans'

# Function to generate polynomial features
def polynomial_features(X, degree):
    """Transform input array X into a polynomial feature matrix."""
    X_poly = np.ones((len(X), 1))  # Add a column of 1s for the intercept
    for i in range(1, degree + 1):
        X_poly = np.column_stack((X_poly, X**i))  # Add X^i column
    return X_poly

# Function to perform polynomial regression
def polynomial_regression(X, y, degree):
    """Fit a polynomial regression model of specified degree."""
    X_poly = polynomial_features(X.reshape(-1, 1), degree)
    # Closed-form solution: w = (X^T * X)^-1 * X^T * y
    w = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y
    return w

# Function to generate synthetic non-linear data
def generate_nonlinear_data(n=100, noise_level=3.0):
    """Generate non-linear data: y = 2x^2 - 5x + 3 + noise."""
    np.random.seed(42)
    X = np.linspace(-3, 3, n)
    # True function: y = 2x^2 - 5x + 3
    true_a, true_b, true_c = 2, -5, 3
    noise = np.random.randn(n) * noise_level
    y = true_a * X**2 + true_b * X + true_c + noise
    return X, y, (true_a, true_b, true_c)

# Function to make predictions with a polynomial model
def predict(X, w):
    """Make predictions using the polynomial model."""
    X_poly = polynomial_features(X.reshape(-1, 1), len(w)-1)
    return X_poly @ w

# Function to calculate RMSE (Root Mean Squared Error)
def compute_rmse(y_true, y_pred):
    """Calculate RMSE between true and predicted values."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Generate data
X, y, true_coeffs = generate_nonlinear_data(n=50, noise_level=5.0)
true_a, true_b, true_c = true_coeffs

# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Maximum degree to consider
max_degree = 20

# Create a smooth grid of X values for plotting the fit lines
X_fit = np.linspace(X.min(), X.max(), 1000)

# Add the true function curve
y_true_curve = true_a * X_fit**2 + true_b * X_fit + true_c

# Pre-calculate all models and errors
all_w = []
train_rmses = []
test_rmses = []
coefficients = []  # Store coefficients for visualization

for degree in range(1, max_degree + 1):
    # Fit polynomial regression
    w = polynomial_regression(X_train, y_train, degree)
    all_w.append(w)
    coefficients.append(w.copy())
    
    # Make predictions
    y_train_pred = predict(X_train, w)
    y_test_pred = predict(X_test, w)
    
    # Calculate RMSE
    train_rmse = compute_rmse(y_train, y_train_pred)
    test_rmse = compute_rmse(y_test, y_test_pred)
    
    train_rmses.append(train_rmse)
    test_rmses.append(test_rmse)

# Find the best, most underfitting, and most overfitting models
degrees = list(range(1, max_degree + 1))
best_degree = degrees[np.argmin(test_rmses)]
underfit_degree = 1  # Typically the linear model underfits quadratic data
rmse_gaps = [test - train for test, train in zip(test_rmses, train_rmses)]
overfit_degree = degrees[np.argmax(rmse_gaps)]

# Create the animation with vertical layout
fig = plt.figure(figsize=(20, 13))
fig.suptitle("Polynomial Regression: Understanding Bias-Variance Tradeoff", fontsize=18, fontweight='bold', y=0.98)

# Create a GridSpec layout with vertical arrangement
gs = plt.GridSpec(3, 2, width_ratios=[2, 1], height_ratios=[1, 1, 1], hspace=0.4)

# Main plot for the polynomial fit
ax1 = fig.add_subplot(gs[0:2, 0])  # Top left, spanning 2 rows
# RMSE plot
ax2 = fig.add_subplot(gs[0, 1])  # Top right
# Gap plot
ax3 = fig.add_subplot(gs[1, 1])  # Middle right
# Coefficient plot
ax4 = fig.add_subplot(gs[2, :])  # Bottom spanning both columns

# Set up main plot (Data visualization)
ax1.scatter(X_train, y_train, color='#3498db', alpha=0.8, s=80, label='Training Data', edgecolor='white')
ax1.scatter(X_test, y_test, color='#e74c3c', alpha=0.8, s=80, label='Test Data', edgecolor='white')
ax1.plot(X_fit, y_true_curve, color='#2c3e50', linestyle='--', linewidth=2, alpha=0.7, label='True Function: $y = 2x^2 - 5x + 3$')
poly_line, = ax1.plot([], [], color='#2ecc71', linewidth=3, label='Polynomial Fit')
ax1.set_title('Polynomial Function Fitting to Data', fontsize=16, pad=10)
ax1.set_xlabel('X', fontsize=14)
ax1.set_ylabel('y', fontsize=14)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='upper left', framealpha=0.8, fontsize=12)

# Set axis limits with some padding
ax1.set_xlim(X.min() - 0.5, X.max() + 0.5)
ax1.set_ylim(min(y.min(), y_true_curve.min()) - 5, max(y.max(), y_true_curve.max()) + 5)

# Set up RMSE plot
train_line, = ax2.plot([], [], color='#3498db', linewidth=3, label='Training RMSE')
test_line, = ax2.plot([], [], color='#e74c3c', linewidth=3, label='Test RMSE')
optimal_line = ax2.axvline(x=best_degree, color='#2ecc71', linestyle='--', alpha=0.7, label=f'Optimal Degree: {best_degree}')
degree_point, = ax2.plot([], [], 'o', color='black', markersize=10, markerfacecolor='white')
ax2.set_xlim(0.5, max_degree + 0.5)
ax2.set_ylim(0, max(max(train_rmses), max(test_rmses)) * 1.1)
ax2.set_title('Error Metrics: Training vs Test', fontsize=14, pad=10)
ax2.set_xlabel('Polynomial Degree', fontsize=12)
ax2.set_ylabel('RMSE', fontsize=12)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='upper right', framealpha=0.8, fontsize=10)

# Set up gap plot
gap_line, = ax3.plot([], [], color='#9b59b6', linewidth=3, label='Overfitting Gap (Test - Train)')
gap_optimal = ax3.axvline(x=best_degree, color='#2ecc71', linestyle='--', alpha=0.7, label=f'Optimal Degree: {best_degree}')
gap_point, = ax3.plot([], [], 'o', color='black', markersize=10, markerfacecolor='white')
ax3.set_xlim(0.5, max_degree + 0.5)
ax3.set_ylim(- max(rmse_gaps) * 1.1, max(rmse_gaps) * 1.1)
ax3.set_title('Measuring Overfitting', fontsize=14, pad=10)
ax3.set_xlabel('Polynomial Degree', fontsize=12)
ax3.set_ylabel('Gap Size', fontsize=12)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.legend(loc='upper left', framealpha=0.8, fontsize=10)

# Set up coefficient plot
bar_container = ax4.bar(np.arange(max_degree + 1), np.zeros(max_degree + 1), 
                        color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c', '#34495e', '#7f8c8d'])
ax4.set_xlim(-0.5, max_degree + 0.5)
ax4.set_xticks(range(0, max_degree + 1))
ax4.set_xticklabels(['$w_0$'] + [f'$w_{{{i}}}$' for i in range(1, max_degree + 1)])
ax4.set_title('Polynomial Coefficients', fontsize=14)
ax4.set_xlabel('Coefficient', fontsize=12)
ax4.set_ylabel('Value', fontsize=12)
ax4.tick_params(axis='x', rotation=45)
ax4.grid(True, alpha=0.3, linestyle='--', axis='y')

# Add text annotations for model classification
model_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, fontsize=14,
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.5'))

# Educational text explaining the current state
educational_text = fig.text(0.5, 0.01, '', fontsize=12, ha='center', va='center',
                           bbox=dict(facecolor='#f8f9fa', alpha=0.8, boxstyle='round,pad=0.5'))

def init():
    poly_line.set_data([], [])
    train_line.set_data([], [])
    test_line.set_data([], [])
    degree_point.set_data([], [])
    gap_line.set_data([], [])
    gap_point.set_data([], [])
    model_text.set_text('')
    educational_text.set_text('')
    # Initialize all bars to zero height
    for bar in bar_container:
        bar.set_height(0)
    return poly_line, train_line, test_line, degree_point, gap_line, gap_point, model_text, educational_text, bar_container

def update(frame):
    degree = frame + 1  # Degrees start from 1
    
    # Update polynomial fit line
    y_fit = predict(X_fit, all_w[frame])
    poly_line.set_data(X_fit, y_fit)
    
    # Update RMSE plot
    train_line.set_data(degrees[:degree], train_rmses[:degree])
    test_line.set_data(degrees[:degree], test_rmses[:degree])
    degree_point.set_data([degree], [test_rmses[frame]])
    
    # Update gap plot
    gap_line.set_data(degrees[:degree], rmse_gaps[:degree])
    gap_point.set_data([degree], [rmse_gaps[frame]])
    
    # Update coefficients plot - display current degree's coefficients
    coeffs = np.zeros(max_degree + 1)
    curr_coeffs = coefficients[frame]
    coeffs[:len(curr_coeffs)] = curr_coeffs
    
    for i, bar in enumerate(bar_container):
        if i < len(curr_coeffs):
            bar.set_height(curr_coeffs[i])
            # Color the bars differently based on coefficient importance
            if i == 0:  # Intercept
                bar.set_color('#3498db')  # Blue
            elif i <= 2:  # Lower-order terms (more important for quadratic function)
                bar.set_color('#2ecc71')  # Green  
            else:  # Higher-order terms (noise capture in overfitting)
                bar.set_color('#e74c3c')  # Red
        else:
            bar.set_height(0)
    
    # Update coefficient plot y-limits dynamically
    coeff_min = min(coeffs) - 1 if len(coeffs) > 0 else -1
    coeff_max = max(coeffs) + 1 if len(coeffs) > 0 else 1
    ax4.set_ylim(coeff_min, coeff_max)
    
    # Update model classification text
    if degree < best_degree:
        model_status = "UNDERFITTING"
        color = '#3498db'  # Blue
    elif degree == best_degree:
        model_status = "OPTIMAL FIT"
        color = '#2ecc71'  # Green
    else:
        model_status = "OVERFITTING"
        color = '#e74c3c'  # Red
    
    model_text.set_text(f'Degree {degree}: {model_status}\nTrain RMSE: {train_rmses[frame]:.2f}\nTest RMSE: {test_rmses[frame]:.2f}')
    model_text.set_bbox(dict(facecolor='white', alpha=0.7, edgecolor=color, boxstyle='round,pad=0.5'))
    
    # Update educational text based on the current state
    if degree < best_degree:
        edu_text = "UNDERFITTING: The model is too simple to capture the true pattern in the data. " + \
                  "This results in high bias (systematic error) but low variance. " + \
                  "Both training and test errors are high."
    elif degree == best_degree:
        edu_text = "OPTIMAL FIT: The model complexity is just right! " + \
                  "It balances bias and variance to capture the underlying pattern without fitting to noise. " + \
                  "This gives the best generalization to unseen data."
    else:
        edu_text = "OVERFITTING: The model is too complex and captures noise in the training data. " + \
                  "This results in low bias but high variance. " + \
                  "Training error continues to decrease while test error increases."
    
    educational_text.set_text(edu_text)
    
    # Update main plot title
    ax1.set_title(f'Polynomial Regression - Degree {degree}', fontsize=16)
    
    return poly_line, train_line, test_line, degree_point, gap_line, gap_point, model_text, educational_text, bar_container

# Create animation
ani = FuncAnimation(fig, update, frames=max_degree, init_func=init, blit=False, interval=800)

plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.08)  # Make room for title and educational text
plt.show()

# To save animation for sharing (optional)
# ani.save('polynomial_regression_animation.mp4', writer='ffmpeg', fps=1.5, dpi=300)

# For Jupyter - return HTML animation
try:
    from IPython.display import HTML
    HTML(ani.to_jshtml())
except:
    print("Animation created. If running in Jupyter, import HTML from IPython.display to view it inline.")