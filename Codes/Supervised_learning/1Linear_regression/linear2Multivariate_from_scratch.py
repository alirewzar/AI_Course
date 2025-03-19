from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load Boston housing dataset from OpenML
data = fetch_openml(name="boston", version=1, as_frame=True)
X, y = data.data, data.target

# Combine features and target into a single DataFrame
df = pd.concat([X, y.rename("MEDV")], axis=1)

# Display the first few rows of the DataFrame
print(df.head())

# Plot the correlation matrix
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Pairplot to visualize relationships
sns.pairplot(df, y_vars=["MEDV"], x_vars=df.columns[:-1])
plt.show()

# Select specific features based on correlation analysis
selected_features = ['RM', 'LSTAT']  # Example features
X = X[selected_features]

# Convert columns to float
X = X.astype(float)
y = y.astype(float)

# Min-Max Normalization
X_min = np.min(X, axis=0)
X_max = np.max(X, axis=0)
X_norm = (X - X_min) / (X_max - X_min)

# Min-Max Normalization
y_min = np.min(y, axis=0)
y_max = np.max(y, axis=0)
y_norm = (y - y_min) / (y_max - y_min)

X_train, X_test, y_train, y_test = train_test_split(X_norm, y_norm, test_size=0.2, random_state=42)

# Display the selected features
print("Selected features for training:")
print(X_train.head())

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

# compute predictions on the test set
y_test_pred = X_test @ w

# compute predictions on the training set
y_train_pred = X_train @ w

# Assuming y_test_pred, y_test, y_train_pred, and y_train are your predicted and actual values
# And X_train and X_test are your feature matrices converted to numpy arrays

# Create a mesh grid for the surface plot
rm_range = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)
lstat_range = np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 100)
rm_grid, lstat_grid = np.meshgrid(rm_range, lstat_range)

# Flatten the grid to pass through the model
grid_points = np.c_[rm_grid.ravel(), lstat_grid.ravel()]

# Predict using the model (assuming w is your weight vector)
z_pred_train = grid_points @ w
z_pred_train = z_pred_train.reshape(rm_grid.shape)

# Create a mesh grid for the test surface plot
rm_range_test = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 100)
lstat_range_test = np.linspace(X_test[:, 1].min(), X_test[:, 1].max(), 100)
rm_grid_test, lstat_grid_test = np.meshgrid(rm_range_test, lstat_range_test)

# Flatten the grid to pass through the model
grid_points_test = np.c_[rm_grid_test.ravel(), lstat_grid_test.ravel()]

# Predict using the model for test data
z_pred_test = grid_points_test @ w
z_pred_test = z_pred_test.reshape(rm_grid_test.shape)

# Plot for Training Data
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X_train[:, 0], X_train[:, 1], y_train, color='green', label='Actual Train Values', alpha=0.5)
ax1.plot_surface(rm_grid, lstat_grid, z_pred_train, color='orange', alpha=0.3, label='Train Prediction Surface')
ax1.set_xlabel('Average Number of Rooms (RM)')
ax1.set_ylabel('Lower Status of Population (LSTAT)')
ax1.set_zlabel('Median Value of Homes (MEDV)')
ax1.set_title('Training Data: Predicted Surface vs Actual Values')
ax1.legend()

# Plot for Test Data
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(X_test[:, 0], X_test[:, 1], y_test, color='blue', label='Actual Test Values', alpha=0.5)
ax2.plot_surface(rm_grid_test, lstat_grid_test, z_pred_test, color='red', alpha=0.3, label='Test Prediction Surface')
ax2.set_xlabel('Average Number of Rooms (RM)')
ax2.set_ylabel('Lower Status of Population (LSTAT)')
ax2.set_zlabel('Median Value of Homes (MEDV)')
ax2.set_title('Test Data: Predicted Surface vs Actual Values')
ax2.legend()

plt.tight_layout()
plt.show()


# Assuming y_pred and y_test are your predicted and actual values for the test set
# And y_train_pred and y_train are your predicted and actual values for the training set

# Plot for Test Data
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_test_pred, alpha=0.5, color='blue', label='Test Data')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values (Test Data)')
plt.legend()
plt.show()

# Plot for Training Data
plt.figure(figsize=(10, 5))
plt.scatter(y_train, y_train_pred, alpha=0.5, color='green', label='Training Data')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values (Training Data)')
plt.legend()
plt.show()


# Compute MSE
mse = np.mean((y_test - y_test_pred) ** 2)
print("Mean Squared Error (MSE) on test set:", mse)

mse_train = np.mean((y_train - y_train_pred) ** 2)
print("Mean Squared Error (MSE) on training set:", mse_train)


print(y_max)
print(y_min)

# Reverse normalization for predictions
y_pred_original = y_test_pred * (y_max - y_min) + y_min
y_test_original = y_test * (y_max - y_min) + y_min

y_train_original = y_train * (y_max - y_min) + y_min
y_train_pred_original = y_train_pred * (y_max - y_min) + y_min

print("y test pred original :", y_pred_original[0:5])
print("y test actual original :", y_test_original[0:5])
print("y train pred original :", y_train_pred_original[0:5])
print("y train actual original :", y_train_original[0:5])

# Function to categorize data into specified number of sections
def categorize_data(data, num_sections, y_min, y_max):
    section_edges = np.linspace(y_min, y_max, num_sections + 1)
    categories = np.digitize(data, section_edges) - 1
    return categories, section_edges


# Number of sections to divide the data into
num_sections = 10  # You can change this value as needed

# Categorize predictions and actual values
y_pred_class, section_edges = categorize_data(y_pred_original, num_sections, y_min, y_max)
y_test_class,_ = categorize_data(y_test_original, num_sections, y_min, y_max)

# Categorize training predictions and actual values
y_train_pred_class, _ = categorize_data(y_train_pred_original, num_sections, y_min, y_max)
y_train_class, _ = categorize_data(y_train_original, num_sections, y_min, y_max)

# Print category ranges
print("\n\n Category Ranges:")
for i in range(len(section_edges) - 1):
    print(f"Category {i}: {section_edges[i]} to {section_edges[i+1]}")

print('\n\ny test pred class :', y_pred_class[0:5])
print('y test actual class :', y_test_class[0:5])
print('y train pred class :', y_train_pred_class[0:5])
print('y train actual class :', y_train_class[0:5])

# Compute accuracy for the test set
accuracy_test = np.mean(y_pred_class == y_test_class)
print("\n\nClassification accuracy on test set:", accuracy_test)

# Compute accuracy for the training set
accuracy_train = np.mean(y_train_pred_class == y_train_class)
print("Classification accuracy on training set:", accuracy_train)