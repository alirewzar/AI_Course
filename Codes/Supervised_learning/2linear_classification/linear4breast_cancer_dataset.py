from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.01, n_epochs=1000):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
        self.errors_ = []  # storing the number of misclassifications in each epoch

    def fit(self, X, y):
        """
        Train the Perceptron model on the provided data.

        Parameters:
        X : array-like, shape = [n_samples, n_features]
            Training vectors.
        y : array-like, shape = [n_samples]
            Target values. Must be +1 or -1.
        """
        #n_samples: Often represents the number of data points or observations in a dataset.
        #n_features: Represents the dimensions or attributes of each data point (i.e., how many features are associated with each sample).
        n_samples, n_features = X.shape

        # starting weights and bias equal zeros
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.n_epochs):
            errors = 0
            for idx in range(n_samples):
                linear_output = np.dot(X[idx], self.weights) + self.bias  # w^T x + b
                y_pred = self._unit_step(linear_output)
                if y[idx] != y_pred: # misclassfied
                    update = self.learning_rate * y[idx]
                    self.weights += update * X[idx]
                    self.bias += update
                    errors += 1
            self.errors_.append(errors)
            # if no errors, convergence achieved
            if errors == 0:
                print(f"Converged after {epoch+1} epochs")
                break

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        X : array-like, shape = [n_samples, n_features]

        Returns:
        array, shape = [n_samples]
            Predicted class labels.
        """
        linear_output = np.dot(X, self.weights) + self.bias
        return self._unit_step(linear_output)

    def _unit_step(self, x):
        return np.where(x >= 0, 1, -1)


data = load_breast_cancer()
X = data.data
y = data.target

df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y

#Show head of datas
'''print(df.head())'''

selected_features = ['mean radius', 'mean texture']
X_selected = df[selected_features].values
y_selected = y  # 0 = malignant, 1 = benign

#Show head of selected datas
'''print(X_selected[45:50])
print(y_selected[45:50])'''

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_selected, test_size=0.2, random_state=42, stratify=y_selected
)

#Count of train and test datas
'''print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")'''

# convert labels: 0 -> -1, 1 -> 1
y_train_perceptron = np.where(y_train == 0, -1, 1)
y_test_perceptron = np.where(y_test == 0, -1, 1)

perceptron = Perceptron(learning_rate=0.01, n_epochs=1000)

perceptron.fit(X_train, y_train_perceptron)

#Show the output bias informations
'''print(f"Final Weights: {perceptron.weights}")
print(f"Final Bias: {perceptron.bias}")'''


# Create a 1x2 subplot layout
fig, axs = plt.subplots(1, 2, figsize=(18, 8))

# Plot 1: Perceptron Learning Progress
axs[0].plot(range(1, len(perceptron.errors_) + 1, 20), perceptron.errors_[::20], marker='o')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Number of Misclassifications')
axs[0].set_title('Perceptron Learning Progress')
axs[0].grid(True)

# Plot 2: Perceptron Decision Boundary and Decision Regions
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))

grid = np.c_[xx.ravel(), yy.ravel()]
Z = perceptron.predict(grid)
Z = Z.reshape(xx.shape)

cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

# Plot the decision regions
axs[1].contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)

if perceptron.weights[1] != 0:
    x_vals = np.array([x_min, x_max])
    y_vals = -(perceptron.weights[0] * x_vals + perceptron.bias) / perceptron.weights[1]
    axs[1].plot(x_vals, y_vals, 'k--', label='Decision Boundary')
else:
    x_val = -perceptron.bias / perceptron.weights[0]
    axs[1].axvline(x=x_val, color='k', linestyle='--', label='Decision Boundary')

# Malignant: 0 (red), Benign: 1 (blue)
axs[1].scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1],
               color='red', marker='o', edgecolor='k', label='Malignant')
axs[1].scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1],
               color='blue', marker='s', edgecolor='k', label='Benign')

axs[1].set_ylim(10, 40)
axs[1].set_xlabel('Mean Radius')
axs[1].set_ylabel('Mean Texture')
axs[1].set_title('Perceptron Decision Boundary and Decision Regions')
axs[1].legend()
axs[1].grid(True)

# Adjust layout
plt.tight_layout()
plt.show()