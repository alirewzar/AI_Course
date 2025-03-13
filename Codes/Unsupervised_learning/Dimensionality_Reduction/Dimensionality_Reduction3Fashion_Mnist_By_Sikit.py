#PCA using Scikit-Learn's PCA Function
'''
While we've implemented PCA step by step, scikit-learn provides a convenient and optimized PCA class that performs PCA efficiently. 
It handles the computations internally and offers additional functionality, such as selecting the number of components based on the 
explained variance ratio.

Scikit-Learn PCA Parameters
n_components: Number of principal components to keep. If n_components is between 0 and 1, it selects the 
number of components such that the amount of variance that needs to be explained is greater than the 
percentage specified by n_components.

whiten: When True, the components are multiplied by the square root of the eigenvalues to ensure 
uncorrelated outputs with unit variance.

svd_solver: Specifies the algorithm to use for computation. Options include 'auto', 'full', 'arpack', and 'randomized'.

Advantages
Optimized for performance and can handle large datasets efficiently.
Simplifies the process with fewer lines of code.
Provides additional features like inverse transformation, which can be useful for tasks like data reconstruction.
'''

# Import PCA from scikit-learn
from sklearn.decomposition import PCA
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

# Define paths to your CSV files
base_path = "./Codes/Unsupervised_learning/Dimensionality _Reduction"  # Note: Space in path might cause issues
train_file = os.path.join(base_path, "fashion-mnist_train.csv")
test_file = os.path.join(base_path, "fashion-mnist_test.csv")

# Load the CSV files into pandas DataFrames
train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

# Extract labels (y_train, y_test)
y_train = train_df['label'].values  # Assuming 'label' is the column name
y_test = test_df['label'].values

# Extract pixel data (X_train, X_test)
X_train = train_df.drop('label', axis=1).values  # Drop the label column
X_test = test_df.drop('label', axis=1).values

# Reshape the data into 28x28 images (if not already done in CSV)
X_train = X_train.reshape(-1, 28, 28)
X_test = X_test.reshape(-1, 28, 28)

# Combine train and test data for PCA
X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

# Flatten images to create feature vectors for PCA
X_flattened = X.reshape(X.shape[0], -1)

# Standardize the data (important for PCA)
scaler = StandardScaler()
X_flattened = scaler.fit_transform(X_flattened)

# Standardize the data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X_flattened)

# Define class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
               
# Initialize PCA with the desired number of components
k = 2  # Number of principal components
pca = PCA(n_components=k)

# Fit PCA on the standardized data
X_pca_sklearn = pca.fit_transform(X_standardized)

# Create a DataFrame with the projected data
principal_df_sklearn = pd.DataFrame(X_pca_sklearn, columns=[f'PC{i+1}' for i in range(k)])
principal_df_sklearn['label'] = y

# Display the explained variance ratio
print("Explained Variance Ratio by scikit-learn PCA:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {ratio:.4f}")

# Plot the projected data using scikit-learn PCA
plt.figure(figsize=(10, 8))

# No sampling - use the entire dataset
for label in np.unique(principal_df_sklearn['label']):
    label_indices = principal_df_sklearn['label'] == label
    plt.scatter(principal_df_sklearn.loc[label_indices, 'PC1'],
                principal_df_sklearn.loc[label_indices, 'PC2'],
                s=1, alpha=0.5, label=class_names[label])

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Fashion MNIST Dataset (scikit-learn)')
plt.legend(markerscale=6)
plt.grid()
plt.show()

#Comparison and Discussion
'''
Consistency: The results obtained using scikit-learn's PCA are consistent with our step-by-step implementation.
Efficiency: Scikit-learn's PCA is optimized and may be more efficient, especially for large datasets.
Explained Variance: We can easily access the explained variance ratio using the explained_variance_ratio_ attribute.
Ease of Use: Scikit-learn simplifies the PCA process with fewer lines of code and additional functionalities.
'''

