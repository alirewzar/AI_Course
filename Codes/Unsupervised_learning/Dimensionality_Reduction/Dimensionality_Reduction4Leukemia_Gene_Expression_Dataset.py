
#4. Case Study: Leukemia Gene Expression Dataset
'''
Dataset Description
The Leukemia Gene Expression Dataset consists of 72 samples with 7,129 gene expression features. The dataset includes 
two classes of leukemia:

Acute Myeloid Leukemia (AML)
Acute Lymphoblastic Leukemia (ALL)
Due to the high dimensionality (7,129 features) and the small number of samples (72), 
this dataset is prone to overfitting. Dimensionality reduction techniques like PCA can be 
effective in preventing overfitting by reducing the number of features.

Data Preprocessing and Visualization
We will:

Load the dataset.
Preprocess the data.
Apply PCA to reduce dimensionality.
Visualize the results.
Evaluate the performance of a logistic regression classifier with and without PCA.
'''

# Import necessary libraries
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')  # Suppress warning messages for cleaner output

# Load the Leukemia dataset from OpenML
# Dataset ID 1104 corresponds to the Leukemia dataset with gene expression data
leukemia = fetch_openml(data_id=1104, as_frame=False)
X = leukemia.data  # Features: gene expression values
y = leukemia.target  # Target: cancer type (AML or ALL)

# Convert string labels to numerical values for model compatibility
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y_encoded))}")

# Data Preprocessing
# Standardization is crucial for PCA and logistic regression to work effectively
# It ensures all features are on the same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model Training and Evaluation
# Using Logistic Regression with 'saga' solver for efficient handling of large feature sets
model = LogisticRegression(max_iter=1000, solver='saga')

# Use Stratified K-Fold to maintain class distribution in each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate model performance on raw data (without dimensionality reduction)
scores_no_pca = cross_val_score(model, X_scaled, y_encoded, cv=cv, scoring='accuracy')
mean_score_no_pca = np.mean(scores_no_pca)

print(f"\nCross-validation accuracy without PCA: {mean_score_no_pca:.4f}")

# Applying PCA
# Reduce dimensionality to 20 components while preserving maximum variance
pca = PCA(n_components=20, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Evaluate model performance with reduced dimensions
scores_pca = cross_val_score(model, X_pca, y_encoded, cv=cv, scoring='accuracy')
mean_score_pca = np.mean(scores_pca)

print(f"Cross-validation accuracy with PCA (20 components): {mean_score_pca:.4f}")

# Experimenting with Different Numbers of Components
# Test various numbers of principal components to find optimal dimensionality
components = [5, 10, 20, 50]
scores = []

for n in components:
    # For each number of components, perform PCA and evaluate model
    pca = PCA(n_components=n, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    score = np.mean(cross_val_score(model, X_pca, y_encoded, cv=cv, scoring='accuracy'))
    scores.append(score)
    print(f"Accuracy with PCA ({n} components): {score:.4f}")


# Visualization of results
# Create comparison plot of PCA components vs accuracy
plt.figure(figsize=(8, 6))
plt.plot(components, scores, marker='o', label='With PCA')
plt.axhline(y=mean_score_no_pca, color='r', linestyle='--', label='Without PCA')
plt.title('Model Accuracy vs. Number of PCA Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cross-validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

