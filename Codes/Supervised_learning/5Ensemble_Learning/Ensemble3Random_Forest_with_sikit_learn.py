from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import pandas as pd

    
breast_cancer = load_breast_cancer()

X, y = breast_cancer.data, breast_cancer.target

print(f"Breast Cancer features: {breast_cancer.feature_names}")
print(f"Breast Cancer target: {breast_cancer.target_names}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Shape of X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"Shape of X_test: {X_test.shape}, y_test: {y_test.shape}")

rf_sklearn = RandomForestClassifier(n_estimators=7, max_depth=1, criterion='entropy', random_state=42)
rf_sklearn.fit(X_train, y_train.ravel())

rf_skl_predictions = rf_sklearn.predict(X_test)

print(f"Sklearn RF Accuracy: {accuracy_score(y_test, rf_skl_predictions):.3f}")
print(f"Sklearn RF F1-Score: {f1_score(y_test, rf_skl_predictions, average='weighted'):.3f}")

for idx, tree in enumerate(rf_sklearn.estimators_):
    plt.figure(figsize=(8,6))
    plot_tree(tree, filled=True, feature_names=list(breast_cancer.feature_names), class_names=list(breast_cancer.target_names))
    plt.title(f"Random Forest Visualization - Tree {idx + 1}")
    plt.tight_layout()
    plt.show()

sample_idx = 112

print(pd.set_option('display.max_columns', None) ,'\n',pd.DataFrame(X_test[112].reshape(1, -1), columns=breast_cancer.feature_names).head())

votes = [tree.predict(X_test[112].reshape(1, -1)) for tree in rf_sklearn.estimators_]
final_prediction = rf_sklearn.predict(X_test[sample_idx].reshape(1, -1))[0]

plt.figure(figsize=(8, 4))
plt.scatter([range(1, len(rf_sklearn.estimators_) + 1)], votes, s=100, alpha=0.7, label='Votes')
plt.axhline(y=final_prediction, color='r', linestyle='--', label='Final Prediction')
plt.yticks([0, 1], ['Class 0', 'Class 1'])
plt.xlabel('Decision Trees')
plt.ylabel('Votes')
plt.title(f'Random Forest: Votes from Each DT for Sample #{sample_idx + 1}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()