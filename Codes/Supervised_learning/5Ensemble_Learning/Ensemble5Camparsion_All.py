from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, to_graphviz
import pandas as pd

breast_cancer = load_breast_cancer()

X, y = breast_cancer.data, breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train.ravel())

xgb_predictions = xgb_model.predict(X_test)

xgb_model.get_booster().feature_names = list(breast_cancer.feature_names)
graph = to_graphviz(xgb_model)
graph

def test_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)

    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_predictions = xgb_model.predict(X_test)

    rf_accuracy = accuracy_score(y_test, rf_predictions)
    xgb_accuracy = accuracy_score(y_test, xgb_predictions)

    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")

    _, ax = plt.subplots(1, 2, figsize=(12, 5))

    cm_rf = confusion_matrix(y_test, rf_predictions)
    ConfusionMatrixDisplay(cm_rf, display_labels=list(breast_cancer.target_names)).plot(ax=ax[0])
    ax[0].set_title('Random Forest Confusion Matrix')

    cm_xgb = confusion_matrix(y_test, xgb_predictions)
    ConfusionMatrixDisplay(cm_xgb, display_labels=list(breast_cancer.target_names)).plot(ax=ax[1])
    ax[1].set_title('XGBoost Confusion Matrix')

    plt.show()

    print("\nRandom Forest Classification Report:\n", classification_report(y_test, rf_predictions, target_names=list(breast_cancer.target_names)))
    print("-" * 60)
    print("\nXGBoost Classification Report:\n", classification_report(y_test, xgb_predictions, target_names=list(breast_cancer.target_names)))

test_dataset(X, y)

pd1 = pd.read_csv("./AI/Codes/Supervised_learning/5Ensemble_Learning/1.csv")
pd2 = pd.read_csv("./AI/Codes/Supervised_learning/5Ensemble_Learning/2.csv")
pd3 = pd.read_csv("./AI/Codes/Supervised_learning/5Ensemble_Learning/3.csv")

print(f"Shape of the first dataset: {pd1.shape}")
print(f"Shape of the second dataset: {pd2.shape}")
print(f"Shape of the third dataset: {pd3.shape}")

X1, y1 = pd1.drop(columns=['target']), pd1['target']
X2, y2 = pd2.drop(columns=['target']), pd2['target']
X3, y3 = pd3.drop(columns=['target']), pd3['target']

test_dataset(X1, y1)
test_dataset(X2, y2)
test_dataset(X3, y3)
