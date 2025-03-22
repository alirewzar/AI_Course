import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

housing = fetch_california_housing()

#just for showing data if you nead
'''
# Fetch the dataset
data = fetch_california_housing()
# Convert it to a DataFrame for easy viewing
df = pd.DataFrame(data.data, columns=data.feature_names)
print(df.head(5))
'''

X, y = housing.data, housing.target
#print(pd.DataFrame(y).head(5))
#print(pd.DataFrame(X).head(5))
#print(len(y))

feature_names = housing.feature_names
#print("Feature names:", feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
'''
print(pd.DataFrame(X_train).head(5))
print(pd.DataFrame(X_test).head(5))
print(pd.DataFrame(y_train).head(5))
print(pd.DataFrame(y_test).head(5))
'''

#This code shows you how the standard scaler works for each parameter
#it scales the features in your dataset to have the following properties:
#Mean = 0: The average value of the feature will be centered at 0.
#Standard Deviation = 1: The spread or dispersion of the data is scaled to 1.
#X_scaled = (X − μ) / σ
#Where:
#X is the original feature value.
#μ is the mean of the feature.
#σ is the standard deviation of the feature.
'''
hand_on = []
for i in range(len(y_train)):
    hand_on.append(X_train[i][0])

mean_hand_on = np.mean(hand_on)
sd_hand_on = np.std(hand_on)

X_train_new = (hand_on - mean_hand_on) / sd_hand_on
print(X_train_new[0:10])
'''
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#print(pd.DataFrame(X_train_scaled).head(5))

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse:.2f}")

coefficients = model.coef_
for name, coef in zip(feature_names, coefficients):
    print(f"{name}: {coef:.3f}")

#print(len(y_pred))

plt.figure(figsize=(18, 10))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

print(y_test.min(), y_test.max())
