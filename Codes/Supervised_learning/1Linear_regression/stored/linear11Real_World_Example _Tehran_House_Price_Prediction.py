import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


file_path = '/home/alireza/Documents/STUDY/AI/DataSets/housePrice.xlsx'
df = pd.read_excel(file_path)

#Show and describe datas
'''print(df.head())
print(df['Area'].describe())
print(df[df['Area'] > 1e6])
print(len(df))'''

#Show datas before removing outlines
'''# Generate indices for the x-axis
indices = np.arange(len(df['Area']))
# Scatter plot
plt.scatter(indices, df['Area'])
# Labels and title
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Scatter Plot of 1×n Matrix")
# Show the plot
plt.show()'''

# Data cleaning - removing outliers based on IQR
Q1_area = df['Area'].quantile(0.25)             #25% of the data points in the 'Area' column are below this value.
Q3_area = df['Area'].quantile(0.75)             #75% of the data points are below this value.
IQR_area = Q3_area - Q1_area

Q1_price = df['PriceUSD'].quantile(0.25)        #25% of the data points in the 'PriceUSD' column are below this value.
Q3_price = df['PriceUSD'].quantile(0.75)        #75% of the data points are below this value.
IQR_price = Q3_price - Q1_price

#📌Why 1.5 × IQR?
# The 1.5 × IQR rule comes from the assumption that a normal distribution (bell curve) contains most data within a certain range.

# In a normal distribution:

# About 99.3% of data falls within ±2.7 standard deviations from the mean.
# Outliers are expected to be rare and fall beyond this range.
# 🔹 IQR captures the middle 50% of the data (between Q1 and Q3).
# 🔹 Multiplying by 1.5 expands this range beyond the typical spread, catching extreme values while avoiding false positives.

# 📊 Example of Why It Works
# Consider a normally distributed dataset:

# Quartile	Value
# Q1 (25th percentile)	                    10
# Q3 (75th percentile)	                    30
# IQR (Q3 - Q1)	                            30 - 10 = 20
# Lower Bound	                                10 - (1.5 × 20) = -20
# Upper Bound	                                30 + (1.5 × 20) = 60

# If a value is below -20 or above 60, it is considered an outlier.

# 📌Is This a Strict Law?
# No, it’s just a common rule of thumb. You can adjust the 1.5 factor if needed:

# Use 3 × IQR for extreme outliers.
# Use 1 × IQR for more sensitive detection.
# Some fields use alternative outlier detection methods, such as:

# Z-Score Method (for normal distributions)
# Modified Z-Score
# Machine Learning (e.g., Isolation Forests, DBSCAN)


lower_bound_area = Q1_area - 1.5 * IQR_area
upper_bound_area = Q3_area + 1.5 * IQR_area

lower_bound_price = Q1_price - 1.5 * IQR_price
upper_bound_price = Q3_price + 1.5 * IQR_price

df_cleaned = df[(df['Area'] >= lower_bound_area) & (df['Area'] <= upper_bound_area) &
                (df['PriceUSD'] >= lower_bound_price) & (df['PriceUSD'] <= upper_bound_price)]


#Show datas after removing outlines
'''# Generate indices for the x-axis
indices = np.arange(len(df_cleaned['Area']))
# Scatter plot
plt.scatter(indices, df_cleaned['Area'])
# Labels and title
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Scatter Plot of 1×n Matrix")
# Show the plot
plt.show()'''

# Check the cleaned dataset
'''print(df_cleaned.describe())'''

#another Show fo datas after removing outlines
'''plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df_cleaned['Area'], bins=20, color='blue', alpha=0.7)
plt.title('Distribution of Area')

plt.subplot(1, 2, 2)
plt.hist(df_cleaned['PriceUSD'], bins=20, color='green', alpha=0.7)
plt.title('Distribution of PriceUSD')

plt.show()'''


# Function to compute the Root Mean Squared Error (RMSE)
def compute_rms_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Function to create polynomial features
def polynomial_features(X, degree):
    poly = PolynomialFeatures(degree=degree)
    return poly.fit_transform(X)

# Function to perform polynomial regression
def polynomial_regression(X, y, degree):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    return model

# Group by 'Address' and perform analysis for each region
#Returns an array of unique values (removes duplicates).
addresses = df_cleaned['Address'].unique()

for address in addresses:
    df_address = df_cleaned[df_cleaned['Address'] == address]

    # Skip if not enough data points
    if len(df_address) < 2:
        print(f"Skipping address {address} due to insufficient samples.")
        continue

    X = df_address[['Area']]
    y = df_address['PriceUSD']

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Skip if train or test sets are empty
    if len(X_train) < 50 or len(X_test) < 25:
        # print(f"Skipping address {address} due to train-test split issues.")
        continue
    else:
      print(f"Processing address: {address}")

    # Set polynomial degrees to evaluate
    degrees = [2, 3, 5, 8, 10]

    train_rms_errors = []
    test_rms_errors = []

    # Visualize Polynomial Regression for each degree
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 12))  # 2 rows, 3 columns

    for idx, degree in enumerate(degrees):
        model = polynomial_regression(X_train, y_train, degree)

        X_train_poly = polynomial_features(X_train, degree)
        y_train_pred = model.predict(X_train_poly)

        X_test_poly = polynomial_features(X_test, degree)
        y_test_pred = model.predict(X_test_poly)
        
        train_rms_error = compute_rms_error(y_train, y_train_pred)
        test_rms_error = compute_rms_error(y_test, y_test_pred)

        train_rms_errors.append(train_rms_error)
        test_rms_errors.append(test_rms_error)
        # print(f"Address: {address}, Degree {degree}: Train RMSE = {train_rms_error:.2f}, Test RMSE = {test_rms_error:.2f}")

        # Scatter plot of actual data and polynomial fit
        # Create a 2x3 grid of subplots


        # Flatten axs for easy indexing
        axs = axs.flatten()


        axs[idx].scatter(X_train, y_train, color='blue', label="Training Data")
        axs[idx].scatter(X_test, y_test, color='red', label="Test Data", alpha=0.6)
        
        X_fit = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
        X_fit_poly = polynomial_features(X_fit, degree)
        y_fit_pred = model.predict(X_fit_poly)
        
        axs[idx].plot(X_fit, y_fit_pred, label=f"Degree {degree} Fit", color='green')
        axs[idx].set_title(f"{address} - Degree {degree}")
        axs[idx].set_xlabel("Area")
        axs[idx].set_ylabel("PriceUSD")
        axs[idx].legend()
        axs[idx].grid(True)

    # Use the 6th subplot for RMSE vs Polynomial Degree
    axs[5].plot(degrees, train_rms_errors, marker='o', label='Train RMSE', color='blue')
    axs[5].plot(degrees, test_rms_errors, marker='o', label='Test RMSE', color='red')
    axs[5].set_title(f"RMSE vs Degree of Polynomial for {address}")
    axs[5].set_xlabel("Polynomial Degree")
    axs[5].set_ylabel("RMSE")
    axs[5].legend()
    axs[5].grid(True)

    # Adjust layout for better readability
    plt.tight_layout()
    plt.show()