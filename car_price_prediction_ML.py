# car price prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

try:
    df = pd.read_csv('car data.csv')
    print("Dataset loaded successfully.")
    print("Initial 5 rows:")
    print(df.head())
    print("\nDataset Info:")
    df.info()
except FileNotFoundError:
    print("Error: 'car data.csv' not found. Please ensure the file is in the correct directory.")
    exit()

print("\n--- Data Preprocessing and Feature Engineering ---")

df['Age'] = 2020 - df['Year']
print(f"Added 'Age' column based on 'Year'. Current Year assumed: 2020.")

df.drop(['Year', 'Car_Name'], axis=1, inplace=True)
print("Dropped 'Year' and 'Car_Name' columns.")

print("Applying One-Hot Encoding to categorical features...")
df = pd.get_dummies(df, columns=['Fuel_Type', 'Selling_type', 'Transmission'], drop_first=True)
print("Categorical features encoded.")
print("\nFirst 5 rows after preprocessing:")
print(df.head())

X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']
print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

print("\n--- Splitting Data ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

print("\n--- Training Linear Regression Model ---")
model = LinearRegression()
model.fit(X_train, y_train)
print("Linear Regression model trained successfully.")

y_pred = model.predict(X_test)
print("Predictions made on the test set.")

print("\n--- Model Evaluation ---")
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"R-squared (R2): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

print("\n--- Generating Visualizations ---")

predictions_df = pd.DataFrame({'Actual_Price': y_test, 'Predicted_Price': y_pred})

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Actual_Price', y='Predicted_Price', data=predictions_df)
plt.title('Actual vs. Predicted Car Prices')
plt.xlabel('Actual Price (in Lakhs)')
plt.ylabel('Predicted Price (in Lakhs)')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.grid(True)
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals (Actual Price - Predicted Price)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

print("\nPython script for Car Price Prediction project completed.")