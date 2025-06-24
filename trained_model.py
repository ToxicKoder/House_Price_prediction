import numpy as np
import pandas as pd
import joblib  # For saving model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("dataset.csv")  # Make sure the dataset file is in the same directory

# Handling missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# Identify categorical columns and encode them
categorical_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Splitting features (X) and target (y)
X = df.drop(columns=["price_numeric"])  # Ensure 'price_numeric' is the actual target column
y = df["price_numeric"]

# Splitting into train-test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save Model, Scaler, and Feature Names
joblib.dump(model, "house_price_model.pkl")  # Model file
joblib.dump(scaler, "scaler.pkl")  # Scaler file
joblib.dump(list(X.columns), "feature_names.pkl")  # Feature names file

print("✅ Model, Scaler, and Feature Names Saved Successfully!")
