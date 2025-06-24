import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("dataset.csv")

# Selecting only the required features
selected_features = [
    "built_up_area_numeric_in_sq_ft",  
    "total_floors",  
    "bathrooms",  
    "Garden",  
    "furnishing"
]

# Convert "furnishing" into numeric values
furnishing_mapping = {"Unfurnished": 0, "Semi-Furnished": 1, "Furnished": 2}
df["furnishing"] = df["furnishing"].map(furnishing_mapping)

# Check if any missing values exist after mapping
if df["furnishing"].isnull().any():
    print("Error: Some furnishing values could not be mapped. Check your dataset!")
    exit()

# Define X (features) and y (target variable - house price)
X = df[selected_features]
y = df["price_numeric"]  # Make sure this is your target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, "house_price_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(selected_features, "feature_names.pkl")

print("✅ New model trained and saved with 5 features!")
