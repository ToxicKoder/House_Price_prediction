import os
from flask import Flask, request, render_template
import numpy as np
import joblib
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load Model, Scaler, and Feature Names
model = joblib.load("house_price_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

print("Features required for input:", feature_names)

# Furnishing Mapping
furnishing_mapping = {
    "Unfurnished": 0, "unfurnished": 0,
    "Semi-Furnished": 1, "semi-furnished": 1, "Semi Furnished": 1,
    "Furnished": 2, "furnished": 2
}

@app.route("/")
def home():
    return render_template("new.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values
        data = request.form
        built_up_area = float(data["feature1"])
        num_floors = float(data["feature2"])
        num_bathrooms = float(data["feature3"])
        garden = float(data["feature4"])
        furnishing_str = data["feature5"].strip()

        print(f"Received Furnishing Type: {furnishing_str}")  # Debugging Output

        # Validate Furnishing Type
        if furnishing_str not in furnishing_mapping:
            return render_template(
                "new.html", 
                prediction_text=f"Error: Invalid Furnishing Type '{furnishing_str}'. Expected: {list(furnishing_mapping.keys())}"
            )

        furnishing = furnishing_mapping[furnishing_str]  # Convert to numeric

        # Prepare input array with 48 features
        input_features = np.zeros(len(feature_names))
        input_dict = {
            "built_up_area_numeric_in_sq_ft": built_up_area,
            "total_floors": num_floors,
            "bathrooms": num_bathrooms,
            "Garden": garden,
            "furnishing": furnishing
        }

        # Map selected inputs to full feature vector
        for i, feature in enumerate(feature_names):
            if feature in input_dict:
                input_features[i] = input_dict[feature]

        # Reshape and scale the input
        input_features = input_features.reshape(1, -1)
        input_scaled = scaler.transform(input_features)

        # Predict House Price (in USD)
        price_usd = model.predict(input_scaled)[0]

        # Convert to INR (1 USD = 83 INR)
        price_inr = price_usd * 83

        # Ensure static folder exists
        if not os.path.exists("static"):
            os.makedirs("static")

        # Plot the Prediction
        plt.figure(figsize=(5, 3))
        plt.bar(["Predicted Price (USD)"], [price_usd], color="blue", label="USD ($)")
        plt.bar(["Predicted Price (INR)"], [price_inr], color="green", label="INR (₹)")
        plt.ylabel("Price")
        plt.title("Predicted House Price")
        plt.legend()
        plot_path = "static/prediction_plot.png"
        plt.savefig(plot_path)
        plt.close()

        return render_template(
            "new.html", 
            prediction_text=f"Predicted House Price: ${round(price_usd, 2)} (₹{round(price_inr, 2)})",
            plot_url=plot_path
        )

    except Exception as e:
        return render_template("new.html", prediction_text="Error: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)
