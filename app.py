from flask import Flask, request, jsonify
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model and encoder
with open("house_price_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("encoder.pkl", "rb") as encoder_file:
    encoder = pickle.load(encoder_file)

# Define feature order for encoding
categorical_features = ["city", "furnishing"]
numerical_features = ["area", "bhk", "bathrooms", "parking", "age", "near_metro", "near_mall", "near_hospital", "near_school"]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract features from request
    city = data["city"]
    furnishing = data["furnishing"]
    numerical_values = [data["area"], data["bhk"], data["bathrooms"], data["parking"], data["age"],
                        data["near_metro"], data["near_mall"], data["near_hospital"], data["near_school"]]

    # Encode categorical features
    input_df = pd.DataFrame([[city, furnishing]], columns=categorical_features)
    encoded_cats = encoder.transform(input_df)
    
    # Combine encoded categorical and numerical features
    final_features = np.hstack((encoded_cats, numerical_values)).reshape(1, -1)

    # Predict price
    prediction = model.predict(final_features)[0]

    return jsonify({'predicted_price': f"₹{prediction:,.2f}"})

if __name__ == '__main__':
    app.run(debug=True)

