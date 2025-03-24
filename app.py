from flask import Flask, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)[0]
    return jsonify({'predicted_price': prediction})

if __name__ == '__main__':
    app.run(debug=True)
