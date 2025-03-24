import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# Load Data
df = pd.read_csv("house_data.csv")

# Define Features and Target
categorical_features = ["city", "furnishing"]
numerical_features = ["area", "bhk", "bathrooms", "parking", "age", "near_metro", "near_mall", "near_hospital", "near_school"]

X = df[categorical_features + numerical_features]
y = df["price"]

# One-Hot Encode Categorical Features
encoder = OneHotEncoder(sparse=False, drop="first")
encoded_cats = encoder.fit_transform(df[categorical_features])
encoded_cat_columns = encoder.get_feature_names_out(categorical_features)

# Create final DataFrame
X_encoded = pd.DataFrame(encoded_cats, columns=encoded_cat_columns)
X_final = pd.concat([X_encoded, df[numerical_features]], axis=1)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📊 Model Performance: MAE = {mae:.2f}, R² = {r2:.2f}")

# Save Model and Encoder
with open("house_price_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("encoder.pkl", "wb") as encoder_file:
    pickle.dump(encoder, encoder_file)

print("✅ Model and encoder saved successfully!")
