import pandas as pd
import pickle
import numpy as np
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Step 1: Load Dataset
df = pd.read_csv("C:\Users\Admin\Desktop\datascience\datset.csv")  # Replace with your dataset
X = df.drop("target", axis=1)  # Replace "target" with your actual target column
y = df["target"]

# Step 2: Preprocess Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Train Model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 4: Save Model
with open("model.pkl", "wb") as f:
    pickle.dump((model, scaler), f)

print("✅ Model trained and saved as 'model.pkl'")

# Step 5: Deploy Flask API
app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to the Codsoft ML API!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    data_scaled = scaler.transform([data])
    prediction = model.predict(data_scaled)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "_main_":
    app.run(debug=True)