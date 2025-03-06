import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# Define paths to the saved outputs in the trainingOutputs folder
output_folder = "trainingOutputs"
model_path = os.path.join(output_folder, "nutrient_prediction_nn.h5")
scaler_path = os.path.join(output_folder, "scaler.pkl")
encoders_path = os.path.join(output_folder, "meal_label_encoders.pkl")

# 1) Load the trained model and preprocessing tools
loaded_model = load_model(model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
scaler = joblib.load(scaler_path)
label_encoders = joblib.load(encoders_path)

# 2) Load the original dataset for meal suggestions (ensure this dataset matches your training data)
df = pd.read_csv("dataset/dataset.csv")
# Convert Gender to numeric (Male -> 0, Female -> 1)
df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects a JSON body with:
    {
      "age": <number>,
      "gender": "Male" or "Female",
      "bmi": <number>,
      "bmr": <number>
    }
    
    Returns a JSON response with:
      "Suggested nutrients per day": {
            "Daily Calorie Target": <value>,
            "Protein (g)": <value>,
            "Carbohydrates (g)": <value>,
            "Fat (g)": <value>
      },
      "Suggested meals": [ <meal1>, <meal2>, <meal3>, ... ]
    """
    data = request.json

    # Extract user inputs
    age = data.get("age")
    gender_str = data.get("gender")
    bmi = data.get("bmi")
    bmr = data.get("bmr")

    # Validate inputs
    if any(v is None for v in [age, gender_str, bmi, bmr]):
        return jsonify({"error": "Missing required fields: age, gender, bmi, bmr"}), 400

    # Convert gender to numeric (Male -> 0, Female -> 1)
    gender = 0 if gender_str.lower() == "male" else 1

    # Prepare input data for model prediction
    input_data = np.array([[age, gender, bmi, bmr]])
    input_data_scaled = scaler.transform(input_data)

    # Predict numeric nutrition values using the loaded model
    numeric_prediction = loaded_model.predict(input_data_scaled)[0]
    cal_target = round(numeric_prediction[0])
    protein = round(numeric_prediction[1])
    carbs = round(numeric_prediction[2])
    fat = round(numeric_prediction[3])

    # Build a dictionary for the suggested nutrients per day
    suggested_nutrients = {
        "Daily Calorie Target": cal_target,
        "Protein (g)": protein,
        "Carbohydrates (g)": carbs,
        "Fat (g)": fat
    }

    # Find the closest matching record in the dataset for meal suggestions
    user_features = np.array([age, gender, bmi, bmr])
    diff = df[["Age", "Gender", "BMI", "BMR"]].apply(lambda row: np.abs(row.values - user_features).sum(), axis=1)
    closest_idx = diff.idxmin()

    # Extract meal suggestions from the closest matching record
    meal_suggestions = df.loc[closest_idx, ["Breakfast Suggestion", "Lunch Suggestion", "Dinner Suggestion", "Snack Suggestion"]].to_dict()
    meal_list = [
        meal_suggestions.get("Breakfast Suggestion"),
        meal_suggestions.get("Lunch Suggestion"),
        meal_suggestions.get("Dinner Suggestion"),
        meal_suggestions.get("Snack Suggestion")
    ]

    # Build the response
    response = {
        "Suggested nutrients per day": suggested_nutrients,
        "Suggested meals": meal_list
    }

    return jsonify(response), 200

if __name__ == "__main__":
    app.run(debug=True)
