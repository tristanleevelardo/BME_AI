from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load dataset
dataset_path = "BMIDataset.csv"
df = pd.read_csv(dataset_path)

# Select relevant columns for model training
df_selected = df[["BMI", "Age", "Physical_Activity_Level", "Allergens", "Diet_Recommendation"]].copy()

# Encode the target variable (Diet_Recommendation) into numerical values
encoder_diet = LabelEncoder()
df_selected["Diet_Recommendation"] = encoder_diet.fit_transform(df_selected["Diet_Recommendation"])

# Split data into features (X) and target (y)
X = df_selected.drop(columns=["Diet_Recommendation"])
y = df_selected["Diet_Recommendation"]

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

@app.route("/predict", methods=["POST"])
def predict():
    """
    API endpoint to predict diet recommendation based on user input.
    Expected JSON input:
    {
        "BMI": float,
        "Age": int,
        "Physical_Activity_Level": int (1-5),
        "Allergens": int (0-5)
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()

        # Extract user inputs
        bmi = data["BMI"]
        age = data["Age"]
        activity_level = data["Physical_Activity_Level"]
        allergens = data["Allergens"]

        # Prepare input data for model prediction
        user_input = [[bmi, age, activity_level, allergens]]
        prediction = model.predict(user_input)

        # Convert predicted numerical label back to diet category
        recommended_diet = encoder_diet.inverse_transform(prediction)[0]

        return jsonify({"Recommended Diet": recommended_diet})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
