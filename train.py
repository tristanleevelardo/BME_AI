import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib

# Create output folder if it doesn't exist
output_folder = "trainingOutputs"
os.makedirs(output_folder, exist_ok=True)

# 1) Load the dataset
df = pd.read_csv("dataset/dataset.csv")

# 2) Prepare input features X and target y
#    We assume we want to predict daily calorie target, protein, carbs, fat,
#    plus meal suggestions based on Age, Gender, BMI, BMR.
X = df[["Age", "Gender", "BMI", "BMR"]].copy()
y = df[[ 
    "Daily Calorie Target", "Protein", "Carbohydrates", "Fat",
    "Breakfast Suggestion", "Lunch Suggestion", "Dinner Suggestion", "Snack Suggestion"
]]

# 3) Convert 'Gender' to numeric for model input (Male -> 0, Female -> 1)
X["Gender"] = X["Gender"].map({"Male": 0, "Female": 1})

# 4) Scale input features (Age, BMI, BMR) for better training
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 5) Encode meal suggestions with LabelEncoder using .loc to avoid view warnings
label_encoders = {}
for col in ["Breakfast Suggestion", "Lunch Suggestion", "Dinner Suggestion", "Snack Suggestion"]:
    le = LabelEncoder()
    y.loc[:, col] = le.fit_transform(y[col])
    label_encoders[col] = le

# 6) Ensure the first 4 target columns are numeric by coercing errors
numeric_cols = ["Daily Calorie Target", "Protein", "Carbohydrates", "Fat"]
y.loc[:, numeric_cols] = y.loc[:, numeric_cols].apply(lambda col: pd.to_numeric(col, errors="coerce"))

# Drop any rows that have NaN values in the numeric columns (due to conversion issues)
nan_rows = y[numeric_cols].isna().any(axis=1)
if nan_rows.any():
    print("Dropping rows with non-numeric values in numeric columns:")
    print(y[numeric_cols][nan_rows])
    X_scaled = X_scaled[~nan_rows.values]
    y = y[~nan_rows.values]

# 7) Separate the numeric outputs (Calories, Protein, Carbs, Fat)
#    from the meal suggestion outputs
y_numeric = y.iloc[:, :4].to_numpy()
y_meals = y.iloc[:, 4:].to_numpy()

# Convert to proper numeric dtype (float32)
X_scaled = X_scaled.astype(np.float32)
y_numeric = y_numeric.astype(np.float32)

# 8) Split the data for training/testing (80/20)
X_train, X_test, y_train_numeric, y_test_numeric = train_test_split(
    X_scaled, y_numeric, test_size=0.2, random_state=42
)
y_train_meals, y_test_meals = train_test_split(
    y_meals, test_size=0.2, random_state=42
)

# 9) Build the neural network to predict the numeric values
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(64, activation="relu"),
    Dense(4, activation="linear")  # 4 outputs: Calories, Protein, Carbs, Fat
])

# 10) Compile the model
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# 11) Train the model
print("Training Neural Network...")
model.fit(X_train, y_train_numeric, epochs=50, batch_size=16, validation_data=(X_test, y_test_numeric))

# 12) Save the trained model and preprocessing tools in the output folder
model_path = os.path.join(output_folder, "nutrient_prediction_nn.h5")
scaler_path = os.path.join(output_folder, "scaler.pkl")
encoders_path = os.path.join(output_folder, "meal_label_encoders.pkl")

model.save(model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(label_encoders, encoders_path)

print(f"Training complete. Model and preprocessing files saved in '{output_folder}'")
