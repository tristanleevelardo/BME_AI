import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv("dataset/dataset.csv")

# Prepare features and numeric target values (we ignore meal suggestions for validation)
X = df[["Age", "Gender", "BMI", "BMR"]].copy()
y = df[["Daily Calorie Target", "Protein", "Carbohydrates", "Fat"]].copy()

# Convert Gender to numeric (Male -> 0, Female -> 1)
X["Gender"] = X["Gender"].map({"Male": 0, "Female": 1})

# Load the scaler and model from the trainingOutputs folder
output_folder = "trainingOutputs"
model_path = os.path.join(output_folder, "nutrient_prediction_nn.h5")
scaler_path = os.path.join(output_folder, "scaler.pkl")

# Load scaler and model (use custom_objects for 'mse')
scaler = joblib.load(scaler_path)
model = load_model(model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

# Scale the input features
X_scaled = scaler.transform(X)

# Split into training and testing sets (we use the test set for validation)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Ensure target is numeric
y_test = y_test.to_numpy().astype(np.float32)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate regression metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Validation Metrics on Test Set:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# -----------------------------------------------------
# Sensitivity Analysis: Evaluate robustness by perturbing inputs
# Here, we perturb the Age feature (+/- 5%) for the first test sample

# Assume the features are ordered as: Age, Gender, BMI, BMR.
# We take the first test sample:
example = X_test[0].copy()  
original_pred = model.predict(np.array([example]))[0]

# Perturb Age by +5%
example_plus = example.copy()
example_plus[0] = example_plus[0] * 1.05
pred_plus = model.predict(np.array([example_plus]))[0]

# Perturb Age by -5%
example_minus = example.copy()
example_minus[0] = example_minus[0] * 0.95
pred_minus = model.predict(np.array([example_minus]))[0]

print("\nSensitivity Analysis on first test sample (perturbing Age by +/- 5%):")
print("Original prediction:", original_pred)
print("Prediction with Age +5%:", pred_plus)
print("Prediction with Age -5%:", pred_minus)
