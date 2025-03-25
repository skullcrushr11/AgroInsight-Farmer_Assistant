import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("Crop Yiled with Soil and Weather.csv")  # Replace with your actual file path

# Features and target
X = data[['Fertilizer', 'temp', 'N(ppm)', 'P(ppm)', 'K(ppm)']]
y = data['yeild']  # Typo in your data ('yeild' instead of 'yield')

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor
model = RandomForestRegressor(
    n_estimators=200,           # Number of trees
    max_depth=None,             # Allow full growth
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1                   # Use all CPU cores
)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate and print evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2 * 100:.2f}%")

# Save the model
import pickle
with open("yield_predictor_rf_model.pkl", "wb") as f:
    pickle.dump(model, f)

# --- Sample Predictions ---
sample_inputs = pd.DataFrame({
    'Fertilizer': [80, 60, 75],
    'temp': [28, 38, 26],
    'N(ppm)': [80, 70, 75],
    'P(ppm)': [24, 20, 22],
    'K(ppm)': [20, 18, 19]
})

# Predict yield for sample inputs
sample_preds = model.predict(sample_inputs)

# Print sample predictions
print("\nSample Predictions:")
for i, (input_vals, pred) in enumerate(zip(sample_inputs.values, sample_preds)):
    print(f"Sample {i+1}: {input_vals} -> Predicted Yield: {pred:.2f}")