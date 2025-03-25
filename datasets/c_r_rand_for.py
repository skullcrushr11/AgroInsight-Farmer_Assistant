import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("Crop_recommendation.csv")  # Replace with your actual file path

# Features and target
X = data[['N(ppm)', 'P(ppm)', 'K(ppm)', 'temperature', 'humidity(relative humidity in %)', 'ph', 'rainfall(in mm)']]
y = data['label']

# Encode target labels (e.g., "rice" -> 0, "maize" -> 1, etc.)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier(
    n_estimators=200,           # Number of trees (increased from default 100)
    max_depth=None,             # Allow trees to grow fully (or tune this)
    min_samples_split=2,        # Minimum samples to split a node
    min_samples_leaf=1,         # Minimum samples per leaf
    random_state=42,            # For reproducibility
    n_jobs=-1                   # Use all CPU cores for faster training
)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the model for later use
import pickle
with open("crop_recommender_rf_model.pkl", "wb") as f:
    pickle.dump(model, f)

# --- Sample Predictions ---
# Create sample inputs (based on your dataset's feature ranges)
sample_inputs = pd.DataFrame({
    'N(ppm)': [90, 60, 40],              # Nitrogen (ppm)
    'P(ppm)': [42, 55, 72],              # Phosphorus (ppm)
    'K(ppm)': [43, 44, 77],              # Potassium (ppm)
    'temperature': [20.88, 23.00, 17.02],  # Temperature (°C)
    'humidity(relative humidity in %)': [82.00, 82.32, 16.99],     # Relative humidity (%)
    'ph': [6.50, 7.84, 7.49],              # pH
    'rainfall(in mm)': [202.94, 263.96, 88.55]    # Rainfall (mm)
})

# Predict crop for sample inputs
sample_preds = model.predict(sample_inputs)
sample_preds_labels = le.inverse_transform(sample_preds)  # Convert back to crop names

# Print sample predictions
print("\nSample Predictions:")
for i, (input_vals, pred) in enumerate(zip(sample_inputs.values, sample_preds_labels)):
    print(f"Sample {i+1}: {input_vals} -> Predicted Crop: {pred}")

# Save the LabelEncoder for later use
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)