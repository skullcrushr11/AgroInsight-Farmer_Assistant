import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os


# Load dataset
df = pd.read_csv("datasets/fertilizer prediction/fert_1/Crop and fertilizer dataset_altered.csv")

# Print the original shape
print(f"Original dataset shape: {df.shape}")

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=['number']).columns
categorical_cols = df.select_dtypes(exclude=['number']).columns

print(f"Numerical columns: {numerical_cols.tolist()}")
print(f"Categorical columns: {categorical_cols.tolist()}")

# Check for negative values in numerical columns
if len(numerical_cols) > 0:
    negative_counts = (df[numerical_cols] < 0).sum().sum()
    print(f"Number of negative values: {negative_counts}")
    
    # Remove rows with negative values in numerical columns only if there are any
    if negative_counts > 0:
        df = df[(df[numerical_cols] >= 0).all(axis=1)]
        print(f"Dataset shape after removing negative values: {df.shape}")

# Initialize label encoders dictionary
label_encoders = {}

# Encode categorical columns
for col in categorical_cols:
    if col != "Fertilizer":  # Don't encode target yet
        print(f"Encoding column: {col}")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Ensure the target column exists
if "Fertilizer" not in df.columns:
    raise ValueError("Fertilizer column not found in the dataset")

# Extract features and target
X = df.drop(columns=["Fertilizer"])
y_fertilizer = df["Fertilizer"]

print(f"X shape: {X.shape}")
print(f"y_fertilizer shape: {y_fertilizer.shape}")

# Create encoder for the target variable
fertilizer_le = LabelEncoder()
y_fertilizer_encoded = fertilizer_le.fit_transform(y_fertilizer)
label_encoders["Fertilizer"] = fertilizer_le

print(f"Unique fertilizer values: {y_fertilizer.unique().tolist()}")

# Add after encoding:
print("\nEncoding mappings:")
for col, encoder in label_encoders.items():
    print(f"\n{col}:")
    for i, label in enumerate(encoder.classes_):
        print(f"{label}: {i}")

# Train-test split
X_train, X_test, y_fertilizer_train, y_fertilizer_test = train_test_split(
    X, y_fertilizer_encoded, test_size=0.2, random_state=42
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Train Random Forest model
fertilizer_model = RandomForestClassifier(n_estimators=100, random_state=42)
fertilizer_model.fit(X_train, y_fertilizer_train)

# Save the model and label encoders
model_path = "datasets/fertilizer prediction/fert_1/fertilizer_model_1.pkl"
encoders_path = "datasets/fertilizer prediction/fert_1/label_encoders_1.pkl"

joblib.dump(fertilizer_model, model_path)
joblib.dump(label_encoders, encoders_path)

# Predict and evaluate
y_fertilizer_pred = fertilizer_model.predict(X_test)
fertilizer_accuracy = accuracy_score(y_fertilizer_test, y_fertilizer_pred)

print(f"Fertilizer Prediction Accuracy: {fertilizer_accuracy:.2f}")

# Verify saved files
loaded_model = joblib.load(model_path)
loaded_encoders = joblib.load(encoders_path)

print(f"Type of loaded_model: {type(loaded_model)}")
print(f"Type of loaded_encoders: {type(loaded_encoders)}")
print(f"Keys in loaded_encoders: {list(loaded_encoders.keys())}")

# Test with a sample input string
sample_input = "Soil_color=Red, Nitrogen=90, Phosphorous=42, Potassium=43, pH=6.5, Temperature=25, Crop=Rice"
print("\nTesting with sample input string:")
print(f"Input: {sample_input}")

# Convert input string to DataFrame
input_dict = dict(item.split("=") for item in sample_input.split(", "))
input_df = pd.DataFrame([input_dict])

# Encode categorical features
for col in categorical_cols:
    if col != "Fertilizer":
        input_df[col] = loaded_encoders[col].transform(input_df[col])

# Make prediction
pred_encoded = loaded_model.predict(input_df)[0]
pred_decoded = loaded_encoders["Fertilizer"].inverse_transform([pred_encoded])[0]
print(f"Predicted fertilizer: {pred_decoded}")