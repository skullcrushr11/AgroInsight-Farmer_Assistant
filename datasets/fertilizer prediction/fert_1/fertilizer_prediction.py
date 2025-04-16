import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Create directory if it doesn't exist
os.makedirs("datasets/fertilizer prediction/fert_1", exist_ok=True)

def train_fertilizer_model():
    try:
        # Load the dataset
        print("Loading dataset...")
        df = pd.read_csv("datasets/fertilizer prediction/fert_1/Crop and fertilizer dataset.csv")
        
        # Print dataset info
        print("\nDataset Info:")
        print(f"Shape: {df.shape}")
        print("\nColumns:", df.columns.tolist())
        print("\nSample data:")
        print(df.head())
        
        # Identify numerical and categorical columns
        numerical_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(exclude=['number']).columns
        
        print("\nNumerical columns:", numerical_cols.tolist())
        print("Categorical columns:", categorical_cols.tolist())
        
        # Initialize label encoders dictionary
        label_encoders = {}
        
        # Encode categorical columns
        for col in categorical_cols:
            if col != "Fertilizer":  # Don't encode target yet
                print(f"\nEncoding column: {col}")
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le
                print(f"Unique values in {col}:", le.classes_)
        
        # Create encoder for the target variable
        fertilizer_le = LabelEncoder()
        df["Fertilizer_encoded"] = fertilizer_le.fit_transform(df["Fertilizer"])
        label_encoders["Fertilizer"] = fertilizer_le
        
        print("\nFertilizer encoding:")
        for i, label in enumerate(fertilizer_le.classes_):
            print(f"{label}: {i}")
        
        # Prepare features and target
        feature_cols = ["Soil_color", "Nitrogen", "Phosphorus", "Potassium", "pH", "Temperature", "Crop"]
        X = df[feature_cols]
        y = df["Fertilizer_encoded"]
        
        # Split the data
        print("\nSplitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        # Train Random Forest model
        print("\nTraining Random Forest model...")
        rf_model = RandomForestClassifier(
            n_estimators=100
        )
        rf_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel Accuracy: {accuracy:.2f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=fertilizer_le.classes_))
        
        # Save the model and label encoders
        print("\nSaving model and label encoders...")
        model_path = "datasets/fertilizer prediction/fert_1/fertilizer_model_1.pkl"
        encoders_path = "datasets/fertilizer prediction/fert_1/label_encoders_1.pkl"
        
        joblib.dump(rf_model, model_path)
        joblib.dump(label_encoders, encoders_path)
        
        print(f"Model saved to: {model_path}")
        print(f"Label encoders saved to: {encoders_path}")
        
        # Test the saved model
        print("\nTesting saved model...")
        loaded_model = joblib.load(model_path)
        loaded_encoders = joblib.load(encoders_path)
        
        # Test with a sample input
        sample_input = {
            "Soil_color": "Red",
            "Nitrogen": 90,
            "Phosphorus": 42,
            "Potassium": 43,
            "pH": 6.5,
            "Temperature": 25,
            "Crop": "Rice"
        }
        
        # Convert sample input to DataFrame
        sample_df = pd.DataFrame([sample_input])
        
        # Encode categorical features
        for col in ["Soil_color", "Crop"]:
            sample_df[col] = loaded_encoders[col].transform(sample_df[col])
        
        # Make prediction
        pred_encoded = loaded_model.predict(sample_df)[0]
        pred_decoded = loaded_encoders["Fertilizer"].inverse_transform([pred_encoded])[0]
        
        print("\nSample prediction test:")
        print(f"Input: {sample_input}")
        print(f"Predicted fertilizer: {pred_decoded}")
        
        return True
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting fertilizer prediction model training...")
    success = train_fertilizer_model()
    if success:
        print("\nModel training completed successfully!")
    else:
        print("\nModel training failed. Please check the error messages above.") 