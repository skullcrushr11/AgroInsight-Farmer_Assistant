import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset from the CSV file
df = pd.read_csv('crop_yield.csv')

# Filter for 'East' region only
df_east = df[df['Region'] == 'East'].copy()

# Drop the 'Region' column since it's constant ('East')
df_east = df_east.drop(columns=['Region'])

# Separate features and target
X = df_east.drop(columns=['Yield_tons_per_hectare'])  # Features
y = df_east['Yield_tons_per_hectare']  # Target

# Encode categorical columns ('Soil_Type', 'Crop', 'Weather_Condition')
label_encoders = {}
for column in ['Soil_Type', 'Crop', 'Weather_Condition']:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Convert boolean columns to integers (True=1, False=0)
X['Fertilizer_Used'] = X['Fertilizer_Used'].astype(int)
X['Irrigation_Used'] = X['Irrigation_Used'].astype(int)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=150, random_state=42,max_depth=25,min_samples_split=4)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})
print("\nFeature Importance:")
print(feature_importance.sort_values(by='Importance', ascending=False))

# Save the trained model and encoders
joblib.dump(rf_model, "random_forest_yield_model_1.pkl")
joblib.dump(label_encoders, "label_encoders_1.pkl")

print("\nModel and encoders saved successfully!")

# Example: Predict yield for a new sample from East region
new_sample = pd.DataFrame({
    'Soil_Type': ['Clay'],
    'Crop': ['Cotton'],
    'Rainfall_mm': [800.0],
    'Temperature_Celsius': [25.0],
    'Fertilizer_Used': [True],
    'Irrigation_Used': [True],
    'Weather_Condition': ['Sunny'],
    'Days_to_Harvest': [120]
})

# Encode categorical features in the new sample
for column in ['Soil_Type', 'Crop', 'Weather_Condition']:
    new_sample[column] = label_encoders[column].transform(new_sample[column])

# Convert boolean to integer
new_sample['Fertilizer_Used'] = new_sample['Fertilizer_Used'].astype(int)
new_sample['Irrigation_Used'] = new_sample['Irrigation_Used'].astype(int)

# Predict
predicted_yield = rf_model.predict(new_sample)
print(f"\nPredicted Yield (tons per hectare): {predicted_yield[0]:.2f}")
