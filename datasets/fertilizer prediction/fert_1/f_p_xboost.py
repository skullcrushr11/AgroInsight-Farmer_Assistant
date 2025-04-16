import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

# Load dataset
df = pd.read_csv("Crop and fertilizer dataset.csv")

# Identify numerical and categorical columns
numerical_cols = df.select_dtypes(include=['number']).columns
categorical_cols = df.select_dtypes(exclude=['number']).columns

# Remove rows with negative values in numerical columns
df = df[(df[numerical_cols] >= 0).all(axis=1)]

# Encode categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split data into features and targets
X = df.drop(columns=["Fertilizer", "Link"], errors='ignore')
y_fertilizer = df["Fertilizer"].astype(str)  # Convert to string before encoding
y_link = df["Link"].astype(str)

# Encode target variables
fertilizer_encoder = LabelEncoder()
y_fertilizer = fertilizer_encoder.fit_transform(y_fertilizer)
link_encoder = LabelEncoder()
y_link = link_encoder.fit_transform(y_link)

# Train-test split
X_train, X_test, y_fertilizer_train, y_fertilizer_test, y_link_train, y_link_test = train_test_split(
    X, y_fertilizer, y_link, test_size=0.2, random_state=42, stratify=y_fertilizer
)

# Train XGBoost models
fertilizer_model = xgb.XGBClassifier(objective="multi:softmax", num_class=len(np.unique(y_fertilizer)), eval_metric='mlogloss')
link_model = xgb.XGBClassifier(objective="multi:softmax", num_class=len(np.unique(y_link)), eval_metric='mlogloss')

# Hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

fertilizer_search = GridSearchCV(fertilizer_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
link_search = GridSearchCV(link_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)

fertilizer_search.fit(X_train, y_fertilizer_train)
link_search.fit(X_train, y_link_train)

# Get best models
best_fertilizer_model = fertilizer_search.best_estimator_
best_link_model = link_search.best_estimator_

# Save models
joblib.dump(best_fertilizer_model, "fertilizer_model_xg_1.pkl")
joblib.dump(best_link_model, "link_model_xg_1.pkl")
joblib.dump(label_encoders, "label_encoders_xg_1.pkl")
joblib.dump(fertilizer_encoder, "fertilizer_encoder_xg_1.pkl")
joblib.dump(link_encoder, "link_encoder_xg_1.pkl")

# Predict and evaluate
y_fertilizer_pred = best_fertilizer_model.predict(X_test)
y_link_pred = best_link_model.predict(X_test)

fertilizer_accuracy = accuracy_score(y_fertilizer_test, y_fertilizer_pred)
link_accuracy = accuracy_score(y_link_test, y_link_pred)

print(f"Fertilizer Prediction Accuracy: {fertilizer_accuracy:.2f}")
print(f"Link Prediction Accuracy: {link_accuracy:.2f}")
