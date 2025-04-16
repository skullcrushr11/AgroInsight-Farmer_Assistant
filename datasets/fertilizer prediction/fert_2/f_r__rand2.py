import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("fertilizer_recommendation_dataset.csv")

# Identify numerical columns and remove rows with negative values only for them
numerical_cols = df.select_dtypes(include=['number']).columns
df = df[(df[numerical_cols] >= 0).all(axis=1)]

# Encode categorical columns
categorical_cols = ["Soil", "Crop", "Fertilizer", "Remark"]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split data into features and targets
X = df.drop(columns=["Fertilizer", "Remark"], errors='ignore')
y_fertilizer = df["Fertilizer"]
y_remark = df["Remark"]

# Train-test split
X_train, X_test, y_fertilizer_train, y_fertilizer_test, y_remark_train, y_remark_test = train_test_split(
    X, y_fertilizer, y_remark, test_size=0.2, random_state=42
)

# Train Random Forest models
fertilizer_model = RandomForestClassifier(n_estimators=100, random_state=42)
remark_model = RandomForestClassifier(n_estimators=100, random_state=42)
fertilizer_model.fit(X_train, y_fertilizer_train)
remark_model.fit(X_train, y_remark_train)

# Save models
joblib.dump(fertilizer_model, "fertilizer_model_2.pkl")
joblib.dump(remark_model, "remark_model_2.pkl")
joblib.dump(label_encoders, "label_encoders1.pkl")

# Predict and evaluate
y_fertilizer_pred = fertilizer_model.predict(X_test)
y_remark_pred = remark_model.predict(X_test)

fertilizer_accuracy = accuracy_score(y_fertilizer_test, y_fertilizer_pred)
remark_accuracy = accuracy_score(y_remark_test, y_remark_pred)

print(f"Fertilizer Prediction Accuracy: {fertilizer_accuracy:.2f}")
print(f"Remark Prediction Accuracy: {remark_accuracy:.2f}")