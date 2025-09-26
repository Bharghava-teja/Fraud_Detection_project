import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("Dataset/fraud_dataset.csv")

# Drop high-cardinality ID columns
df = df.drop(columns=["nameOrig", "nameDest"])

# Encode categorical 'type'
le = LabelEncoder()
df["type"] = le.fit_transform(df["type"])

# Split data
X = df.drop(columns=["isFraud"])
y = df["isFraud"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
import joblib
joblib.dump(model, "fraud_model.pkl")
