import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load model
print("Loading model...")
package = joblib.load("models/sign_language_random_forest.pkl")

model = package["model"]
features = package["features"]

print("✓ Model loaded")

# Load test data
print("\nLoading test data...")
df = pd.read_csv("dataset/test.csv")

X_test = df[features]
y_test = df["label"]

# Predict
print("\nMaking predictions...")
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print("MODEL CHECK")
print("=" * 50)

print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification report
print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# Show some actual vs predicted
print("\n" + "=" * 50)
print("SAMPLE PREDICTIONS")
print("=" * 50)

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print(results.head(30).to_string(index=False))