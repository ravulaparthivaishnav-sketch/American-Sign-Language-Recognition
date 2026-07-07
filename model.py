import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ----------------------------
# Load Dataset
# ----------------------------
data = pd.read_csv("dataset.csv", header=None)

X = data.iloc[:, 1:]
y = data.iloc[:, 0]

# ----------------------------
# Encode Labels
# ----------------------------
encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("\nSamples per class:")
print(data.iloc[:, 0].value_counts().sort_index())

# ----------------------------
# Train/Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ----------------------------
# Train Model
# ----------------------------
model = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ----------------------------
# Predictions
# ----------------------------
pred = model.predict(X_test)

# ----------------------------
# Accuracy
# ----------------------------
accuracy = accuracy_score(y_test, pred)
print(f"\nAccuracy: {accuracy:.4f}")

# ----------------------------
# Classification Report
# ----------------------------
print("\nClassification Report:\n")
print(classification_report(
    y_test,
    pred,
    target_names=encoder.classes_
))

# ----------------------------
# Confusion Matrix
# ----------------------------
cm = confusion_matrix(y_test, pred)

print("\nConfusion Matrix:\n")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=encoder.classes_
)

fig, ax = plt.subplots(figsize=(12, 12))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=90)
plt.title("ASL Letter Confusion Matrix")
plt.tight_layout()
plt.show()

# ----------------------------
# Cross Validation
# ----------------------------
print("\nRunning 5-Fold Cross Validation...")

scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    n_jobs=-1
)

print("Fold Accuracies:", scores)
print("Average CV Accuracy:", scores.mean())

# ----------------------------
# Save Model
# ----------------------------
joblib.dump(model, "asl_model.pkl")
joblib.dump(encoder, "label_encoder.pkl")

print("\nModel and encoder saved successfully!")


