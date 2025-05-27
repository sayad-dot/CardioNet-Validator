from src.models.model_utils import *
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# Load dataset
df = load_data("data/processed/clean_dataset1.csv")
print("Dataset loaded:", df.shape)

# Label mapping: convert 1 → 0 and 2 → 1 for binary classification
df["heart disease"] = df["heart disease"].map({1: 0, 2: 1})
print("Unique target values after mapping:", df["heart disease"].unique())

# Features and label
X = df.drop("heart disease", axis=1)
y = df["heart disease"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
models = train_models(X_train, y_train)

# Evaluate
results = evaluate_models(models, X_test, y_test)

# Print results
for name, result in results.items():
    print(f"\nModel: {name}")
    print("Classification Report:")
    print(result["report"])
    print("Confusion Matrix:")
    print(result["confusion_matrix"])

# Save trained models
save_models(models, "outputs/models_dataset1/")
