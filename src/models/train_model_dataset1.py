# File: src/models/train_model_dataset1.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# import your helpers explicitly
from src.models.model_utils import train_models, evaluate_models, save_models

def main():
    print("=== Verifying Original Dataset ===")
    orig_path = os.path.join(os.getcwd(), "data", "dataset1.csv")
    orig_df = pd.read_csv(orig_path)
    print("Original target values:", orig_df["heart disease"].unique())
    print("Class distribution in original data:")
    print(orig_df["heart disease"].value_counts())

    # Load your cleaned & unified dataset
    proc_path = os.path.join(os.getcwd(), "data", "processed", "clean_dataset1.csv")
    df = pd.read_csv(proc_path)
    print("\nLoaded processed dataset:", df.shape)

    # Map labels if they come in as {1,2} → {0,1}
    unique_labels = set(df["heart disease"].unique())
    if unique_labels == {1, 2}:
        df["heart disease"] = df["heart disease"].map({1: 0, 2: 1})
        print("\nLabels mapped 1→0, 2→1")
    else:
        print("\nLabels assumed already 0/1")

    print("Post-mapping class counts:")
    print(df["heart disease"].value_counts())

    # Split off features and target
    X = df.drop("heart disease", axis=1)
    y = df["heart disease"]

    # Stratify to ensure both classes in train & test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nAfter stratified split:")
    print(" y_train distribution:", "\n", y_train.value_counts())
    print(" y_test  distribution:", "\n", y_test.value_counts())

    # Train and evaluate
    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)

    for name, res in results.items():
        print(f"\n=== Model: {name} ===")
        print("Classification Report:")
        print(res["report"])
        print("Confusion Matrix:")
        print(res["confusion_matrix"])

    # Save your trained models
    save_dir = os.path.join("outputs", "models_dataset1")
    save_models(models, save_dir)
    print(f"\nModels saved under {save_dir}")

if __name__ == "__main__":
    main()
