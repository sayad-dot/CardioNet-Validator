# File: src/pipeline/train_final_model.py

import os
import joblib
from xgboost import XGBClassifier
import pandas as pd

from src.preprocessing.preprocess_utils import (
    load_dataset,
    handle_missing_values,
    encode_categorical,
    normalize_numerical
)

def main():
    # 1) Load raw Dataset3
    raw_path = os.path.join("data", "dataset3.csv")
    df = load_dataset(raw_path)
    print("Raw Dataset3 shape:", df.shape)

    # 2) Preprocess
    df = handle_missing_values(df)
    print(" After dropna:", df.shape)
    df = encode_categorical(df)
    df = normalize_numerical(df)
    print(" After encode+normalize:", df.shape)

    # 3) Map labels 1→0,2→1 if needed
    if set(df["heart disease"].unique()) == {1, 2}:
        df["heart disease"] = df["heart disease"].map({1: 0, 2: 1})

    # 4) Split features/target
    X = df.drop("heart disease", axis=1)
    y = df["heart disease"]

    # 5) Train final XGBoost
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
    model.fit(X, y)
    print("Model trained on Dataset3.")

    # 6) Save model
    out_dir = os.path.join("outputs", "final_model")
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "xgb_dataset3.pkl")
    joblib.dump(model, model_path)
    print(f"Saved final model to {model_path}")

if __name__ == "__main__":
    main()
