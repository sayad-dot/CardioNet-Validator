# src/deployment/test_inference.py

import os
import pandas as pd
import joblib

def get_feature_names_from_processed(dataset_id=2):
    """
    Read one row of the processed CSV (e.g. clean_dataset2.csv) to extract
    the exact column names used during training (minus the "heart disease" column).
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    processed_path = os.path.join(
        project_root,
        "data",
        "processed",
        f"clean_dataset{dataset_id}.csv"
    )
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed CSV not found at {processed_path}")

    df = pd.read_csv(processed_path, nrows=1)  # only need one row to get columns
    if "heart disease" in df.columns:
        features = [col for col in df.columns if col != "heart disease"]
    elif "target" in df.columns:
        features = [col for col in df.columns if col != "target"]
    else:
        raise ValueError("Processed CSV does not contain 'heart disease' or 'target' column")
    return features

def load_model(model_name="xgboost", dataset_id=2):
    """
    Try to load the tuned model from either:
      - src/outputs/hyperparameter_tuning/models_dataset{dataset_id}/{model_name}_best.pkl
      - outputs/hyperparameter_tuning/models_dataset{dataset_id}/{model_name}_best.pkl
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # 1) First check under src/outputs/
    src_model_path = os.path.join(
        project_root,
        "src",
        "outputs",
        "hyperparameter_tuning",
        f"models_dataset{dataset_id}",
        f"{model_name}_best.pkl"
    )
    if os.path.exists(src_model_path):
        return joblib.load(src_model_path)

    # 2) Fallback: check under project-root outputs/
    root_model_path = os.path.join(
        project_root,
        "outputs",
        "hyperparameter_tuning",
        f"models_dataset{dataset_id}",
        f"{model_name}_best.pkl"
    )
    if os.path.exists(root_model_path):
        return joblib.load(root_model_path)

    raise FileNotFoundError(
        f"Model not found in either:\n  {src_model_path}\n  {root_model_path}"
    )

def preprocess_row(df_row: pd.DataFrame):
    """
    If you performed any additional preprocessing (scaling, encoding, etc.),
    apply it here. If not, simply return df_row unchanged.
    """
    return df_row

def predict_from_csv(csv_path: str, model_name="xgboost", dataset_id=2):
    # 1) Determine the correct feature names from the processed CSV
    feature_cols = get_feature_names_from_processed(dataset_id)

    # 2) Read new data
    df_new = pd.read_csv(csv_path)

    # 3) Verify that the new data has exactly those same feature columns
    missing = set(feature_cols) - set(df_new.columns)
    extra   = set(df_new.columns)   - set(feature_cols)
    if missing:
        raise ValueError(f"Input CSV is missing these required columns:\n  {sorted(missing)}")
    if extra:
        raise ValueError(f"Input CSV has unexpected extra columns:\n  {sorted(extra)}")

    df_ordered = df_new[feature_cols].copy()

    # 4) Preprocess (if needed)
    X_new = preprocess_row(df_ordered)

    # 5) Load model
    model = load_model(model_name=model_name, dataset_id=dataset_id)

    # 6) Predict probability and class
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_new)[:, 1]  # probability of class “1”
    else:
        # if the model only has decision_function (rare), scale to [0,1]
        scores = model.decision_function(X_new)
        proba = (scores - scores.min()) / (scores.max() - scores.min())

    preds = model.predict(X_new)

    # 7) Attach results and print
    df_out = df_ordered.copy()
    df_out["predicted_probability"] = proba
    df_out["predicted_class"] = preds

    print(df_out)

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage: python test_inference.py <path_to_csv> [model_name] [dataset_id]")
        print("  <path_to_csv>   – path to a CSV file whose headers match the processed CSV")
        print("  [model_name]    – one of 'logistic', 'random_forest', 'xgboost' (default: xgboost)")
        print("  [dataset_id]    – 1, 2 or 3 (default: 2)")
        sys.exit(1)

    csv_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) >= 3 else "xgboost"
    dataset_id = int(sys.argv[3]) if len(sys.argv) == 4 else 2

    predict_from_csv(csv_path, model_name=model_name, dataset_id=dataset_id)
