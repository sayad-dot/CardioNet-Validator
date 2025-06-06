# src/models/tune_model_dataset3.py

import os
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

def load_processed_dataset(path: str):
    """
    Load the processed dataset3 CSV (already 0/1 target).
    """
    df = pd.read_csv(path)

    # Rename if target column is still "target" instead of "heart disease"
    if "target" in df.columns and "heart disease" not in df.columns:
        df = df.rename(columns={"target": "heart disease"})

    X = df.drop(columns=["heart disease"])
    y = df["heart disease"]
    return X, y

def main():
    # ─── Locate project root and processed folder ───
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

    # ─── Correct filename: clean_dataset3.csv (no extra underscore) ───
    processed_path = os.path.join(PROCESSED_DATA_DIR, "clean_dataset3.csv")

    # ─── Output folders ───
    output_dir = os.path.join(os.path.dirname(__file__), os.pardir, "outputs", "hyperparameter_tuning", "models_dataset3")
    os.makedirs(output_dir, exist_ok=True)

    results_dir = os.path.join(os.path.dirname(__file__), os.pardir, "outputs", "hyperparameter_tuning")
    os.makedirs(results_dir, exist_ok=True)

    # ─── Load data ───
    X, y = load_processed_dataset(processed_path)

    # ─── Stratified train/test split ───
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in splitter.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # ─── Define models + parameter grids ───
    estimators = {
        "logistic": (
            LogisticRegression(solver="liblinear", random_state=42),
            {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l2"],
            }
        ),
        "random_forest": (
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": [100, 200],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5],
            }
        ),
        "xgboost": (
            XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
            {
                "n_estimators": [50, 100],
                "max_depth": [3, 6],
                "learning_rate": [0.01, 0.1],
            }
        )
    }

    summary_rows = []
    for name, (model, param_grid) in estimators.items():
        print(f"\n=== HYPERPARAM TUNING: {name.upper()} (Dataset 3) ===")

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=5,
            n_jobs=-1,
            return_train_score=True,
        )
        grid.fit(X_train, y_train)

        print(f"Best params for {name}: {grid.best_params_}")
        best_model = grid.best_estimator_

        # ─── Evaluate on test set ───
        y_pred = best_model.predict(X_test)
        print(f"\n--- Classification report ({name}) on TEST set ---")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # ─── Save best model ───
        model_filename = os.path.join(output_dir, f"{name}_best.pkl")
        joblib.dump(best_model, model_filename)
        print(f"Saved best {name} to {model_filename}")

        # ─── Collect CV results ───
        results_df = pd.DataFrame(grid.cv_results_)
        results_df["model"] = name
        summary_rows.append(results_df)

    # ─── Write all CV results to CSV ───
    all_results = pd.concat(summary_rows, ignore_index=True)
    results_csv_path = os.path.join(results_dir, "search_results_dataset3.csv")
    all_results.to_csv(results_csv_path, index=False)
    print(f"\nAll search results for dataset3 saved to {results_csv_path}")

if __name__ == "__main__":
    main()
