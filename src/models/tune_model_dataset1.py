# src/models/tune_model_dataset1.py

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
    This should mirror the loading logic in your train_model_dataset1 script.
    """
    df = pd.read_csv(path)
    # Assuming you already have “clean_dataset1.csv” saved in data/processed/
    # and that target was originally 1/2 and you need to map 2→1, 1→0.
    # Example:
    df['heart disease'] = df['heart disease'].map({1: 0, 2: 1})
    X = df.drop(columns=['heart disease'])
    y = df['heart disease']
    return X, y

def main():
    # 1) Paths
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    processed_path = os.path.join(PROCESSED_DATA_DIR, "clean_dataset1.csv")
    output_dir = os.path.join(os.path.dirname(__file__), os.pardir, "outputs", "hyperparameter_tuning", "models_dataset1")
    os.makedirs(output_dir, exist_ok=True)
    results_dir = os.path.join(os.path.dirname(__file__), os.pardir, "outputs", "hyperparameter_tuning")
    os.makedirs(results_dir, exist_ok=True)

    # 2) Load data
    X, y = load_processed_dataset(processed_path)

    # 3) Stratified train/test split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in splitter.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # 4) Define models + parameter grids
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

    # 5) For each estimator: run GridSearchCV, save best model + CSV of CV results
    summary_rows = []
    for name, (model, param_grid) in estimators.items():
        print(f"\n=== HYPERPARAM TUNING: {name.upper()} ===")

        grid = GridSearchCV(
            estimator = model,
            param_grid = param_grid,
            scoring = "roc_auc",
            cv = 5,
            n_jobs = -1,
            return_train_score = True,
        )
        grid.fit(X_train, y_train)

        print(f"Best params for {name}: {grid.best_params_}")
        best_model = grid.best_estimator_

        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        print(f"\n--- Classification report ({name}) on TEST set ---")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # 6) Save best model
        model_filename = os.path.join(output_dir, f"{name}_best.pkl")
        joblib.dump(best_model, model_filename)
        print(f"Saved best {name} to {model_filename}")

        # 7) Save CV results to a temporary DataFrame
        results_df = pd.DataFrame(grid.cv_results_)
        results_df["model"] = name
        summary_rows.append(results_df)

    # 8) Concatenate all CV results and write to CSV
    all_results = pd.concat(summary_rows, ignore_index=True)
    results_csv_path = os.path.join(results_dir, "search_results_dataset1.csv")
    all_results.to_csv(results_csv_path, index=False)
    print(f"\nAll search results for dataset1 saved to {results_csv_path}")

if __name__ == "__main__":
    main()
