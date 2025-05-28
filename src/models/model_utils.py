import pandas as pd
import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix


def load_data(path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    """
    return pd.read_csv(path)


def train_models(X_train, y_train):
    """
    Train a suite of classification models on the provided training data.
    Assumes y_train is already encoded as {0,1}.
    """
    # If labels are still {1,2}, map them once
    unique_labels = set(y_train.unique())
    if unique_labels == {1, 2}:
        y_train = y_train.replace({1: 0, 2: 1})

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    return models


def evaluate_models(models, X_test, y_test):
    """
    Generate classification reports and confusion matrices for each trained model.
    Assumes y_test is already encoded as {0,1}.
    """
    # If test labels are still {1,2}, map them once
    unique_labels = set(y_test.unique())
    if unique_labels == {1, 2}:
        y_test = y_test.replace({1: 0, 2: 1})

    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        results[name] = {
            "report": report,
            "confusion_matrix": cm
        }

    return results


def save_models(models: dict, folder: str):
    """
    Persist trained models to disk using joblib. Creates the folder if it doesn't exist.
    """
    os.makedirs(folder, exist_ok=True)
    for name, model in models.items():
        filename = os.path.join(folder, f"{name.replace(' ', '_').lower()}.pkl")
        joblib.dump(model, filename)
        print(f"Saved {name} to {filename}")
