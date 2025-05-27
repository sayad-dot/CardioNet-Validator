import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import os

def load_data(path):
    df = pd.read_csv(path)
    return df

def train_models(X_train, y_train):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained

def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        report = classification_report(y_test, preds, output_dict=True)
        cm = confusion_matrix(y_test, preds)
        results[name] = {
            "report": report,
            "confusion_matrix": cm
        }
    return results

def save_models(models, folder):
    os.makedirs(folder, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, os.path.join(folder, f"{name}.pkl"))
