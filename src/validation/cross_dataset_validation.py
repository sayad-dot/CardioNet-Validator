import pandas as pd
import os

from src.preprocessing.column_mapping import UNIFIED_COLUMNS
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from src.models.model_utils import train_models


def load_and_standardize_dataset(path, name):
    print(f"\n=== Loading {name} ===")
    df = pd.read_csv(path)
    print(f"Original shape: {df.shape}")

    df.columns = df.columns.str.strip()
    df = df.rename(columns=lambda x: UNIFIED_COLUMNS.get(x, x))
    target = 'heart disease'
    if target not in df.columns:
        raise ValueError(f"Missing target column '{target}' in {name}")

    vals = sorted(df[target].unique())
    if set(vals) == {1, 2}:
        df[target] = df[target].map({1: 0, 2: 1})
        print(f"Converted target from [1,2] to [0,1]")
    else:
        print(f"Target already in [0,1]")

    print(f"Final target values: {df[target].unique()}")
    X = df.drop(target, axis=1)
    y = df[target]
    return X, y


def standardize(X_train, *X_tests):
    scaler = StandardScaler().fit(X_train)
    X_train_s = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_tests_s = [pd.DataFrame(scaler.transform(X), columns=X_train.columns, index=X.index) for X in X_tests]
    return X_train_s, X_tests_s


def evaluate_model(model, X, y, mname, dname):
    print(f"\n--- {mname} on {dname} ---")
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_proba) if y_proba is not None else None

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC:   {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }


def cross_dataset_validation():
    datasets = {
        'Dataset1': 'data/processed/clean_dataset1.csv',
        'Dataset2': 'data/processed/clean_dataset2.csv',
        'Dataset3': 'data/processed/clean_dataset3.csv'
    }

    # Load and prepare all datasets
    data = {}
    for name, path in datasets.items():
        X, y = load_and_standardize_dataset(path, name)
        data[name] = {'X': X, 'y': y}

    # Restrict to common features
    common = sorted(set.intersection(*(set(v['X'].columns) for v in data.values())))
    for v in data.values():
        v['X'] = v['X'][common]

    results = []

    # For each training dataset
    for train_name in datasets:
        print(f"\n===== Training on {train_name} =====")
        X_train = data[train_name]['X']
        y_train = data[train_name]['y']

        # Identify test datasets
        test_names = [n for n in datasets if n != train_name]
        X_tests = [data[n]['X'] for n in test_names]
        y_tests = [data[n]['y'] for n in test_names]

        # Standardize features
        X_train_s, X_tests_s = standardize(X_train, *X_tests)

        # Train fresh models
        models = train_models(X_train_s, y_train)

        # Evaluate each model on each test set
        for model_name, model in models.items():
            for test_name, X_test_s, y_test in zip(test_names, X_tests_s, y_tests):
                metrics = evaluate_model(model, X_test_s, y_test, model_name, test_name)
                results.append({
                    'Train_Dataset': train_name,
                    'Model': model_name,
                    'Test_Dataset': test_name,
                    **metrics
                })

    # Summary
    summary_df = pd.DataFrame(results)
    print("\n📊 Cross-Dataset Validation Summary")
    print(summary_df)

    os.makedirs('outputs', exist_ok=True)
    summary_df.to_csv('outputs/cross_dataset_validation_summary.csv', index=False)
    print("Summary saved to outputs/cross_dataset_validation_summary.csv")


if __name__ == '__main__':
    cross_dataset_validation()
