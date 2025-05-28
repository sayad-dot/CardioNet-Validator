import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.model_utils import train_models, evaluate_models, save_models


def main():
    print("=== Verifying Original Dataset 3 ===")
    raw_path = os.path.join(os.getcwd(), "data", "dataset3.csv")
    raw_df = pd.read_csv(raw_path)
    # original label might be 'target' or 'heart disease'
    original_label_col = 'target' if 'target' in raw_df.columns else 'heart disease'
    print("Original target values:", raw_df[original_label_col].unique())
    print("Class distribution in original data:")
    print(raw_df[original_label_col].value_counts())

    # load the cleaned, unified dataset
    proc_path = os.path.join(os.getcwd(), "data", "processed", "clean_dataset3.csv")
    df = pd.read_csv(proc_path)
    print("\nLoaded processed dataset:", df.shape)

    # Ensure label column is named 'heart disease'
    if 'target' in df.columns:
        df = df.rename(columns={'target': 'heart disease'})

    # remap labels {1,2} -> {0,1} if necessary
    unique_labels = set(df['heart disease'].unique())
    if unique_labels == {1, 2}:
        df['heart disease'] = df['heart disease'].map({1: 0, 2: 1})
        print("\nLabels mapped 1->0, 2->1")
    else:
        print("\nLabels assumed already 0/1")
    print("Post-mapping class counts:")
    print(df['heart disease'].value_counts())

    # features and target
    X = df.drop('heart disease', axis=1)
    y = df['heart disease']

    # stratified split to preserve distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("\nAfter stratified split:")
    print(" y_train distribution:\n", y_train.value_counts())
    print(" y_test  distribution:\n", y_test.value_counts())

    # train and evaluate
    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)

    for name, res in results.items():
        print(f"\n=== Model: {name} ===")
        print("Classification Report:")
        print(res['report'])
        print("Confusion Matrix:")
        print(res['confusion_matrix'])

    # save models
    save_dir = os.path.join('outputs', 'models_dataset3')
    save_models(models, save_dir)
    print(f"\nModels saved under {save_dir}")


if __name__ == '__main__':
    main()

