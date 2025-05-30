# File: src/pipeline/predict_final_model.py

import os
import joblib
import pandas as pd

from src.preprocessing.preprocess_utils import (
    load_dataset,
    handle_missing_values,
    encode_categorical,
    normalize_numerical
)


def predict(input_csv: str) -> pd.DataFrame:
    """
    Load input CSV, apply preprocessing, load final XGBoost model, and return
    a DataFrame with original data plus predicted labels and probabilities.
    """
    # Load raw input
    df_orig = pd.read_csv(input_csv)
    df = df_orig.copy()

    # Preprocess exactly as in training
    df = handle_missing_values(df)
    df = encode_categorical(df)
    df = normalize_numerical(df)

    # Drop target column if present
    if 'heart disease' in df.columns:
        df = df.drop(columns=['heart disease'])

    # Load trained model
    model_path = os.path.join('outputs', 'final_model', 'xgb_dataset3.pkl')
    model = joblib.load(model_path)

    # Make predictions
    preds = model.predict(df)
    probas = model.predict_proba(df)[:, 1]

    # Attach results to original DataFrame
    df_orig['predicted_label'] = preds
    df_orig['predicted_proba'] = probas
    return df_orig


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print('Usage: python -m src.pipeline.predict_final_model <input_csv>')
        sys.exit(1)

    input_path = sys.argv[1]
    output_df = predict(input_path)
    print(output_df)
