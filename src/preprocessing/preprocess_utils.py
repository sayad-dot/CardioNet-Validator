# src/preprocessing/preprocess_utils.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_dataset(path):
    return pd.read_csv(path)

def handle_missing_values(df):
    # Drop rows with missing values (or you can use df.fillna if appropriate)
    return df.dropna()

def encode_categorical(df):
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        if col.lower() not in ['target', 'heart disease']:  # Exclude label
            df[col] = le.fit_transform(df[col])
    return df

def normalize_numerical(df):
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    num_cols = [col for col in num_cols if col.lower() not in ['target', 'heart disease']]  # Exclude label
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

def save_dataset(df, path):
    df.to_csv(path, index=False)

