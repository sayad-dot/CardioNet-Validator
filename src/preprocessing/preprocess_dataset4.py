# src/preprocessing/preprocess_dataset4.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # for relative imports

from preprocessing.preprocess_utils import *

def preprocess_dataset4():
    df = load_dataset('data/external/dataset4.csv')
    print("Original shape:", df.shape)

    # Rename the target column
    df.rename(columns={"DEATH_EVENT": "heart disease"}, inplace=True)

    df = handle_missing_values(df)
    print("After missing value removal:", df.shape)

    df = encode_categorical(df)
    df = normalize_numerical(df)
    print("Final columns:", df.columns)

    save_dataset(df, 'data/processed/clean_dataset4.csv')
    print("Dataset 4 cleaned and saved.")

if __name__ == '__main__':
    preprocess_dataset4()
