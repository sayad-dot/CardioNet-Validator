# src/preprocessing/preprocess_dataset1.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # So we can import from src

from preprocessing.preprocess_utils import *

def preprocess_dataset1():
    df = load_dataset('data/dataset1.csv')
    print("Original shape:", df.shape)

    df = handle_missing_values(df)
    print("After missing value removal:", df.shape)

    df = encode_categorical(df)
    df = normalize_numerical(df)

    save_dataset(df, 'data/clean_dataset1.csv')
    print("Dataset 1 cleaned and saved.")

if __name__ == '__main__':
    preprocess_dataset1()
