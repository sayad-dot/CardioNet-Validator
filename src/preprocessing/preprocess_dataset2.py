
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.preprocess_utils import *

def preprocess_dataset2():
    df = load_dataset('data/dataset2.csv')
    print("Original shape:", df.shape)

    df = handle_missing_values(df)
    print("After missing value removal:", df.shape)

    df = encode_categorical(df)
    df = normalize_numerical(df)

    save_dataset(df, 'data/processed/clean_dataset2.csv')
    print("Dataset 2 cleaned and saved.")

if __name__ == '__main__':
    preprocess_dataset2()
