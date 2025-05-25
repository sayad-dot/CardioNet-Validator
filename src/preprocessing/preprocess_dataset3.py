import pandas as pd
import os
from src.preprocessing.preprocess_utils import *

# File paths
raw_path = "data/dataset3.csv"
processed_path = "data/processed/dataset3_cleaned.csv"

# Load raw data
df = pd.read_csv(raw_path)
print("Original shape:", df.shape)

# Drop rows with missing values
df.dropna(inplace=True)
print("After missing value removal:", df.shape)

# Save cleaned data
os.makedirs("data/processed", exist_ok=True)
df.to_csv(processed_path, index=False)
print("Dataset 3 cleaned and saved.")
