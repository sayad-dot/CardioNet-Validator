import pandas as pd

# Load all processed datasets
df1 = pd.read_csv('data/processed/clean_dataset1.csv')
df2 = pd.read_csv('data/processed/clean_dataset2.csv') 
df3 = pd.read_csv('data/processed/clean_dataset3.csv')

print("=== DATASET 1 ===")
print("Columns:", df1.columns.tolist())
print("Shape:", df1.shape)
print("Target values:", df1['heart disease'].unique())
print("First few rows:")
print(df1.head(2))

print("\n=== DATASET 2 ===")
print("Columns:", df2.columns.tolist()) 
print("Shape:", df2.shape)
print("Target values:", df2['heart disease'].unique())
print("First few rows:")
print(df2.head(2))

print("\n=== DATASET 3 ===")
print("Columns:", df3.columns.tolist())
print("Shape:", df3.shape) 
print("Target values:", df3['heart disease'].unique())
print("First few rows:")
print(df3.head(2))

# Check if all datasets have the same columns
cols1 = set(df1.columns)
cols2 = set(df2.columns)
cols3 = set(df3.columns)

print("\n=== COLUMN COMPARISON ===")
print("All datasets have same columns?", cols1 == cols2 == cols3)
if cols1 != cols2:
    print("Difference between Dataset 1 & 2:", cols1.symmetric_difference(cols2))
if cols1 != cols3:
    print("Difference between Dataset 1 & 3:", cols1.symmetric_difference(cols3))
if cols2 != cols3:
    print("Difference between Dataset 2 & 3:", cols2.symmetric_difference(cols3))