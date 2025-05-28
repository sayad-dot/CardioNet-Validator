# fix_target_encoding.py
import pandas as pd

def fix_dataset_targets():
    """
    Fix inconsistent target encoding across all datasets
    """
    datasets = [
        ('data/processed/clean_dataset1.csv', 'Dataset1'),
        ('data/processed/clean_dataset2.csv', 'Dataset2'), 
        ('data/processed/clean_dataset3.csv', 'Dataset3')
    ]
    
    for path, name in datasets:
        print(f"\n=== Fixing {name} ===")
        df = pd.read_csv(path)
        
        # Check current target values
        target_col = 'heart disease'
        current_values = sorted(df[target_col].unique())
        print(f"Current target values: {current_values}")
        
        # Standardize to [0, 1]
        if set(current_values) == {1, 2}:
            df[target_col] = df[target_col].map({1: 0, 2: 1})
            print("Converted [1,2] -> [0,1]")
        elif set(current_values) == {0, 1}:
            print("Already in correct format [0,1]")
        else:
            print(f"⚠️  Unexpected values: {current_values}")
            continue
            
        # Verify conversion
        final_values = sorted(df[target_col].unique())
        print(f"Final target values: {final_values}")
        
        # Save fixed dataset
        df.to_csv(path, index=False)
        print(f"✅ {name} fixed and saved")

if __name__ == "__main__":
    fix_dataset_targets()
    print("\n🎉 All datasets now have consistent target encoding [0,1]")