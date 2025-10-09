import pandas as pd
import sys

def validate_data(path):
    df = pd.read_csv(path)
    print(f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    print("📊 Columns:", list(df.columns))
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✅ No missing values found")
    else:
        print("⚠️ Missing values detected:")
        print(missing[missing > 0])

    print("\n🎯 Target column distribution:")
    if 'Churn' in df.columns:
        print(df['Churn'].value_counts(normalize=True))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Usage: python data_validator.py <path_to_csv>")
    else:
        validate_data(sys.argv[1])
