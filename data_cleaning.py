import pandas as pd
import numpy as np
import json
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'dstrIPC_1.csv')
CLEANED_PATH = os.path.join(os.path.dirname(__file__), 'data', 'cleaned.csv')
EDA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'eda_stats.json')

# Key crime columns (avoiding sub-columns to prevent double counting)
CRIME_COLS = [
    'MURDER',
    'ATTEMPT TO MURDER',
    'CULPABLE HOMICIDE NOT AMOUNTING TO MURDER',
    'RAPE',
    'KIDNAPPING & ABDUCTION',
    'DACOITY',
    'ROBBERY',
    'BURGLARY',
    'THEFT',
    'RIOTS',
    'CRIMINAL BREACH OF TRUST',
    'CHEATING',
    'COUNTERFIETING',
    'ARSON',
    'HURT/GREVIOUS HURT',
    'DOWRY DEATHS',
    'ASSAULT ON WOMEN WITH INTENT TO OUTRAGE HER MODESTY',
    'CRUELTY BY HUSBAND OR HIS RELATIVES',
    'CAUSING DEATH BY NEGLIGENCE',
    'OTHER IPC CRIMES',
]

def clean_and_prepare():
    df = pd.read_csv(DATA_PATH)

    # Standardize column names
    df.columns = df.columns.str.strip()

    # Strip whitespace from string columns
    df['STATE/UT'] = df['STATE/UT'].str.strip().str.upper()
    df['DISTRICT'] = df['DISTRICT'].str.strip().str.upper()

    # Drop rows with any nulls (there are none, but safety check)
    df.dropna(inplace=True)

    # Remove rows where TOTAL IPC CRIMES is 0 (likely missing/corrupt entries)
    df = df[df['TOTAL IPC CRIMES'] > 0]

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Compute Safety Score (0–100)
    # Higher crimes → lower safety score
    # Use log-scaled normalization to handle extreme outliers
    log_total = np.log1p(df['TOTAL IPC CRIMES'])
    max_log = log_total.max()
    min_log = log_total.min()
    df['SAFETY_SCORE'] = 100 - ((log_total - min_log) / (max_log - min_log) * 100)
    df['SAFETY_SCORE'] = df['SAFETY_SCORE'].round(2)

    # Safety label
    def label(score):
        if score >= 75: return 'Safe'
        elif score >= 50: return 'Moderate'
        elif score >= 25: return 'Unsafe'
        else: return 'Dangerous'

    df['SAFETY_LABEL'] = df['SAFETY_SCORE'].apply(label)

    df.to_csv(CLEANED_PATH, index=False)
    print(f"Cleaned data saved: {df.shape}")

    # EDA stats
    eda = {
        "total_records": int(len(df)),
        "states": sorted(df['STATE/UT'].unique().tolist()),
        "years": sorted(df['YEAR'].unique().tolist()),
        "districts_per_state": df.groupby('STATE/UT')['DISTRICT'].nunique().to_dict(),
        "avg_safety_by_state": df.groupby('STATE/UT')['SAFETY_SCORE'].mean().round(2).to_dict(),
        "crime_totals": {col: int(df[col].sum()) for col in CRIME_COLS if col in df.columns},
        "top10_dangerous": df.groupby('DISTRICT')['TOTAL IPC CRIMES'].mean().nlargest(10).round(0).astype(int).to_dict(),
        "top10_safest": df.groupby('DISTRICT')['TOTAL IPC CRIMES'].mean().nsmallest(10).round(0).astype(int).to_dict(),
        "yearly_avg_crimes": df.groupby('YEAR')['TOTAL IPC CRIMES'].mean().round(0).astype(int).to_dict(),
    }

    with open(EDA_PATH, 'w') as f:
        json.dump(eda, f, indent=2)
    print("EDA stats saved.")
    return df

if __name__ == '__main__':
    clean_and_prepare()
