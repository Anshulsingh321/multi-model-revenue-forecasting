import pandas as pd

def load_data(file):
    df = pd.read_csv(file, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if df['date'].isna().any():
        raise ValueError("Invalid or missing date values in dataset")
    df = df.sort_values('date')
    df = df.dropna(subset=['date'])
    return df