import pandas as pd
from pathlib import Path

def load_csv(path: Path):
    return pd.read_csv(path)

def save_csv(df, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)