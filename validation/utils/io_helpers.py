import pandas as pd
from pathlib import Path

def load_csv(path: Path):
    return pd.read_csv(path)

def save_csv(df, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


## for loading qualisys_markers
def load_tsv(path: Path) -> pd.DataFrame:
    """Load the TSV file, skipping the header."""
    header_length = get_header_length(path)
    data = pd.read_csv(
        path,
        delimiter='\t',
        skiprows=header_length
    )
    return data

def get_header_length(path:Path) -> int:
    """Determine the length of the header in the TSV file."""
    with path.open('r') as file:
        for i, line in enumerate(file):
            if line.startswith('TRAJECTORY_TYPES'):
                return i + 1
    raise ValueError("Header not found in the TSV file.")