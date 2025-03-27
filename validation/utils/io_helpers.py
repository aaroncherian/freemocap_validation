import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import NamedTuple


def load_csv(path: Path):
    return pd.read_csv(path)

def save_csv(df, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


class QualisysTSVData(NamedTuple):
    dataframe: pd.DataFrame
    unix_start_time: float

def load_qualisys_tsv_and_start_timestamp(path: Path) -> QualisysTSVData:
    header_length = get_header_length(path)
    data = pd.read_csv(
        path,
        delimiter='\t',
        skiprows=header_length
    )
    with open(path, 'r') as file:
        for line in file:
            if line.startswith('TIME_STAMP'):
                timestamp_str = line.strip().split('\t')[1]
                datetime_obj = datetime.strptime(timestamp_str, '%Y-%m-%d, %H:%M:%S.%f')
                return QualisysTSVData(dataframe=data, unix_start_time=datetime_obj.timestamp())
    raise ValueError(f"No TIME_STAMP found in file: {path}")




def get_header_length(path:Path) -> int:
    """Determine the length of the header in the TSV file."""
    with path.open('r') as file:
        for i, line in enumerate(file):
            if line.startswith('TRAJECTORY_TYPES'):
                return i + 1
    raise ValueError("Header not found in the TSV file.")