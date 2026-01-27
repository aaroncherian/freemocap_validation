from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter

QUALISYS_REL = Path("validation/qualisys/gait_parameters/qualisys_gait_metrics.csv")
FMC_REL      = Path("validation/mediapipe_dlc/gait_parameters/gait_metrics.csv")



def load_gait_dataframe(
        folder_list: list,
        qualisys_rel: Path = QUALISYS_REL,
        fmc_rel: Path = FMC_REL
) -> pd.DataFrame:

    rows = [] 
       








path_to_recordings = Path(r'D:\2023-06-07_TF01\1.0_recordings\four_camera')

list_of_folders = [p for p in path_to_recordings.iterdir() if p.is_dir()]

list_of_valid_folders = []
for p in list_of_folders:
    if (p/'validation').is_dir():
        list_of_valid_folders.append(p)
    else:
        print(f"Skipping {p}")

load_gait_dataframe(list_of_folders)



f = 2