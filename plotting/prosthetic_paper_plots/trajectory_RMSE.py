from pathlib import Path
import pandas as pd


TRACKER = "rtmpose"

path_to_recordings = Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera")
list_of_folders = [p for p in path_to_recordings.iterdir() if p.is_dir()]


list_of_valid_folders: list[Path] = []
for p in list_of_folders:
    if (p / "validation").is_dir():
        list_of_valid_folders.append(p)
    else:
        print(f"Skipping {p}")

mean_rmses_per_trial = pd.DataFrame()

for valid_folder in list_of_valid_folders:
    print(f"Processing {valid_folder}...")

    path_to_rmses = valid_folder / "validation" / TRACKER / "trajectories" / "trajectories_per_stride_rmse_stats.csv"
    stride_rmse = pd.read_csv(path_to_rmses)
    mean_of_stride_rmses = stride_rmse.groupby(["marker"]).agg(rmse_x_mean = ("rmse_x", "mean"), rmse_y_mean = ("rmse_y", "mean"), rmse_z_mean = ("rmse_z", "mean"))
    mean_of_stride_rmses["trial_name"] = valid_folder.name
    mean_rmses_per_trial = pd.concat([mean_rmses_per_trial, mean_of_stride_rmses], ignore_index=False)
    f = 2

total_mean_rmses = mean_rmses_per_trial.groupby("marker").agg(total_x_rmse = ("rmse_x_mean", "mean"),
                                                              total_x_std =  ("rmse_x_mean", "std"), 
                                                              total_y_rmse = ("rmse_y_mean", "mean"),
                                                              total_y_std = ("rmse_y_mean", "std"),
                                                              total_z_rmse = ("rmse_z_mean", "mean"),
                                                              total_z_std = ("rmse_z_mean", "std"))



f = 2