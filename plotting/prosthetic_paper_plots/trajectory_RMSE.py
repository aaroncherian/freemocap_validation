from pathlib import Path
import pandas as pd

TRACKERS = ["mediapipe", "rtmpose", "rtmpose_dlc"]

path_to_recordings = Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera")
trial_folders = [p for p in path_to_recordings.iterdir() if p.is_dir()]

valid_trials = []
for p in trial_folders:
    if (p / "validation").is_dir():
        valid_trials.append(p)
    else:
        print(f"Skipping (no validation folder): {p}")

rows = []  # accumulate then build one DataFrame at the end (faster than repeated concat)

for trial_folder in valid_trials:
    print(f"Processing {trial_folder.name}...")

    for tracker in TRACKERS:
        path_to_rmses = (
            trial_folder
            / "validation"
            / tracker
            / "trajectories"
            / "trajectories_per_stride_rmse_stats.csv"
        )

        if not path_to_rmses.exists():
            print(f"  Missing: {path_to_rmses}")
            continue

        stride_rmse = pd.read_csv(path_to_rmses)

        # Mean across strides WITHIN this trial, per marker
        mean_per_marker = (
            stride_rmse
            .groupby("marker", as_index=False)
            .agg(
                rmse_x_mean=("rmse_x", "mean"),
                rmse_y_mean=("rmse_y", "mean"),
                rmse_z_mean=("rmse_z", "mean"),
                rmse_3d_mean=("rmse_3d", "mean"),  # optional, but your CSV has it
            )
        )

        mean_per_marker["trial_name"] = trial_folder.name
        mean_per_marker["tracker"] = tracker

        rows.append(mean_per_marker)

mean_rmses_per_trial = pd.concat(rows, ignore_index=True)

# Mean (and std) across TRIALS, per marker x tracker
total_mean_rmses = (
    mean_rmses_per_trial
    .groupby(["marker", "tracker"], as_index=False)
    .agg(
        total_x_rmse=("rmse_x_mean", "mean"),
        total_x_std =("rmse_x_mean", "std"),
        total_y_rmse=("rmse_y_mean", "mean"),
        total_y_std =("rmse_y_mean", "std"),
        total_z_rmse=("rmse_z_mean", "mean"),
        total_z_std =("rmse_z_mean", "std"),
        n_trials=("trial_name", "nunique"),  
)

print(total_mean_rmses.head())