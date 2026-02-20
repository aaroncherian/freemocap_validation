from pathlib import Path
import pandas as pd
from collections import OrderedDict



TRACKERS = ["mediapipe", "rtmpose", "rtmpose_dlc"]

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
    for tracker in TRACKERS:

        path_to_rmses = valid_folder / "validation" / tracker / "trajectories" / "trajectories_per_stride_rmse_stats.csv"
        stride_rmse = pd.read_csv(path_to_rmses)

        mean_of_stride_rmses = stride_rmse.groupby(
            ["marker"]
            ).agg(
                rmse_x_mean = ("rmse_x", "mean"), 
                rmse_y_mean = ("rmse_y", "mean"), 
                rmse_z_mean = ("rmse_z", "mean")
                ).reset_index()
        mean_of_stride_rmses["trial_name"] = valid_folder.name
        mean_of_stride_rmses["tracker"] = tracker
        mean_rmses_per_trial = pd.concat([mean_rmses_per_trial, mean_of_stride_rmses], ignore_index=True)
        f = 2

total_mean_rmses = mean_rmses_per_trial.groupby(["marker", "tracker"]).agg(total_x_rmse = ("rmse_x_mean", "mean"),
                                                              total_x_std =  ("rmse_x_mean", "std"), 
                                                              total_y_rmse = ("rmse_y_mean", "mean"),
                                                              total_y_std = ("rmse_y_mean", "std"),
                                                              total_z_rmse = ("rmse_z_mean", "mean"),
                                                              total_z_std = ("rmse_z_mean", "std"))


prosthetic_markers = [
    "right_ankle",
    "right_knee",
    "right_heel",
    "right_foot_index",
]

marker_map = {
    "right_ankle": "Right Ankle",
    "right_knee": "Right Knee",
    "right_heel": "Right Heel",
    "right_foot_index": "Right Toe",
}

tracker_map = {
    "mediapipe": "MediaPipe",
    "rtmpose": "RTMPose",
    "rtmpose_dlc": "DeepLabCut",
}

# Filter + rename
table = (
    total_mean_rmses
    .reset_index()
    .query("marker in @prosthetic_markers")
    .copy()
)

table["marker"] = table["marker"].map(marker_map)
table["tracker"] = table["tracker"].map(tracker_map)

# Sort anatomically
marker_order = [ "Right Knee", "Right Ankle", "Right Heel", "Right Toe"]
tracker_order = ["MediaPipe", "RTMPose", "DeepLabCut"]

table["marker"] = pd.Categorical(table["marker"], marker_order)
table["tracker"] = pd.Categorical(table["tracker"], tracker_order)

table = table.sort_values(["marker", "tracker"])

# Format mean ± SD
for dim in ["x", "y", "z"]:
    table[f"{dim}_formatted"] = (
        table[f"total_{dim}_rmse"].round(1).astype(str)
        + " $\\pm$ "
        + table[f"total_{dim}_std"].round(1).astype(str)
    )

# --- Build multirow tabular body ---
rows = []

# group by marker in your sorted order
for marker, g in table.groupby("marker", sort=False):
    g = g.sort_values("tracker")  # tracker categorical already enforces order
    n = len(g)

    for i, (_, row) in enumerate(g.iterrows()):
        marker_cell = f"\\multirow{{{n}}}{{*}}{{{marker}}}" if i == 0 else ""
        rows.append(
            f"{marker_cell} & {row['tracker']} & "
            f"{row['x_formatted']} & {row['y_formatted']} & {row['z_formatted']} \\\\"
        )

    # optional: separator line between markers (looks nice with booktabs)
    rows.append("\\midrule")

# remove last midrule
if rows and rows[-1] == "\\midrule":
    rows.pop()

tabular_body = "\n".join(rows)

latex_output = f"""
\\begin{{table}}[t]
\\centering
\\caption{{Prosthetic-side marker RMSE (mm), mean $\\pm$ SD across trials.}}
\\label{{tab:prosthetic_rmse}}
\\resizebox{{\\columnwidth}}{{!}}{{%
\\begin{{tabular}}{{llccc}}
\\toprule
Marker & Tracker & X (mm) & Y (mm) & Z (mm) \\\\
\\midrule
{tabular_body}
\\\\
\\bottomrule
\\end{{tabular}}%
}}
\\end{{table}}
"""

print(latex_output)
f = 2