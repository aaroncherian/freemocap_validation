from pathlib import Path
import pandas as pd

conditions = {
    "neutral": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1",
    "neg_2_8": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_03_15_TF01_flexion_neg_2_8_trial_1",
    "neg_5_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_11_55_05_TF01_flexion_neg_5_6_trial_1",
    "pos_2_8": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_09_05_TF01_flexion_pos_2_8_trial_1",
    "pos_5_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_12_36_TF01_flexion_pos_5_6_trial_1",
}
tracker = "mediapipe_dlc"

summary_strides_df = pd.DataFrame()
summary_means_df = pd.DataFrame()

for condition, path in conditions.items():
    path = Path(path)


    path_to_angle_strides = path/'validation'/tracker/f'{tracker}_joint_angle_by_stride.csv'

    df = pd.read_csv(path_to_angle_strides)
    df = df[df["angle"] == "ankle_dorsi_plantar_r"]      # keep only that angle
    df = df.rename(columns={"value": "ankle_angle"})     # match old naming
    df = df.drop(columns=["angle"])                      # optional: drop the angle column
    df["condition"] = condition

    means = df.groupby(["system", "percent_gait_cycle"], as_index=False)["ankle_angle"].mean()
    means['condition'] = condition

    summary_strides_df = pd.concat([summary_strides_df, df], ignore_index = True)
    summary_means_df = pd.concat([summary_means_df, means], ignore_index=True)

g = summary_means_df.groupby(["system", "condition"])

# Row indices of the max/min within each group
idx_max = g["ankle_angle"].idxmax()
idx_min = g["ankle_angle"].idxmin()

# Grab those rows and label them
rows_max = summary_means_df.loc[idx_max].assign(metric="max")
rows_min = summary_means_df.loc[idx_min].assign(metric="min")

# Stack and keep just what we need
peaks = (
    pd.concat([rows_max, rows_min], ignore_index=True)
      .loc[:, ["system", "condition", "metric", "percent_gait_cycle", "ankle_angle"]]
      .rename(columns={
          "percent_gait_cycle": "pct_gait_cycle",
          "ankle_angle": "value"
      })
      .sort_values(["system", "condition", "metric"], ignore_index=True)
)

peaks_wide = (
    peaks.pivot(index=["system", "condition"], columns="metric", values="value")
          .reset_index()
)

peaks_wide["ROM"] = peaks_wide["max"] - peaks_wide["min"]


neutral_rom = (
    peaks_wide.query("condition == 'neutral'")
          .rename(columns={"ROM": "ROM_neutral"})
          .loc[:, ["system", "ROM_neutral"]]
)

rom_df = peaks_wide.merge(neutral_rom, on="system", how="left")
rom_df["ROM_delta"] = rom_df["ROM"] - rom_df["ROM_neutral"]

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Assume rom_df has: system, condition, ROM, ROM_delta
fig = make_subplots(
    rows=1, cols=1,
    subplot_titles=("ΔROM (relative to neutral)",)
)

# --- Left subplot: ROM ---
for sys, group in rom_df.groupby("system"):
    fig.add_trace(
        go.Scatter(
            x=group["condition"],
            y=group["ROM_delta"],
            mode="markers+lines",
            name=sys,
            showlegend=False  # keep legend only once
        ),
        row=1, col=1
    )


fig.update_layout(
    height=500,
    width=1000,
    title_text="ΔROM vs. neutral",
    xaxis_title="Condition",

)
fig.show()


f = 2
