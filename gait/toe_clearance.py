from pathlib import Path
import pandas as pd

conditions = {
    "neg_5_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_11_55_05_TF01_flexion_neg_5_6_trial_1",
    "neg_2_8": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_03_15_TF01_flexion_neg_2_8_trial_1",
    "neutral": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1",
    "pos_2_8": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_09_05_TF01_flexion_pos_2_8_trial_1",
    "pos_5_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_12_36_TF01_flexion_pos_5_6_trial_1",
}
tracker = "mediapipe_dlc"

summary_df = pd.DataFrame()
for condition, path in conditions.items():
    path = Path(path)
    path_to_gait_events = path/'validation'/tracker/f'{tracker}_joint_trajectory_by_stride.csv'

    df = pd.read_csv(path_to_gait_events)
    df = df.query("marker == 'toe'")
    df['condition'] = condition

    summary_df = pd.concat([summary_df, df], ignore_index = True)

toe = summary_df.loc[:, ["system","condition","stride","percent_gait_cycle","z"]].copy()

# keep only the analysis window (you chose >= 80%)
toe80 = toe.query("percent_gait_cycle >= 80").copy()

# find the index (within toe80) of the minimum z per (system, condition, stride)
idx_min = (
    toe80
    .groupby(["system","condition","stride"])["z"]
    .idxmin()
)

# extract those rows (one per stride), keep value and %GC
per_stride_min = (
    toe80.loc[idx_min, ["system","condition","stride","percent_gait_cycle","z"]]
         .rename(columns={"percent_gait_cycle":"pct_at_min",
                          "z":"toe_clear_min"})
         .reset_index(drop=True)
)

# ensure numeric (in case CSV dtypes are weird)
per_stride_min["pct_at_min"] = pd.to_numeric(per_stride_min["pct_at_min"], errors="coerce")
per_stride_min["toe_clear_min"] = pd.to_numeric(per_stride_min["toe_clear_min"], errors="coerce")

# --- summarize across strides: mean ± SD and n ---
min_summary = (
    per_stride_min
    .groupby(["system","condition"], as_index=False)
    .agg(
        mean_min_mm      = ("toe_clear_min", "mean"),
        sd_min_mm        = ("toe_clear_min", "std"),
        mean_pct_at_min  = ("pct_at_min", "mean"),
        sd_pct_at_min    = ("pct_at_min", "std"),
        n_strides        = ("toe_clear_min", "size"),
    )
    .sort_values(["system","condition"])
)

print("Per-stride minima (one row per stride):")
print(per_stride_min.head())

print("\nSummary (mean ± SD across strides):")
print(min_summary)


import plotly.graph_objects as go

COND_ORDER = ["neg_5_6","neg_2_8","neutral","pos_2_8","pos_5_6"]
COLOR_MAP = {
    "mediapipe_dlc": "rgb(31,119,180)",  # FMC blue
    "qualisys": "rgb(214,39,40)",        # QTM red
}

fig = go.Figure()
shown = set()

for cond in COND_ORDER:
    for sys in ["mediapipe_dlc", "qualisys"]:
        df_sub = per_stride_min.query("condition == @cond and system == @sys")
        if df_sub.empty:
            continue
        side = "positive" if sys == "qualisys" else "negative"

        fig.add_trace(go.Violin(
            y=df_sub["toe_clear_min"],
            x=[cond]*len(df_sub),
            name=sys,
            legendgroup=sys,
            showlegend=(sys not in shown),   # one legend entry per system
            side=side,
            line_color=COLOR_MAP[sys],
            fillcolor=COLOR_MAP[sys],
            opacity=0.6,
            box=dict(visible=True, width=0.25),      # inner boxplot
            meanline=dict(visible=True, color="black", width=1),  # mean line
            points="all",
            jitter=0.3,
            scalemode="width"
        ))
        shown.add(sys)

fig.update_layout(
    title="Split Violin: Minimum Toe Clearance by Condition",
    violinmode="overlay",  # required for split violins
    xaxis_title="Condition",
    yaxis_title="Minimum Toe Clearance (mm)",
    legend=dict(orientation="h", y=1.05),
    margin=dict(t=80, r=20, l=70, b=60)
)

fig.show()
