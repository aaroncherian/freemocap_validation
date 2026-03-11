from pathlib import Path
import pandas as pd
from scipy.signal import argrelextrema
import numpy as np


recordings = {
        "neutral": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1",
        "neg_2_8": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_03_15_TF01_flexion_neg_2_8_trial_1",
        "neg_5_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_11_55_05_TF01_flexion_neg_5_6_trial_1",
        "pos_2_8": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_09_05_TF01_flexion_pos_2_8_trial_1",
        "pos_5_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_12_36_TF01_flexion_pos_5_6_trial_1",
    }

trackers = ["qualisys", "rtmpose_dlc"]  # column order: left then right
summaries = []
for condition, recording in recordings.items():
    for tracker in trackers:
        recording = Path(recording)
        path_to_data = recording/"validation"/tracker/"trajectories"/"trajectories_per_stride.csv"
        
        data = pd.read_csv(path_to_data)

        right_toe_height = data.query("marker == 'right_foot_index'")[['z', 'cycle','percent_gait_cycle']]

        right_toe_height_per_cycle = right_toe_height.groupby("cycle")

        rows = []
        for cycle, groupby in right_toe_height_per_cycle:

            stance_min_idx = groupby["z"].idxmin()
            stance_min_pct = groupby.loc[stance_min_idx, "percent_gait_cycle"]
            
            swing = groupby.query("percent_gait_cycle > 70 and percent_gait_cycle<95").reset_index(drop=True)
            swing_min_idx = swing['z'].idxmin()

            row = swing.loc[swing_min_idx]
            rows.append(row)
        df = pd.DataFrame(rows)
        summary_df = pd.DataFrame([{
            "mean_height": df["z"].mean(),
            "std_height": df["z"].std(),
            "mean_pct": df["percent_gait_cycle"].mean(),
            "std_pct": df["percent_gait_cycle"].std(),
            "condition": condition,
            "tracker": tracker
        }])
        summaries.append(summary_df)

df_all = pd.concat(summaries, ignore_index=True)

import numpy as np
import plotly.graph_objects as go

# --- Figure dimensions ---
FIG_W_IN = 1.8
FIG_H_IN = 1.3
DPI = 300
fig_width_px = int(FIG_W_IN * DPI)
fig_height_px = int(FIG_H_IN * DPI)

# --- Font sizes ---
BASE_FONT = 15
TICK_FONT = 14
LEGEND_FONT = 14

# --- Colors ---
COLOR_FMC = "#1f77b4"
COLOR_Q = "#d62728"
MARKER_SIZE = 7

# --- Condition setup ---
COND_ORDER = ["neg_5_6", "neg_2_8", "neutral", "pos_2_8", "pos_5_6"]
tick_label_map = {
    "neg_5_6": "−5.6°",
    "neg_2_8": "−2.8°",
    "neutral": "Neutral",
    "pos_2_8": "+2.8°",
    "pos_5_6": "+5.6°",
}
tick_text = [tick_label_map[c] for c in COND_ORDER]

# --- Ensure condition ordering ---
df_all["condition"] = pd.Categorical(df_all["condition"], categories=COND_ORDER, ordered=True)
df_all = df_all.sort_values(["condition", "tracker"])

# --- Numeric x with jitter ---
x_base = np.arange(len(COND_ORDER))
offset = 0.1
x_fmc = x_base - offset
x_q = x_base + offset

# --- Build plot ---
fig = go.Figure()

# FreeMoCap trace
df_fmc = df_all.query("tracker == 'rtmpose_dlc'").reset_index(drop=True)
fig.add_trace(
    go.Scatter(
        x=x_fmc,
        y=df_fmc["mean_height"],
        mode="markers+lines",
        name="FreeMoCap-DLC",
        marker=dict(color=COLOR_FMC, size=MARKER_SIZE, symbol="circle",
                    line=dict(width=0.5, color="black")),
        line=dict(width=1.5, color=COLOR_FMC),
        error_y=dict(type="data", array=df_fmc["std_height"],
                     visible=True, thickness=1.2, width=4),
        hovertemplate=(
            "Condition: %{customdata}<br>"
            "Mean: %{y:.1f} mm<br>"
            "SD: %{error_y.array:.1f} mm<extra></extra>"
        ),
        customdata=df_fmc["condition"],
    )
)

# Qualisys trace
df_q = df_all.query("tracker == 'qualisys'").reset_index(drop=True)
fig.add_trace(
    go.Scatter(
        x=x_q,
        y=df_q["mean_height"],
        mode="markers+lines",
        name="Qualisys",
        marker=dict(color=COLOR_Q, size=MARKER_SIZE, symbol="square",
                    line=dict(width=0.5, color="black")),
        line=dict(width=1.5, color=COLOR_Q),
        error_y=dict(type="data", array=df_q["std_height"],
                     visible=True, thickness=1.2, width=4),
        hovertemplate=(
            "Condition: %{customdata}<br>"
            "Mean: %{y:.1f} mm<br>"
            "SD: %{error_y.array:.1f} mm<extra></extra>"
        ),
        customdata=df_q["condition"],
    )
)

# --- Layout ---
fig.update_layout(
    template="simple_white",
    width=fig_width_px,
    height=fig_height_px,
    font=dict(family="Arial", size=BASE_FONT, color="black"),
    margin=dict(l=55, r=15, t=10, b=45),
    xaxis=dict(
        title="<b>Prosthetic flexion adjustment (°)</b>",
        tickmode="array", tickvals=x_base, ticktext=tick_text,
        title_font=dict(size=BASE_FONT), tickfont=dict(size=TICK_FONT),
        showline=True, linecolor="black", mirror=True,
        ticks="outside", ticklen=4,
    ),
    yaxis=dict(
        title="<b>Minimum toe clearance (mm)</b>",
        title_font=dict(size=BASE_FONT), tickfont=dict(size=TICK_FONT),
        showline=True, linecolor="black", mirror=True,
        ticks="outside", ticklen=4, zeroline=False,
    ),
    legend=dict(
        orientation="h", x=0.02, y=0.98,
        xanchor="left", yanchor="top",
        bgcolor="rgba(255,255,255,0.7)",
        bordercolor="rgba(0,0,0,0.2)", borderwidth=1,
        font=dict(size=LEGEND_FONT),
    ),
)

fig.show()

import plotly.io as pio
pio.kaleido.scope.mathjax = None
path_to_save = Path(r"C:\Users\aaron\Documents\prosthetics_paper")
fig.write_image(path_to_save / "toe_clearance.pdf")

f = 2 
