import pandas as pd
import sqlite3
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd

conn = sqlite3.connect("validation.db")

query = """
SELECT t.participant_code,
       t.trial_name,
       a.path,
       a.component_name,
       a.condition,
       a.tracker
FROM artifacts a
JOIN trials t ON a.trial_id = t.id
WHERE t.trial_type = "treadmill"
  AND a.category = "trajectories_per_stride"
  AND a.tracker = "mediapipe"                -- RMSE lives under the freemocap tracker
  AND a.file_exists = 1
  AND a.component_name LIKE "%rmse_stats"    -- <- key change
ORDER BY t.trial_name, a.path
"""

path_df = pd.read_sql_query(query, conn)

dfs = []
for _, row in path_df.iterrows():
    path = row["path"]
    tracker = row["tracker"]
    condition = row["condition"]
    participant = row["participant_code"]
    trial = row["trial_name"]

    sub_df = pd.read_csv(path)
    # Expect columns: cycle, marker, rmse_x, rmse_y, rmse_z, rmse_3d

    sub_df["participant_code"] = participant
    sub_df["trial_name"] = trial
    sub_df["tracker"] = tracker
    sub_df["condition"] = condition

    dfs.append(sub_df)

stride_rmse  = pd.concat(dfs, ignore_index=True)

import plotly.express as px
participant_rmse = (
    stride_rmse
    .groupby(["participant_code", "condition", "marker"], as_index=False)["rmse_3d"]
    .mean()
    .rename(columns={"rmse_3d": "participant_mean_rmse_3d"})
)

# ---- ordering ----
marker_order = [
    "left_hip", "left_knee", "left_ankle", "left_heel", "left_foot_index",
    "right_hip", "right_knee", "right_ankle", "right_heel", "right_foot_index",
]
# ---- prettier labels ----
pretty_markers = {
    "left_hip": "Left hip",
    "left_knee": "Left knee",
    "left_ankle": "Left ankle",
    "left_heel": "Left heel",
    "left_foot_index": "Left toe",
    "right_hip": "Right hip",
    "right_knee": "Right knee",
    "right_ankle": "Right ankle",
    "right_heel": "Right heel",
    "right_foot_index": "Right toe",
}
participant_rmse["marker_pretty"] = participant_rmse["marker"].map(pretty_markers)

speed_labels = {
    "speed_0_5": "0.5",
    "speed_1_0": "1.0",
    "speed_1_5": "1.5",
    "speed_2_0": "2.0",
    "speed_2_5": "2.5",
}


participant_rmse["condition_pretty"] = participant_rmse["condition"].map(speed_labels)

marker_order_pretty = [
    pretty_markers[m] for m in marker_order
]

speed_levels = ["0.5", "1.0", "1.5", "2.0", "2.5"]
colors = px.colors.sample_colorscale("Viridis", [0, 0.25, 0.5, 0.75, 1.0])
color_map = dict(zip(speed_levels, colors))

# --- base grey boxplot ---
fig = px.box(
    participant_rmse,
    x="condition_pretty",
    y="participant_mean_rmse_3d",
    color="condition_pretty",
    facet_col="marker_pretty",
    facet_col_wrap=5,
    category_orders={
        "marker_pretty": marker_order_pretty,
        "condition_pretty": speed_levels,
    },
    points="all",
    hover_data=["participant_code", "condition_pretty"],
    template="simple_white",
)

# grey boxes + grey jitter
fig.update_traces(
    selector=dict(type="box"),
    fillcolor="rgba(200,200,200,0.25)",
    line=dict(color="rgba(140,140,140,1)", width=1),
    marker=dict(color="rgba(0,0,0,0.4)", size=3),
)

# --- axes / layout cleanup ---
fig.update_yaxes(
    matches="y",
    range=[0, participant_rmse["participant_mean_rmse_3d"].max() * 1.1],
    gridcolor="rgba(0,0,0,0.08)",
    title=""
)
fig.update_xaxes(matches="x", title="")

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig.update_layout(
    title="Participant-level 3D RMSE per marker across treadmill speeds",
    xaxis_title="Treadmill speed (m/s)",
    yaxis_title="Mean 3D RMSE (mm)",
    legend_title_text="Median (m/s)",
    height=700,
    width=1300,
    margin=dict(l=70, r=20, t=80, b=70),
    font=dict(size=12),
)
fig.update_layout(showlegend=False)
fig.show()


import plotly.graph_objects as go
import plotly.express as px
agg_rmse = (
    participant_rmse
    .groupby(["condition_pretty", "marker_pretty"], as_index=False)["participant_mean_rmse_3d"]
    .agg(mean_rmse="mean", sd_rmse="std", n_participants="count")
)

# numeric speed for x-axis
speed_map = {"0.5": 0.5, "1.0": 1.0, "1.5": 1.5, "2.0": 2.0, "2.5": 2.5}
agg_rmse["speed"] = agg_rmse["condition_pretty"].map(speed_map)

# facet order (pretty names)
marker_order_pretty = [
    "Left hip", "Left knee", "Left ankle", "Left heel", "Left toe",
    "Right hip", "Right knee", "Right ankle", "Right heel", "Right toe",
]

# --- system-level summary plot ---
fig = px.line(
    agg_rmse,
    x="speed",
    y="mean_rmse",
    error_y="sd_rmse",
    facet_col="marker_pretty",
    facet_col_wrap=5,
    category_orders={"marker_pretty": marker_order_pretty},
    markers=True,
    template="simple_white",
)

# ---- axis formatting ----
fig.update_yaxes(
    showgrid=True,
    gridcolor="rgba(0,0,0,0.08)",
    zeroline=False,
)

fig.update_xaxes(
    showgrid=True,
    gridcolor="rgba(0,0,0,0.08)",
    zeroline=False,
)

# 1) Clear all axis titles first
fig.update_xaxes(title="")
fig.update_yaxes(title="")

# 2) Y-axis title only on the first column (top-left and bottom-left)
fig.layout.yaxis.title = "Mean 3D RMSE (mm)"   # row 1, col 1
fig.layout.yaxis6.title = "Mean 3D RMSE (mm)"  # row 2, col 1

# 3) X-axis titles only on the second row (subplots 6–10)
for i in range(1, 6):  # xaxis6 ... xaxis10
    getattr(fig.layout, f"xaxis{i}").title = "Treadmill speed (m/s)"


# ---- clean facet titles ----
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig.update_traces(
    line=dict(color="rgba(80,80,80,1)", width=2),
    marker=dict(color="rgba(60,60,60,1)", size=6),
    error_y=dict(color="rgba(120,120,120,1)", thickness=1.5),
)

fig.update_layout(
    title="System-level 3D RMSE vs treadmill speed (per marker)",
    height=700,
    width=1300,
    showlegend=False,
    margin=dict(l=70, r=20, t=80, b=70),
    font=dict(size=12, color="rgba(20,20,20,1)"),
    template="simple_white"
)

fig.show()

axis_long = stride_rmse.melt(
    id_vars=["participant_code", "condition", "marker"],
    value_vars=["rmse_x", "rmse_y", "rmse_z"],
    var_name="axis_raw",
    value_name="rmse"
)

# Map rmse_x / rmse_y / rmse_z → nicer axis labels
axis_map = {
    "rmse_x": "X (ML)",
    "rmse_y": "Y (AP)",
    "rmse_z": "Z (Vertical)",
}
axis_long["axis"] = axis_long["axis_raw"].map(axis_map)

# Participant-level mean RMSE per axis
participant_axis_rmse = (
    axis_long
    .groupby(["participant_code", "condition", "marker", "axis"], as_index=False)["rmse"]
    .mean()
    .rename(columns={"rmse": "participant_mean_rmse_axis"})
)

# Pretty marker + pretty condition labels (reuse your dicts)
participant_axis_rmse["marker_pretty"] = participant_axis_rmse["marker"].map(pretty_markers)
participant_axis_rmse["condition_pretty"] = participant_axis_rmse["condition"].map(speed_labels)

agg_axis_rmse = (
    participant_axis_rmse
    .groupby(["condition_pretty", "marker_pretty", "axis"], as_index=False)["participant_mean_rmse_axis"]
    .agg(mean_rmse="mean", sd_rmse="std", n_participants="count")
)

# Numeric speed for x-axis
speed_map_simple = {"0.5": 0.5, "1.0": 1.0, "1.5": 1.5, "2.0": 2.0, "2.5": 2.5}
agg_axis_rmse["speed"] = agg_axis_rmse["condition_pretty"].map(speed_map_simple)

# Order markers and axes
marker_order_pretty = [
    "Left hip", "Left knee", "Left ankle", "Left heel", "Left toe",
    "Right hip", "Right knee", "Right ankle", "Right heel", "Right toe",
]
axis_order = ["X (ML)", "Y (AP)", "Z (Vertical)"]

import plotly.express as px

fig = px.line(
    agg_axis_rmse,
    x="speed",
    y="mean_rmse",
    error_y="sd_rmse",
    color="axis",                      # X / Y / Z as separate lines
    facet_col="marker_pretty",
    facet_col_wrap=5,
    category_orders={
        "marker_pretty": marker_order_pretty,
        "axis": axis_order,
    },
    markers=True,
    template="simple_white",
)

# Styling similar to your gray theme, but keep distinct colors for axes
fig.update_traces(line=dict(width=2), marker=dict(size=5))

fig.update_yaxes(
    showgrid=True,
    gridcolor="rgba(0,0,0,0.08)",
    zeroline=False,
)
fig.update_xaxes(
    showgrid=True,
    gridcolor="rgba(0,0,0,0.08)",
    zeroline=False,
)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig.update_layout(
    title="System-level axis RMSE vs treadmill speed (per marker)",
    height=700,
    width=1300,
    margin=dict(l=70, r=20, t=80, b=70),
    font=dict(size=12),
    legend_title_text="Axis",
)

# 2) Y-axis title only on the first column (top-left and bottom-left)
fig.layout.yaxis.title = "RMSE (mm)"   # row 1, col 1
fig.layout.yaxis6.title = "RMSE (mm)"  # row 2, col 1

# 3) X-axis titles only on the second row (subplots 6–10)
for i in range(1, 6):  # xaxis6 ... xaxis10
    getattr(fig.layout, f"xaxis{i}").title = "Treadmill speed (m/s)"


fig.show()


agg_3d = agg_rmse.rename(columns={
    "mean_rmse": "mean_rmse_3d",
    "sd_rmse": "sd_rmse_3d"
})

agg_3d["axis"] = "3D"
agg_3d = agg_3d[["condition_pretty", "marker_pretty", "axis", "mean_rmse_3d", "sd_rmse_3d", "speed"]]
agg_3d = agg_3d.rename(columns={
    "mean_rmse_3d": "mean_rmse",
    "sd_rmse_3d": "sd_rmse"
})

combined_for_plot = pd.concat([
    agg_axis_rmse,   # X/Y/Z RMSE
    agg_3d           # 3D RMSE
], ignore_index=True)

import plotly.express as px

fig = px.line(
    combined_for_plot,
    x="speed",
    y="mean_rmse",
    error_y="sd_rmse",
    color="axis",
    facet_col="marker_pretty",
    facet_col_wrap=5,
    category_orders={
        "marker_pretty": marker_order_pretty,
        "axis": ["X (ML)", "Y (AP)", "Z (Vertical)", "3D"],
    },
    markers=True,
    template="simple_white",
)

# Make 3D line black and thick
fig.for_each_trace(
    lambda tr: tr.update(
        line=dict(color="black", width=3),
        marker=dict(color="black", size=7)
    ) if tr.name == "3D" else tr.update(
        line=dict(width=2),
        marker=dict(size=5)
    )
)

fig.update_yaxes(
    showgrid=True,
    gridcolor="rgba(0,0,0,0.08)",
    zeroline=False,
    title="RMSE (mm)",
    matches="y"
)
fig.update_xaxes(
    showgrid=True,
    gridcolor="rgba(0,0,0,0.08)",
    zeroline=False,
    title="Treadmill speed (m/s)",
    matches="x"
)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig.update_layout(
    title="Axis-wise and 3D RMSE vs treadmill speed (per marker)",
    height=700,
    width=1300,
    margin=dict(l=70, r=20, t=80, b=70),
    font=dict(size=12),
    legend_title_text="Measurement",
)
fig.show()