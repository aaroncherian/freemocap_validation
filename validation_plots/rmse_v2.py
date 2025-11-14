import pandas as pd
import sqlite3
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import plotly.express as px

# ---- Database connection and data loading ----
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
    AND a.tracker IN ("mediapipe", "qualisys")
    AND a.file_exists = 1
    AND a.component_name LIKE "%summary_stats"
ORDER BY t.trial_name, a.path
"""
def rmse(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(arr**2)))

path_df = pd.read_sql_query(query, conn)
dfs = []

for _, row in path_df.iterrows():
    path = row["path"]
    tracker = row["tracker"]
    condition = row.get("condition") or ""
    participant = row["participant_code"]
    trial = row["trial_name"]

    sub_df = pd.read_csv(path)
    sub_df["participant_code"] = participant
    sub_df["trial_name"] = trial
    sub_df["tracker"] = tracker
    sub_df["condition"] = condition if condition else "none"
    dfs.append(sub_df)

combined_df = pd.concat(dfs, ignore_index=True)


# ---------- 1) keep only mean cycles ----------
# your CSV has both mean & std rows; we only want mean
mean_df = combined_df.query("stat == 'mean'").copy()

# (optional but nice) restrict to xyz axes if other axes exist
mean_df = mean_df[mean_df["axis"].isin(["x", "y", "z"])]

# ---------- 2) participant-level mean across trials ----------
# If a participant has multiple trials at the same condition,
# this averages those "mean cycles" into a participant-level mean.
participant_mean_df = (
    mean_df
    .groupby(
        [
            "participant_code",
            "condition",
            "tracker",
            "marker",
            "axis",
            "percent_gait_cycle",
        ],
        as_index=False,
    )["value"]
    .mean()
)

# ---------- 3) compute per-participant RMSE (paper-style) ----------

rows = []

for (participant, condition, marker), sub in participant_mean_df.groupby(
    ["participant_code", "condition", "marker"]
):
    # pivot to get shape: index = percent_gait_cycle,
    # columns = (tracker, axis) -> values = value
    wide = (
        sub.pivot_table(
            index="percent_gait_cycle",
            columns=["tracker", "axis"],
            values="value",
        )
        .sort_index()
    )

    # need both systems present; skip otherwise
    if not {("mediapipe", "x"), ("qualisys", "x")}.issubset(wide.columns):
        print(f"Skipping {participant}, {condition}, {marker} (missing tracker data)")
        continue

    # extract xyz trajectories for each system in consistent order
    try:
        fmc_xyz = wide[
            [("mediapipe", "x"), ("mediapipe", "y"), ("mediapipe", "z")]
        ].to_numpy()
        qual_xyz = wide[
            [("qualisys", "x"), ("qualisys", "y"), ("qualisys", "z")]
        ].to_numpy()
    except KeyError:
        print(f"Skipping {participant}, {condition}, {marker} (missing axis data)")
        continue

    # safety: shapes must match
    if fmc_xyz.shape != qual_xyz.shape:
        print(
            f"Shape mismatch for {participant}, {condition}, {marker}: "
            f"FMC {fmc_xyz.shape}, QUAL {qual_xyz.shape}"
        )
        continue

    diff = fmc_xyz - qual_xyz  # [N_time, 3]

    rmse_x = rmse(diff[:, 0])
    rmse_y = rmse(diff[:, 1])
    rmse_z = rmse(diff[:, 2])

    err_3d = np.linalg.norm(diff, axis=1)
    rmse_3d = rmse(err_3d)

    rows.append(
        {
            "participant_code": participant,
            "condition": condition,
            "marker": marker,
            "rmse_x": rmse_x,
            "rmse_y": rmse_y,
            "rmse_z": rmse_z,
            "rmse_3d": rmse_3d,
        }
    )

participant_rmse = pd.DataFrame(rows)

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
axis_long = participant_rmse.melt(
    id_vars=["participant_code", "condition", "marker"],
    value_vars=["rmse_x", "rmse_y", "rmse_z"],
    var_name="axis_raw",
    value_name="rmse"
)

axis_map = {
    "rmse_x": "X (ML)",
    "rmse_y": "Y (AP)",
    "rmse_z": "Z (Vertical)",
}
axis_long["axis"] = axis_long["axis_raw"].map(axis_map)

# participant-level mean (in case there are multiple rows per person)
participant_axis_rmse = (
    axis_long
    .groupby(["participant_code", "condition", "marker", "axis"], as_index=False)["rmse"]
    .mean()
    .rename(columns={"rmse": "participant_mean_rmse_axis"})
)

# pretty labels
participant_axis_rmse["marker_pretty"] = participant_axis_rmse["marker"].map(pretty_markers)
participant_axis_rmse["condition_pretty"] = participant_axis_rmse["condition"].map(speed_labels)

# group-level stats for axes
agg_axis_rmse = (
    participant_axis_rmse
    .groupby(["condition_pretty", "marker_pretty", "axis"], as_index=False)["participant_mean_rmse_axis"]
    .agg(mean_rmse="mean", sd_rmse="std", n_participants="count")
)

# numeric speed for x-axis
speed_map_simple = {"0.5": 0.5, "1.0": 1.0, "1.5": 1.5, "2.0": 2.0, "2.5": 2.5}
agg_axis_rmse["speed"] = agg_axis_rmse["condition_pretty"].map(speed_map_simple)

marker_order_pretty = [
    "Left hip", "Left knee", "Left ankle", "Left heel", "Left toe",
    "Right hip", "Right knee", "Right ankle", "Right heel", "Right toe",
]
axis_order = ["X (ML)", "Y (AP)", "Z (Vertical)"]

# --- 2. 3D RMSE in the same tidy format ---

participant_3d = participant_rmse.copy()
participant_3d["marker_pretty"] = participant_3d["marker"].map(pretty_markers)
participant_3d["condition_pretty"] = participant_3d["condition"].map(speed_labels)

agg_3d = (
    participant_3d
    .groupby(["condition_pretty", "marker_pretty"], as_index=False)["rmse_3d"]
    .agg(mean_rmse="mean", sd_rmse="std", n_participants="count")
)

agg_3d["axis"] = "3D"
agg_3d["speed"] = agg_3d["condition_pretty"].map(speed_map_simple)

# columns to match axis table
agg_3d = agg_3d[["condition_pretty", "marker_pretty", "axis", "mean_rmse", "sd_rmse", "speed"]]

# --- 3. Combine axes + 3D and plot ---

combined_for_plot = pd.concat(
    [agg_axis_rmse, agg_3d],
    ignore_index=True
)

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

# 3D line: black + thicker, others: normal
fig.for_each_trace(
    lambda tr: tr.update(
        line=dict(color="black", width=1),
        marker=dict(color="black", size=5),
        opacity=.7
    ) if tr.name == "3D" else tr.update(
        line=dict(width=1),
        marker=dict(size=5),
        opacity=0.7
    )
)

fig.update_yaxes(
    showgrid=True,
    gridcolor="rgba(0,0,0,0.08)",
    zeroline=False,
    title="RMSE (mm)",
    matches="y",
)
fig.update_xaxes(
    showgrid=True,
    gridcolor="rgba(0,0,0,0.08)",
    zeroline=False,
    title="Treadmill speed (m/s)",
    matches="x",
)

# clean facet titles ("marker_pretty=Left hip" → "Left hip")
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig.update_layout(
    title="Axis-wise and 3D RMSE vs treadmill speed (per marker)",
    height=700,
    width=1000,
    margin=dict(l=70, r=20, t=80, b=70),
    font=dict(size=12),
    legend_title_text="Measurement",
)
# 1) Clear all axis titles first
fig.update_xaxes(title="")
fig.update_yaxes(title="")

# 2) Y-axis title only on the first column (top-left and bottom-left)
fig.layout.yaxis.title = "Mean RMSE (mm)"   # row 1, col 1
fig.layout.yaxis6.title = "Mean RMSE (mm)"  # row 2, col 1

# 3) X-axis titles only on the second row (subplots 6–10)
for i in range(1, 6):  # xaxis6 ... xaxis10
    getattr(fig.layout, f"xaxis{i}").title = "Treadmill speed (m/s)"

fig.write_html("docs/gait_data/trajectory_rmse.html", include_plotlyjs="cdn", full_html=True)