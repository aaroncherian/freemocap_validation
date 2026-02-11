import re
import pandas as pd
import sqlite3
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ------------------------
# 1) Load data from SQLite
# ------------------------

TRACKERS = ["mediapipe", "rtmpose", "qualisys",]
root_dir = Path(r"D:\validation\joint_angles")
root_dir.mkdir(exist_ok=True, parents=True)

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
  AND a.category = "joint_angles_per_stride"
  AND a.tracker IN ("mediapipe", "rtmpose", "qualisys")
  AND a.file_exists = 1
  AND a.condition LIKE "speed_%"
  AND a.component_name LIKE "%summary_stats"
ORDER BY t.trial_name, a.path
"""
reference_system = "qualisys"

path_df = pd.read_sql_query(query, conn)

dfs = []
for _, row in path_df.iterrows():
    sub = pd.read_csv(row["path"])
    sub["participant_code"] = row["participant_code"]
    sub["trial_name"] = row["trial_name"]
    sub["tracker"] = (row["tracker"] or "").lower()
    sub["condition"] = row["condition"] or "none"
    dfs.append(sub)

combined_df = pd.concat(dfs, ignore_index=True)

# ------------------------
# 2) Normalize + keep only major motions
# ------------------------
for col in ["joint", "side", "tracker", "stat", "component"]:
    if col in combined_df.columns:
        combined_df[col] = combined_df[col].astype(str).str.lower()

combined_df["component"] = combined_df["component"].replace(
    {"inversion_eversion": "inv_ev"}
)

MAJOR = {
    ("hip", "flex_ext"),
    ("knee", "flex_ext"),
    ("ankle", "dorsi_plantar"),
}
combined_df = combined_df[
    combined_df.apply(lambda r: (r["joint"], r["component"]) in MAJOR, axis=1)
].copy()

JOINT_ORDER = ["hip", "knee", "ankle"]

COMP_LABEL = {
    "flex_ext": "Flex/Ext",
    "dorsi_plantar": "Dorsi/Plantar",
}

# ------------------------
# 3) Speed parsing / ordering
# ------------------------
def speed_key(cond: str) -> float:
    m = re.search(r"speed_(\d+)[_\.](\d+)", str(cond))
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    m2 = re.search(r"speed_(\d+)", str(cond))
    if m2:
        return float(m2.group(1))
    return float("inf")

def speed_label(cond: str) -> str:
    k = speed_key(cond)
    return "?" if not np.isfinite(k) else f"{k:g} m/s"

SPEEDS = sorted(combined_df["condition"].unique().tolist(), key=speed_key)

# ------------------------
# 4) Collapse sides (within-trial L/R mean), then summarize across trials
# ------------------------
df_means = combined_df[combined_df["stat"] == "mean"].copy()

df_trial_lr_mean = (
    df_means
    .groupby(
        ["condition", "tracker", "participant_code", "trial_name",
         "joint", "component", "percent_gait_cycle"],
        as_index=False
    )
    .agg(trial_mean_angle=("value", "mean"))
)

print(
    df_trial_lr_mean.groupby(["condition","tracker"])["trial_name"]
    .nunique()
    .unstack(fill_value=0)
)

angle_summary = (
    df_trial_lr_mean
    .groupby(["condition", "tracker", "joint", "component", "percent_gait_cycle"], as_index=False)
    .agg(
        mean_angle=("trial_mean_angle", "mean"),
        std_angle =("trial_mean_angle", "std"),
        n_trials  =("trial_name", "nunique"),
    )
)

id_cols = [
    "condition",
    "joint",
    "component",
    "percent_gait_cycle",
]

wide = (angle_summary.pivot_table(
    index = id_cols,
    columns = "tracker",
    values = "mean_angle",
    aggfunc = "first",
).reset_index()
)
##for checking specific angles/speeds and whatnot
# sub = wide[
#     (wide["condition"] == "speed_2_0") &
#     (wide["joint"] == "ankle") &
#     (wide["component"] == "dorsi_plantar")
# ].sort_values("percent_gait_cycle")

# abs_err = np.abs(sub["mediapipe"] - sub[reference_system])

# print("Mean abs error:", abs_err.mean())
# # print("Max abs error:", abs_err.max())

# wide.columns.name = None
# wide = wide.rename(columns={reference_system: "reference_system"})

# tracker_cols_present = [t for t in TRACKERS if t in wide.columns]

# paired_df = wide.melt(
#     id_vars = id_cols + ["reference_system"],
#     value_vars = tracker_cols_present,
#     var_name = "tracker",
#     value_name = "tracker_value"
# )
# def calculate_rmse(tracker:pd.Series, reference:pd.Series):
#     tracker = tracker.to_numpy()
#     reference = reference.to_numpy()
#     return np.sqrt(np.mean((tracker - reference)**2))


# rmse_table = (
#     paired_df.groupby(["condition", "joint", "component", "tracker"], as_index = False)
#     .apply(lambda g:calculate_rmse(g["tracker_value"], g["reference_system"]))
#     .rename(columns = {None: "rmse"})
# )

# rmse_table = rmse_table.copy()
# rmse_table["speed"] = (
#     rmse_table["condition"]
#     .astype(str)
#     .str.extract(r"speed_(\d+[_\.]\d+|\d+)")[0]
#     .str.replace("_", ".", regex=False)
#     .astype(float)
# )

# # -----------------------------
# # Build slide-ready table
# # -----------------------------
# def joint_rmse_slide_table(rmse_df: pd.DataFrame, joint_name: str) -> pd.DataFrame:
#     out = (
#         rmse_df[rmse_df["joint"] == joint_name]
#         .pivot_table(
#             index="tracker",
#             columns="speed",
#             values="rmse",
#             aggfunc="first",
#         )
#         .sort_index(axis=1)  # sort speeds
#     )

#     # Presentation polish
#     out = out.round(1)
#     out.index = out.index.str.capitalize()
#     out.columns = [f"{c:g}" for c in out.columns]  # 0.5, 1.0, etc.

#     # Insert the Speed (m/s) header as the first column label
#     out.insert(0, "Speed (m/s)", out.index)
#     out = out.set_index("Speed (m/s)")

#     out['mean'] = out.T.agg('mean')
#     out['std'] = out.T.agg('std')

#     return out

# # -----------------------------
# # Generate tables
# # -----------------------------
# hip_table   = joint_rmse_slide_table(rmse_table, "hip")
# knee_table  = joint_rmse_slide_table(rmse_table, "knee")
# ankle_table = joint_rmse_slide_table(rmse_table, "ankle")

# print("Hip RMSE (°)")
# print(hip_table)
# print("\nKnee RMSE (°)")
# print(knee_table)
# print("\nAnkle RMSE (°)")
# print(ankle_table)

# # -----------------------------
# # Export CSVs (slide-ready)
# # -----------------------------
# hip_table.to_csv(root_dir / "hip_rmse_table.csv")
# knee_table.to_csv(root_dir / "knee_rmse_table.csv")
# ankle_table.to_csv(root_dir / "ankle_rmse_table.csv")
# # ------------------------
# # 5) Publication-ready styling
# # ------------------------

# Subplot size (in inches) - adjust these to control individual panel size
SUBPLOT_WIDTH_IN = 1.5
SUBPLOT_HEIGHT_IN = 1.5
DPI = 100

# Margins (in inches)
MARGIN_LEFT_IN = 1.5
MARGIN_RIGHT_IN = 0.2
MARGIN_TOP_IN = 0.6
MARGIN_BOTTOM_IN = 0.7

LINE_WIDTH = 2
SD_OPACITY = 0.12

# Reference system: black solid
# Comparison systems: colorblind-safe with distinct dash patterns
TRACKER_STYLE = {
    "qualisys": {
        "name": "Qualisys",
        "color": "#313131",  # lighter gray
        "dash": "solid",
        "width": 1.5,        # thinner than comparison systems
        "fill_opacity": 0.3,  # subtler SD band
        "line_opacity": 0.5,
    },
    "mediapipe": {
        "name": "MediaPipe",
        "color": "#0072B2",  # blue (colorblind safe)
        "dash": "solid",
        "width": LINE_WIDTH,
        "fill_opacity": SD_OPACITY,
        "line_opacity": 0.7,
    },
    "rtmpose": {
        "name": "RTMPose",
        "color": "#D55E00",  # vermillion (colorblind safe)
        "dash": "solid",
        "width": LINE_WIDTH,
        "fill_opacity": SD_OPACITY,
        "line_opacity": 0.7,
    },
    "vitpose": {
        "name": "ViTPose",
        "color": "#009E73",  # bluish green (colorblind safe)
        "dash": "dashdot",
        "width": LINE_WIDTH,
        "fill_opacity": SD_OPACITY,
        "line_opacity": 0.7,
    },
}

# Draw order: comparison systems first, reference on top
DRAW_ORDER = ["qualisys", "mediapipe", "rtmpose", "vitpose"]

def rgba(hex_color, alpha):
    """Convert hex to rgba string."""
    h = hex_color.lstrip("#")
    r, g, b = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"

V_SPACING = 0.08
H_SPACING = 0.015

# ------------------------
# 6) Figure grid
# ------------------------
n_rows = len(JOINT_ORDER)
n_cols = len(SPEEDS)

# Calculate total figure size from subplot size + margins
FIG_WIDTH_IN = MARGIN_LEFT_IN + (n_cols * SUBPLOT_WIDTH_IN) + MARGIN_RIGHT_IN
FIG_HEIGHT_IN = MARGIN_TOP_IN + (n_rows * SUBPLOT_HEIGHT_IN) + MARGIN_BOTTOM_IN

FIG_WIDTH_PX = int(FIG_WIDTH_IN * DPI)
FIG_HEIGHT_PX = int(FIG_HEIGHT_IN * DPI)

# Convert margins to pixels
MARGIN_LEFT_PX = int(MARGIN_LEFT_IN * DPI)
MARGIN_RIGHT_PX = int(MARGIN_RIGHT_IN * DPI)
MARGIN_TOP_PX = int(MARGIN_TOP_IN * DPI)
MARGIN_BOTTOM_PX = int(MARGIN_BOTTOM_IN * DPI)

fig = make_subplots(
    rows=n_rows,
    cols=n_cols,
    shared_xaxes=True,
    shared_yaxes=False,
    vertical_spacing=V_SPACING,
    horizontal_spacing=H_SPACING,
    column_titles=[speed_label(s) for s in SPEEDS],
)

y_minmax = {j: [np.inf, -np.inf] for j in JOINT_ORDER}

# ------------------------
# 7) Traces
# ------------------------
for cond_idx, cond in enumerate(SPEEDS, start=1):
    for joint in JOINT_ORDER:
        component = "flex_ext" if joint in ("hip", "knee") else "dorsi_plantar"
        row = JOINT_ORDER.index(joint) + 1
        col = cond_idx

        for tracker in DRAW_ORDER:
            if tracker not in TRACKERS:
                continue
            style = TRACKER_STYLE[tracker]
            
            sub = angle_summary[
                (angle_summary["condition"] == cond) &
                (angle_summary["tracker"] == tracker) &
                (angle_summary["joint"] == joint) &
                (angle_summary["component"] == component)
            ]
            if sub.empty:
                continue

            sub = sub.sort_values("percent_gait_cycle")
            x = sub["percent_gait_cycle"].to_numpy()
            mean = sub["mean_angle"].to_numpy()
            sd = sub["std_angle"].to_numpy()

            if np.all(np.isnan(sd)):
                sd = np.zeros_like(mean)

            lower, upper = mean - sd, mean + sd

            y_minmax[joint][0] = min(y_minmax[joint][0], np.nanmin(lower))
            y_minmax[joint][1] = max(y_minmax[joint][1], np.nanmax(upper))

            fill_color = rgba(style["color"], style["fill_opacity"])
            line_color = rgba(style["color"], style.get("line_opacity", 1.0))


            # SD ribbon
            fig.add_trace(
                go.Scatter(
                    x=x, y=upper, mode="lines",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=tracker,
                ),
                row=row, col=col
            )
            fig.add_trace(
                go.Scatter(
                    x=x, y=lower, mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=fill_color,
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=tracker,
                ),
                row=row, col=col
            )

            # Mean line
            showleg = (row == 1 and col == 1)
            fig.add_trace(
                go.Scatter(
                    x=x, y=mean, mode="lines",
                    line=dict(
                        color=line_color,
                        width=style.get("width", LINE_WIDTH),
                        dash=style["dash"],
                    ),
                    name=style["name"],
                    legendgroup=tracker,
                    showlegend=showleg,
                    hovertemplate=(
                        f"{style['name']}<br>"
                        "Angle: %{y:.1f}°<br>"
                        "Gait cycle: %{x:.0f}%<extra></extra>"
                    ),
                ),
                row=row, col=col
            )

# ------------------------
# 8) Sync y ranges per joint
# ------------------------
for joint in JOINT_ORDER:
    lo, hi = y_minmax[joint]
    if not (np.isfinite(lo) and np.isfinite(hi)):
        lo, hi = -30, 30
    pad = (hi - lo) * 0.08
    rng = [lo - pad, hi + pad]

    r = JOINT_ORDER.index(joint) + 1
    for c in range(1, n_cols + 1):
        fig.update_yaxes(range=rng, row=r, col=c)

# ------------------------
# 9) Row labels and axis formatting
# ------------------------
for joint in JOINT_ORDER:
    component = "flex_ext" if joint in ("hip", "knee") else "dorsi_plantar"
    label = COMP_LABEL[component]
    r = JOINT_ORDER.index(joint) + 1

    fig.add_annotation(
        x=-0.045, xref="paper",
        y=1 - (r - 0.5) / n_rows, yref="paper",
        text=f"<b>{joint.title()}</b><br>{label} (°)",
        showarrow=False,
        xanchor="right",
        font=dict(size=12, color="#333"),
        align="right",
    )

tickvals = list(range(0, 101, 20))
for r in range(1, n_rows + 1):
    for c in range(1, n_cols + 1):
        # X-axis: title only on bottom row
        fig.update_xaxes(
            title_text="<b>Gait cycle (%)</b>" if r == n_rows else None,
            title_font=dict(size=12, color="#333"),
            title_standoff=5,
            tickvals=tickvals,
            tickfont=dict(size=9),
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="#333",
            mirror=True,  # Box outline
            row=r, col=c
        )
        # Y-axis with box outline
        fig.update_yaxes(
            showticklabels=(c == 1),
            tickfont=dict(size=9),
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="#333",
            mirror=True,  # Box outline
            row=r, col=c
        )

# ------------------------
# 10) Layout
# ------------------------
fig.update_layout(
    template="plotly_white",
    height=FIG_HEIGHT_PX,
    width=FIG_WIDTH_PX,
    margin=dict(l=MARGIN_LEFT_PX, r=MARGIN_RIGHT_PX, t=MARGIN_TOP_PX, b=MARGIN_BOTTOM_PX),
    title=dict(
        text="<b>Sagittal Plane Joint Angles Across Treadmill Speeds</b>",
        font=dict(size=14),
        y=0.97, x=0.5, xanchor="center", yanchor="top",
    ),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.12,
        xanchor="center",
        x=0.5,
        font=dict(size=11),
        itemwidth=40,
    ),
    paper_bgcolor="white",
    plot_bgcolor="white",
)

# Style column titles (speed labels)
for annotation in fig.layout.annotations:
    if "m/s" in annotation.text:
        annotation.text = f"<b>{annotation.text}</b>"
        annotation.font = dict(size=11, color="#333")

fig.show()




# Save the exact summary your plot uses (mean ± SD across trials)
angle_summary.to_csv(root_dir / "joint_angles_summary.csv", index=False)
fig.write_image(root_dir / "joint_angles_by_speed.png", scale=3)
# # Optional: also save the trial-level waveforms used to compute it (nice for debugging / SPM reuse)
# df_trial_lr_mean.to_csv(root_dir / "joint_angles_trial_lr_mean.csv", index=False)

print("Saved:",
      root_dir / "joint_angles_summary.csv")
# Export at higher DPI for publication (scale=3 gives 300 DPI equivalent)
# fig.write_image("joint_angles_by_speed.png", scale=300/DPI)
# fig.write_image("joint_angles_by_speed.pdf")  # Vector for publication