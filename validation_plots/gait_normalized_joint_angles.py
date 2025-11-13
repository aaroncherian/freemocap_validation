import pandas as pd
import sqlite3
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------
# 1) Load data from SQLite
# ------------------------
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
  AND a.tracker IN ("mediapipe", "qualisys")
  AND a.file_exists = 1
  AND a.condition = "speed_0_5"
  AND a.component_name LIKE "%summary_stats"
ORDER BY t.trial_name, a.path
"""
path_df = pd.read_sql_query(query, conn)

dfs = []
for _, row in path_df.iterrows():
    sub = pd.read_csv(row["path"])
    sub["participant_code"] = row["participant_code"]
    sub["trial_name"] = row["trial_name"]
    sub["tracker"] = row["tracker"].lower()
    sub["condition"] = row["condition"] or "none"
    dfs.append(sub)

combined_df = pd.concat(dfs, ignore_index=True)

# ------------------------
# 2) Config / normalization
# ------------------------
JOINTS = ["HIP", "KNEE", "ANKLE"]
SIDES = ["left", "right"]

COMPONENTS_BY_JOINT_DEFAULT = {
    "hip":   ["flex_ext","abd_add","int_ext"],
    "knee":  ["flex_ext","abd_add","int_ext"],
    "ankle": ["dorsi_plantar","inv_ev","int_ext"],
}
COMP_LABEL = {
    "flex_ext":      "Flex / Ext (°)",
    "abd_add":       "Abd / Add (°)",
    "int_ext":       "Int / Ext Rot (°)",
    "dorsi_plantar": "Dorsi / Plantar (°)",
    "inv_ev":        "Inversion / Eversion (°)",
}

# normalize naming
combined_df["component"] = (
    combined_df["component"].str.lower().replace({"inversion_eversion": "inv_ev"})
)
for col in ["joint", "side", "tracker", "stat"]:
    if col in combined_df.columns:
        combined_df[col] = combined_df[col].str.lower()

# ------------------------
# 3) Aggregate: mean across trials (per tracker)
#    (std here is the SD across trial means — between-trial spread)
# ------------------------
df_means = combined_df[combined_df["stat"] == "mean"].copy()
angle_summary = (
    df_means.groupby(["tracker","joint","side","component","percent_gait_cycle"], as_index=False)
            .agg(mean_angle=("value","mean"),
                 std_angle =("value","std"),
                 n_trials  =("trial_name","nunique"))
)

# ------------------------
# 4) Styling
# ------------------------
LINE_WIDTH   = 2.5
TRACKER_LINE = {"mediapipe": "#1f77b4", "qualisys": "#d62728"}                 # FM blue, Q red
TRACKER_FILL = {"mediapipe": "rgba(31,119,180,0.22)", "qualisys": "rgba(214,39,40,0.22)"}  # soft ribbons

# layout spacing
V_SPACING = 0.05
H_SPACING = 0.08

# ------------------------
# 5) Figure grid
# ------------------------
n_rows = len(JOINTS) * 2
n_cols = 3
fig = make_subplots(
    rows=n_rows, cols=n_cols,
    shared_xaxes=True,
    vertical_spacing=V_SPACING,
    horizontal_spacing=H_SPACING,
)

def ax_key(kind, row, col):
    idx = (row - 1) * n_cols + col
    return f"{kind}axis" + ("" if idx == 1 else str(idx))

def get_row_col(joint, side, comp_idx):
    j_idx = JOINTS.index(joint.upper())
    row = j_idx * 2 + (1 if side == "left" else 2)
    col = comp_idx
    return row, col

# ------------------------
# 6) Traces + y-range sync
# ------------------------
y_minmax = {}
for j in JOINTS:
    for c_idx, comp in enumerate(COMPONENTS_BY_JOINT_DEFAULT[j.lower()], start=1):
        y_minmax[(j, c_idx)] = [np.inf, -np.inf]

for j in JOINTS:
    for side in SIDES:
        comps = COMPONENTS_BY_JOINT_DEFAULT[j.lower()]
        for c_idx, comp in enumerate(comps, start=1):
            for tracker in ["mediapipe", "qualisys"]:
                sub = angle_summary[
                    (angle_summary["tracker"] == tracker) &
                    (angle_summary["joint"]   == j.lower()) &
                    (angle_summary["side"]    == side) &
                    (angle_summary["component"] == comp)
                ]
                if sub.empty:
                    continue
                sub = sub.sort_values("percent_gait_cycle")
                x = sub["percent_gait_cycle"].to_numpy()
                mean = sub["mean_angle"].to_numpy()
                sd   = sub["std_angle"].to_numpy()
                lower, upper = mean - sd, mean + sd
                lo, hi = np.nanmin(lower), np.nanmax(upper)
                y_minmax[(j, c_idx)][0] = min(y_minmax[(j, c_idx)][0], lo)
                y_minmax[(j, c_idx)][1] = max(y_minmax[(j, c_idx)][1], hi)
                row, col = get_row_col(j, side, c_idx)

                # ribbon (upper -> lower) with transparent fill
                fig.add_trace(
                    go.Scatter(x=x, y=upper, mode="lines",
                               line=dict(width=0, color="rgba(0,0,0,0)"),
                               hoverinfo="skip", showlegend=False),
                    row=row, col=col
                )
                fig.add_trace(
                    go.Scatter(x=x, y=lower, mode="lines",
                               line=dict(width=0, color="rgba(0,0,0,0)"),
                               fill="tonexty", fillcolor=TRACKER_FILL[tracker],
                               hoverinfo="skip", showlegend=False),
                    row=row, col=col
                )
                # mean line
                showleg = (row == 1 and col == 1)
                fig.add_trace(
                    go.Scatter(x=x, y=mean, mode="lines",
                               line=dict(color=TRACKER_LINE[tracker], width=LINE_WIDTH),
                               name="FreeMoCap" if tracker == "mediapipe" else "Qualisys",
                               legendgroup=tracker, showlegend=showleg,
                               hovertemplate="Angle: %{y:.1f}°<br>Gait: %{x:.0f}%<extra></extra>"),
                    row=row, col=col
                )

# sync y ranges left/right per joint/component
for j in JOINTS:
    for c_idx in range(1, 4):
        lo, hi = y_minmax.get((j, c_idx), [-30, 30])
        if np.isfinite(lo) and np.isfinite(hi):
            pad = (hi - lo) * 0.10 if hi > lo else 5
            rng = [lo - pad, hi + pad]
            fig.update_yaxes(range=rng, row=JOINTS.index(j) * 2 + 1, col=c_idx)
            fig.update_yaxes(range=rng, row=JOINTS.index(j) * 2 + 2, col=c_idx)

# ------------------------
# 7) Titles per joint & per-joint component headers
# ------------------------
for j_idx, joint in enumerate(JOINTS):
    top_row = j_idx * 2 + 1
    # joint block title
    y_top = fig.layout[ax_key("y", top_row, 1)].domain[1] + 0.05
    fig.add_annotation(x=0.5, y=y_top, xref="paper", yref="paper",
                       text=f"<b>{joint.title()}</b>",
                       showarrow=False, xanchor="center",
                       font=dict(size=14, color="#000"))
    # per-joint component titles centered above each column of the top row
    comps = COMPONENTS_BY_JOINT_DEFAULT[joint.lower()]
    for c_idx, comp in enumerate(comps, start=1):
        x0, x1 = fig.layout[ax_key("x", top_row, c_idx)].domain
        xmid = 0.5 * (x0 + x1)
        y_hdr = fig.layout[ax_key("y", top_row, c_idx)].domain[1] + 0.015
        fig.add_annotation(x=xmid, y=y_hdr, xref="paper", yref="paper",
                           text=f"<b>{COMP_LABEL.get(comp, comp)}</b>",
                           showarrow=False, xanchor="center",
                           font=dict(size=12))

# ------------------------
# 8) Left/Right on EVERY y-axis
# ------------------------
for r in range(1, n_rows + 1):
    side = "Left" if (r % 2 == 1) else "Right"
    for c in range(1, n_cols + 1):
        fig.update_yaxes(title_text=side, row=r, col=c)

# ------------------------
# 9) Alternating row backgrounds + clean dividers
# ------------------------
# alternating row backgrounds
for r in range(1, n_rows + 1):
    y0, y1 = fig.layout[ax_key("y", r, 1)].domain
    y0 += 0.002; y1 -= 0.002  # small inset so it doesn't touch axis frame
    fill = "rgba(0,0,0,0.03)" if (r % 2 == 1) else "rgba(0,0,0,0.015)"
    fig.add_shape(type="rect", layer="below",
                  xref="paper", yref="paper",
                  x0=0.0, x1=1.0, y0=y0, y1=y1,
                  line=dict(width=0), fillcolor=fill)

# dividers between joint blocks (midpoint of the inter-block gap)
for j_idx in range(len(JOINTS) - 1):
    r_bottom = j_idx * 2 + 2
    r_top_next = (j_idx + 1) * 2 + 1
    y_bottom = fig.layout[ax_key("y", r_bottom, 1)].domain[0]
    y_top_next = fig.layout[ax_key("y", r_top_next, 1)].domain[1]
    y_mid = y_bottom + 0.35 * (y_top_next - y_bottom)
    fig.add_shape(type="line", layer="below",
                  xref="paper", yref="paper",
                  x0=0.0, x1=1.0, y0=y_mid, y1=y_mid,
                  line=dict(width=1, color="rgba(0,0,0,0.55)"))

# ------------------------
# 10) Axes, legend, layout
# ------------------------
tickvals = list(range(0, 101, 20))
for c in range(1, n_cols + 1):
    fig.update_xaxes(tickvals=tickvals, row=n_rows, col=c, title_text="Percent gait cycle")

# global y label (kept, even though each axis shows Left/Right)
fig.add_annotation(x=-0.08, xref="paper", y=0.5, yref="paper",
                   text="<b>Angle (°)</b>", textangle=-90, showarrow=False,
                   font=dict(size=13, color="#000"))

fig.update_layout(
    template="plotly_white",
    height=120 * n_rows + 420,
    width=1120,
    margin=dict(l=90, r=40, t=90, b=90),
    title=dict(text="<b>Joint Angles (mean ± SD) per component</b>",
               y=0.98, x=0.5, xanchor="center", yanchor="top"),
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="center", x=0.5)
)

fig.update_xaxes(showgrid=True, zeroline=False)
fig.update_yaxes(showgrid=True, zeroline=False)

fig.show()
