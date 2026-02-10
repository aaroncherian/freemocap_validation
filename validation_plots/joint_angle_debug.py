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


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc

# ── Config ──
SPEED_COND = "speed_2_5"
TRACKERS_DBG = ["qualisys", "rtmpose"]
SIDES_DBG = ["left", "right"]
JOINTS_DBG = [("hip", "flex_ext"), ("knee", "flex_ext"), ("ankle", "dorsi_plantar")]
COMP_LABEL_DBG = {"flex_ext": "Flex/Ext", "dorsi_plantar": "Dorsi/Plantar"}

# Row layout: tracker × side
ROW_DEFS = [(t, s) for t in TRACKERS_DBG for s in SIDES_DBG]  # 4 rows

# ── Filter df_means to speed 2.0 and relevant trackers/joints ──
joint_set = set(JOINTS_DBG)
mask = (
    (df_means["condition"] == SPEED_COND)
    & (df_means["tracker"].isin(TRACKERS_DBG))
    & (df_means["side"].isin(SIDES_DBG))
    & (df_means.apply(lambda r: (r["joint"], r["component"]) in joint_set, axis=1))
)
dbg = df_means[mask].copy()

# ── Participant colors ──
participants = sorted(dbg["participant_code"].unique())
palette = pc.qualitative.D3 if len(participants) <= 10 else pc.qualitative.Light24
colors = {p: palette[i % len(palette)] for i, p in enumerate(participants)}

# ── Build figure: 4 rows × 3 cols ──
fig = make_subplots(
    rows=4, cols=3,
    shared_xaxes=True,
    vertical_spacing=0.06,
    horizontal_spacing=0.06,
    row_titles=[f"{t.capitalize()} {s.capitalize()}" for t, s in ROW_DEFS],
    column_titles=[f"{j.title()} {COMP_LABEL_DBG[c]}" for j, c in JOINTS_DBG],
)

for row_idx, (tracker, side) in enumerate(ROW_DEFS, start=1):
    for col_idx, (joint, comp) in enumerate(JOINTS_DBG, start=1):
        sub = dbg[
            (dbg["tracker"] == tracker)
            & (dbg["side"] == side)
            & (dbg["joint"] == joint)
            & (dbg["component"] == comp)
        ]

        # Each unique (participant, trial) gets its own trace
        for (pid, trial), tsub in sub.groupby(["participant_code", "trial_name"]):
            tsub = tsub.sort_values("percent_gait_cycle")

            fig.add_trace(
                go.Scatter(
                    x=tsub["percent_gait_cycle"],
                    y=tsub["value"],
                    mode="lines",
                    name=pid,
                    legendgroup=pid,
                    showlegend=(row_idx == 1 and col_idx == 1
                                and trial == sub[sub["participant_code"] == pid]["trial_name"].iloc[0]),
                    line=dict(color=colors[pid], width=1.2),
                    hovertemplate=(
                        f"{pid} | {trial}<br>"
                        "%{y:.1f}° @ %{x:.0f}%<extra></extra>"
                    ),
                ),
                row=row_idx, col=col_idx,
            )

# ── Formatting ──
fig.update_xaxes(title_text="Gait cycle (%)", row=4)
fig.update_layout(
    height=800,
    width=1100,
    title=f"<b>Debug: Per-trial L/R traces at {SPEED_COND.replace('_', ' ')}</b>",
    template="plotly_white",
    legend=dict(
        orientation="h", yanchor="top", y=-0.08,
        xanchor="center", x=0.5, font=dict(size=10),
    ),
)

fig.show()