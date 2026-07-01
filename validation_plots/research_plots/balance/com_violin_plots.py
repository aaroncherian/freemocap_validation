import pandas as pd
import sqlite3
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# =========================
# Paper-ready figure params
# =========================
DPI = 300
FIG_W_IN = 2         # typical single-column ~3.5", double-column ~7"
FIG_H_IN = 3         # adjust as needed
FIG_W_PX = int(FIG_W_IN * DPI)
FIG_H_PX = int(FIG_H_IN * DPI)

EXPORT_BASENAME = "com_velocity_violin"  # writes PNG + PDF

root_path = Path(r"D:\validation\balance")
root_path.mkdir(exist_ok=True, parents=True)
# -------------------
# Load paths from DB
# -------------------
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
WHERE t.trial_type = "balance"
    AND a.category = "com_analysis"
    AND a.tracker IN ("mediapipe", "qualisys")
    AND a.file_exists = 1
    AND a.component_name LIKE '%balance_velocities'
ORDER BY t.trial_name, a.path;
"""
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
    sub_df["condition"] = condition
    sub_df["tracker"] = tracker
    dfs.append(sub_df)

final_df = pd.concat(dfs, ignore_index=True)

id_cols = ["participant_code", "trial_name", "Frame", "tracker"]

# all the condition+axis columns
value_cols = [c for c in final_df.columns if ("Eyes" in c or "Ground" in c or "Foam" in c)]

long_df = final_df.melt(
    id_vars=id_cols,
    value_vars=value_cols,
    var_name="cond_axis",
    value_name="velocity",
)

# split "Eyes Open/Solid Ground_x" → condition="Eyes Open/Solid Ground", axis="x"
long_df[["condition", "axis"]] = long_df["cond_axis"].str.rsplit("_", n=1, expand=True)

# drop NaNs (frames outside that condition)
long_df = long_df.dropna(subset=["velocity"])

# -------------------
# Plot configuration
# -------------------
colors = {
    "qualisys":  "#7A7A7A",   # neutral reference gray
    "mediapipe": "#014E9C",   # FreeMoCap blue
}

condition_order = [
    "Eyes Open/Solid Ground",
    "Eyes Closed/Solid Ground",
    "Eyes Open/Foam",
    "Eyes Closed/Foam",
]

tickvals = condition_order
ticktext = [
    "Eyes Open<br>Solid Ground",
    "Eyes Closed<br>Solid Ground",
    "Eyes Open<br>Foam",
    "Eyes Closed<br>Foam",
]

legend_labels = [
    "Eyes Open <br<"
]

axis_order = ["x", "y", "z"]
axis_titles = {
    "x": "Mediolateral center-of-mass velocity (X)",
    "y": "Anteroposterior center-of-mass velocity (Y)",
    "z": "Vertical center-of-mass velocity (Z)",
}

# -------------------
# Build combined figure
# -------------------
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.06,
    subplot_titles=[axis_titles[a] for a in axis_order],
)

# enforce condition ordering globally
long_df["condition"] = pd.Categorical(long_df["condition"], categories=condition_order, ordered=True)

for r, axis in enumerate(axis_order, start=1):
    df_axis = long_df[long_df["axis"] == axis].copy()

    # FreeMoCap (mediapipe) negative side
    df_qs = df_axis[df_axis["tracker"] == "qualisys"]
    fig.add_trace(
        go.Violin(
            x=df_qs["condition"],
            y=df_qs["velocity"],
            legendgroup="qualisys",
            scalegroup=f"qualisys_{axis}",
            name="Qualisys",
            side="negative",
            line_color=colors["qualisys"],
            width=0.85,
            showlegend=(r == 1),
            opacity=0.6,   # slightly quieter
        ),
        row=r, col=1
    )

    # ---- FreeMoCap on the RIGHT (system of interest) ----
    df_fmc = df_axis[df_axis["tracker"] == "mediapipe"]
    fig.add_trace(
        go.Violin(
            x=df_fmc["condition"],
            y=df_fmc["velocity"],
            legendgroup="freemocap",
            scalegroup=f"freemocap_{axis}",
            name="FreeMoCap",
            side="positive",
            line_color=colors["mediapipe"],
            width=0.85,
            showlegend=(r == 1),
            opacity=0.8,
        ),
        row=r, col=1
    )
        # Per-panel y label (optional — could also label only the middle one)
    fig.update_yaxes(title_text="COM velocity (mm/s)", row=r, col=1)

# Style all violins
fig.update_traces(
    box_visible=True,
    meanline_visible=True,
    points=False,
    scalemode="count",
    opacity=0.75,           # slightly toned down for print
    meanline=dict(width=2),
)

# Layout: paper-like
fig.update_layout(
    template="simple_white",     # cleaner than plotly_white
    width=FIG_W_PX,
    height=FIG_H_PX,
    margin=dict(l=80, r=20, t=80, b=85),
    font=dict(size=12),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.12,
        xanchor="center",
        x=0.5,
        title_text="",
    ),
)

# Share x tick labels only on bottom row
fig.update_xaxes(showticklabels=False, row=1, col=1)
fig.update_xaxes(showticklabels=False, row=2, col=1)
fig.update_xaxes(title_text="Condition", row=3, col=1)

fig.update_yaxes(range=[-2.4, 2.4], row=1, col=1)
fig.update_yaxes(range=[-2.4, 2.4], row=2, col=1)
fig.update_yaxes(range=[-1.5, 1.5], row=3, col=1)

# Ensure condition order on x-axis
fig.update_xaxes(categoryorder="array", categoryarray=condition_order)
fig.update_xaxes(
    row=3, col=1,
    tickmode="array",
    tickvals=tickvals,
    ticktext=ticktext,
    tickfont=dict(size=12),   # try 11 if still tight
    automargin=True,
)


# Optional: tighten the whitespace between category labels
fig.update_xaxes(tickangle=0)

# fig.show()


# -------------------
# Export at 300 dpi
# -------------------
# pip install -U kaleido
fig.write_image(root_path / f"{EXPORT_BASENAME}.png", width=FIG_W_PX, height=FIG_H_PX, scale=3)
# fig.write_image(f"{EXPORT_BASENAME}.pdf", width=FIG_W_PX, height=FIG_H_PX, scale=1)


# ============================================================
# Additional analysis: 2D resultant COM velocity (x-y plane)
# Keeps original framewise violin workflow intact
# ============================================================

# Use only x and y velocity components
xy_df = long_df[long_df["axis"].isin(["x", "y"])].copy()

# Wide format so each frame has x and y side by side
xy_wide = (
    xy_df.pivot_table(
        index=["participant_code", "trial_name", "Frame", "tracker", "condition"],
        columns="axis",
        values="velocity",
        aggfunc="first",
    )
    .reset_index()
)

# Keep only rows where both x and y exist
xy_wide = xy_wide.dropna(subset=["x", "y"])

# Framewise 2D resultant velocity magnitude
xy_wide["velocity_2d"] = (xy_wide["x"]**2 + xy_wide["y"]**2) ** 0.5


# ------------------------------------------------------------
# Trial-level mean 2D velocity
# ------------------------------------------------------------
trial_mean_velocity_2d_df = (
    xy_wide
    .groupby(
        ["participant_code", "trial_name", "tracker", "condition"],
        as_index=False
    )["velocity_2d"]
    .mean()
    .rename(columns={"velocity_2d": "trial_mean_velocity_2d"})
)

# ------------------------------------------------------------
# Group-level mean 2D velocity (averaged over trials)
# ------------------------------------------------------------
group_mean_velocity_2d_df = (
    trial_mean_velocity_2d_df
    .groupby(["tracker", "condition"], as_index=False)["trial_mean_velocity_2d"]
    .agg(
        group_mean_velocity_2d="mean",
        sd_velocity_2d="std",
        n_trials="count"
    )
)

# Keep your preferred condition order
trial_mean_velocity_2d_df["condition"] = pd.Categorical(
    trial_mean_velocity_2d_df["condition"],
    categories=condition_order,
    ordered=True
)
group_mean_velocity_2d_df["condition"] = pd.Categorical(
    group_mean_velocity_2d_df["condition"],
    categories=condition_order,
    ordered=True
)

trial_mean_velocity_2d_df = trial_mean_velocity_2d_df.sort_values(
    ["condition", "tracker", "trial_name"]
)
group_mean_velocity_2d_df = group_mean_velocity_2d_df.sort_values(
    ["condition", "tracker"]
)

print("\nTrial-level mean 2D COM velocity:")
print(trial_mean_velocity_2d_df)

print("\nGroup-level mean 2D COM velocity:")
print(group_mean_velocity_2d_df)

# Optional pivoted table for easier reading
group_mean_velocity_2d_pivot = group_mean_velocity_2d_df.pivot(
    index="condition",
    columns="tracker",
    values="group_mean_velocity_2d"
)

print("\nGroup-level mean 2D COM velocity (pivoted):")
print(group_mean_velocity_2d_pivot)

# # Save tables
# trial_mean_velocity_2d_df.to_csv(
#     root_path / "trial_mean_velocity_2d.csv",
#     index=False
# )
# group_mean_velocity_2d_df.to_csv(
#     root_path / "group_mean_velocity_2d.csv",
#     index=False
# )
# group_mean_velocity_2d_pivot.to_csv(
#     root_path / "group_mean_velocity_2d_pivot.csv"
# )