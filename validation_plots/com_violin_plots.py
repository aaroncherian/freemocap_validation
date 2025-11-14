import pandas as pd
import sqlite3
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go

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
    condition = row.get("condition") or ""  # handle None/empty
    participant = row["participant_code"]
    trial = row["trial_name"]

    sub_df = pd.read_csv(path)

    # Add participant, trial, condition, and tracker info
    sub_df["participant_code"] = participant
    sub_df["trial_name"] = trial
    sub_df["condition"] = condition
    sub_df["tracker"] = tracker

    dfs.append(sub_df)

# Concatenate all dataframes
final_df = pd.concat(dfs, ignore_index=True)
id_cols = ["participant_code", "trial_name", "Frame", "tracker"]

# all the condition+axis columns
value_cols = [c for c in final_df.columns 
              if "Eyes" in c or "Ground" in c or "Foam" in c]

long_df = final_df.melt(
    id_vars=id_cols,
    value_vars=value_cols,
    var_name="cond_axis",
    value_name="velocity",
)

# split "Eyes Open/Solid Ground_x" → condition="Eyes Open/Solid Ground", axis="x"
long_df[["condition", "axis"]] = long_df["cond_axis"].str.rsplit("_", n=1, expand=True)

# optional: drop NaNs (frames outside that condition)
long_df = long_df.dropna(subset=["velocity"])
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Color mapping (FreeMoCap = mediapipe)
colors = {
    "mediapipe": "#014E9C",   # freemocap blue
    "qualisys":  "#BE4302",   # qualisys orange
}

# Nice condition order (optional but helps the layout)
condition_order = [
    "Eyes Open/Solid Ground",
    "Eyes Closed/Solid Ground",
    "Eyes Open/Foam",
    "Eyes Closed/Foam",
]

# Make one figure per axis (X, Y, Z)
for axis in ["x", "y", "z"]:
    df_axis = long_df[long_df["axis"] == axis].copy()

    # Enforce condition order
    df_axis["condition"] = pd.Categorical(
        df_axis["condition"],
        categories=condition_order,
        ordered=True,
    )

    fig = make_subplots(rows=1, cols=1)

    # ---- FreeMoCap (mediapipe) on the left (negative side) ----
    df_fmc = df_axis[df_axis["tracker"] == "mediapipe"]
    fig.add_trace(
        go.Violin(
            x=df_fmc["condition"],
            y=df_fmc["velocity"],
            legendgroup="freemocap",
            scalegroup="freemocap",
            name="freemocap",
            side="negative",
            line_color=colors["mediapipe"],
            width=0.8,            
        )
    )

    # ---- Qualisys on the right (positive side) ----
    df_qs = df_axis[df_axis["tracker"] == "qualisys"]
    fig.add_trace(
        go.Violin(
            x=df_qs["condition"],
            y=df_qs["velocity"],
            legendgroup="qualisys",
            scalegroup="qualisys",
            name="qualisys",
            side="positive",
            line_color=colors["qualisys"],
            width=0.8,           
        )
    )

    fig.update_traces(
        box_visible=True,
        meanline_visible=True,
        points=False,      # no raw dots
        scalemode="count", # area ~ number of samples
        opacity=0.9,
    )

    fig.update_layout(
        template="plotly_white",
        title=f"COM {axis.upper()} Velocity vs. Condition",
        xaxis_title="",
        yaxis_title="COM velocity",
        legend_title="",
        violingap=0,
        violinmode="overlay",
        width=1000,
        height=500,
    )

    fig.write_html(f"docs/balance_data/com_velocity_violin_{axis}.html", full_html=False, include_plotlyjs='cdn')