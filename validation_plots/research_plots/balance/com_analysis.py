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
    AND a.component_name LIKE '%path_length_com'
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

    # Load file — autodetect type
    sub_df = pd.read_json(path)
    sub_df = sub_df.rename(columns={"Frame Intervals": "frame_interval",
                                    "Path Lengths:": "path_length"}).reset_index()
    sub_df = sub_df.rename(columns={"index": "condition"})
    # Add metadata columns
    sub_df["participant_code"] = participant
    sub_df["trial_name"] = trial
    sub_df["tracker"] = tracker

    dfs.append(sub_df)

# Concatenate all into one tidy DataFrame
combined_df = pd.concat(dfs, ignore_index=True)


condition_order = [
    "Eyes Open/Solid Ground",
    "Eyes Closed/Solid Ground",
    "Eyes Open/Foam",
    "Eyes Closed/Foam",
]
display_x = [c.replace('/', '<br>') for c in condition_order]


df = combined_df.copy()
df["condition"] = pd.Categorical(df["condition"], categories=condition_order, ordered=True)


# 2) Colors and subplot mapping
colors = {
    "mediapipe": "#7994B0",   # your bluish color
    "qualisys": "#C67548",    # your brown/orange
}
sub_title = {"mediapipe": "Freemocap-MediaPipe", "qualisys": "Qualisys"}
col_for = {"mediapipe": 1, "qualisys": 2}

# 3) Figure
fig = make_subplots(
    rows=1, cols=2, shared_yaxes=True,
    subplot_titles=(sub_title["mediapipe"], sub_title["qualisys"])
)

# 4) Individual trial lines for each tracker
#    one line per (participant_code, trial_name)
for tracker in ["mediapipe", "qualisys"]:
    dft = df[df["tracker"] == tracker]

    # unique trial id
    dft = dft.assign(trial_id=dft["participant_code"] + " | " + dft["trial_name"])
    for trial_id, sub in dft.groupby("trial_id", sort=False):
        # ensure x is in canonical order
        s = (
            sub.set_index("condition")["path_length"]
               .reindex(condition_order)
        )
        fig.add_trace(
            go.Scatter(
                x=display_x,
                y=s.values,
                mode="lines+markers",
                line=dict(color=colors[tracker], width=0.5),
                showlegend=False,
                hovertemplate=f"Path length: %{{y:.3f}}<extra></extra>",
                opacity=0.5,
            ),
            row=1, col=col_for[tracker],
        )

# 5) Mean ± SD overlay per tracker
agg = (
    df.groupby(["tracker", "condition"])["path_length"]
      .agg(["mean", "std"])
      .reindex(pd.MultiIndex.from_product([["mediapipe","qualisys"], condition_order]))
)

for tracker in ["mediapipe", "qualisys"]:
    means = agg.loc[(tracker,), "mean"].values
    stds  = agg.loc[(tracker,), "std"].values
    fig.add_trace(
        go.Scatter(
            x=display_x,
            y=means,
            mode="lines+markers",
            line=dict(color="black"),
            name="Mean",
            error_y=dict(type="data", array=stds, visible=True),
            hovertemplate="%{x}<br>Mean: %{y:.3f}<br>SD: %{customdata:.3f}<extra></extra>",
            customdata=stds,
        ),
        row=1, col=col_for[tracker],
    )

# 6) Layout
fig.update_layout(
    height=600,
    width=700,
    title_text="Center-of-mass path length across balance conditions (FreeMoCap vs. Qualisys)",
    template="plotly_white",
    margin=dict(l=70, r=20, t=60, b=60),
)
fig.update_yaxes(title_text="Path Length (mm)", row=1, col=1)
fig.update_xaxes(title_text="Condition", row=1, col=1)
fig.update_xaxes(title_text="Condition", row=1, col=2)

fig.show()
# Save (or fig.show())
fig.write_html("docs/balance_data/path_length_line_plots.html", full_html=False, include_plotlyjs='cdn')
# print(f"Wrote {out_html}")