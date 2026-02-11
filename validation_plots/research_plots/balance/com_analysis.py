import pandas as pd
import sqlite3
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pathlib import Path

conn = sqlite3.connect("validation.db")

root_path = Path(r"D:\validation\balance")
root_path.mkdir(exist_ok=True, parents=True)

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
    AND a.tracker IN ("mediapipe", "qualisys", "rtmpose")
    AND a.file_exists = 1
    AND a.component_name LIKE '%path_length_com'
ORDER BY t.trial_name, a.path;
"""
path_df = pd.read_sql_query(query, conn)

dfs = []
for _, row in path_df.iterrows():
    path = row["path"]
    tracker = row["tracker"]
    participant = row["participant_code"]
    trial = row["trial_name"]

    sub_df = pd.read_json(path)
    sub_df = sub_df.rename(columns={
        "Frame Intervals": "frame_interval",
        "Path Lengths:": "path_length"
    }).reset_index()
    sub_df = sub_df.rename(columns={"index": "condition"})

    sub_df["participant_code"] = participant
    sub_df["trial_name"] = trial
    sub_df["tracker"] = tracker
    dfs.append(sub_df)

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

# Order: qualisys, mediapipe, rtmpose
TRACKERS = ["qualisys", "mediapipe", "rtmpose"]

colors = {
    "mediapipe": "#4774EE",
    "qualisys": "#747474",
    "rtmpose":  "#EB7303",  # pick any you like; change if you have a canonical one
}

sub_title = {
    "mediapipe": "Freemocap-MediaPipe",
    "qualisys": "Qualisys",
    "rtmpose":  "Freemocap-RTMPose",
}

col_for = {trk: i + 1 for i, trk in enumerate(TRACKERS)}

fig = make_subplots(
    rows=1, cols=len(TRACKERS), shared_yaxes=True,
    subplot_titles=tuple(sub_title[t] for t in TRACKERS)
)

# Individual trial lines
for tracker in TRACKERS:
    dft = df[df["tracker"] == tracker].copy()
    if dft.empty:
        continue

    dft["trial_id"] = dft["participant_code"] + " | " + dft["trial_name"]

    for trial_id, sub in dft.groupby("trial_id", sort=False):
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
                hovertemplate=f"{trial_id}<br>%{{x}}<br>Path length: %{{y:.3f}}<extra></extra>",
                opacity=0.5,
            ),
            row=1, col=col_for[tracker],
        )

# Mean ± SD overlay
agg = (
    df.groupby(["tracker", "condition"])["path_length"]
      .agg(["mean", "std"])
)

for tracker in TRACKERS:
    if tracker not in agg.index.get_level_values(0):
        continue

    sub = agg.loc[tracker].reindex(condition_order)  # <- key change
    means = sub["mean"].to_numpy()
    stds  = sub["std"].to_numpy()

    fig.add_trace(
        go.Scatter(
            x=display_x,
            y=means,
            mode="lines+markers",
            line=dict(color="black"),
            name="Mean",
            showlegend=(tracker == TRACKERS[0]),
            error_y=dict(type="data", array=stds, visible=True),
            hovertemplate="%{x}<br>Mean: %{y:.3f}<br>SD: %{customdata:.3f}<extra></extra>",
            customdata=stds,
        ),
        row=1, col=col_for[tracker],
    )

fig.update_layout(
    height=600,
    width=1050,
    title_text="Center-of-mass path length across balance conditions (Qualisys vs. FreeMoCap trackers)",
    template="plotly_white",
    margin=dict(l=70, r=20, t=60, b=60),
    legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
)

fig.update_yaxes(title_text="Path Length (mm)", row=1, col=1)
for c in range(1, len(TRACKERS) + 1):
    fig.update_xaxes(title_text="Condition", row=1, col=c)

fig.show()

fig.write_image(root_path / "com_path_length.png", scale=3)

# fig.write_html(
#     "docs/balance_data/path_length_line_plots_qualisys_mediapipe_rtmpose.html",
#     full_html=False,
#     include_plotlyjs="cdn",
# )
