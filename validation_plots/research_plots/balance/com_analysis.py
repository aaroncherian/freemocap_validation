import pandas as pd
import sqlite3
from pathlib import Path
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pathlib import Path

conn = sqlite3.connect("validation.db")

# root_path = Path(r"D:\validation\balance")

root_path = Path(r"C:\Users\aaron\Documents\GitHub\dissertation\neu_coe_typst_starter\chapters\balance\figures")
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
    AND a.tracker IN ("mediapipe", "qualisys", "rtmpose" , "vitpose")
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

# Order: q
TRACKERS = ["qualisys", "mediapipe", "rtmpose", "vitpose"]

# colors = {
#     "mediapipe": "#4774EE",
#     "qualisys": "#747474",
#     "rtmpose":  "#EB7303",  
#     "vitpose":  "#006D43"
# }

colors = {
    "mediapipe": "#747474",
    "qualisys": "#747474",
    "rtmpose":  "#747474",  
    "vitpose":  "#747474"
}

sub_title = {
    "mediapipe": "FMC-MediaPipe",
    "qualisys": "Reference",
    "rtmpose":  "FMC-RTMPose",
    "vitpose":  "FMC-ViTPose",
}

col_for = {trk: i + 1 for i, trk in enumerate(TRACKERS)}

display_x_short = ["EO-S", "EC-S", "EO-F", "EC-F"]

fig = make_subplots(
    rows=1, cols=len(TRACKERS), shared_yaxes=True,
    subplot_titles=tuple(sub_title[t] for t in TRACKERS),
    horizontal_spacing=0.05,  # more room between panels
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
                x=display_x_short,
                y=s.values,
                mode="lines+markers",
                line=dict(color="rgba(150,150,150,0.4)", width=2),
                marker=dict(size=5, color="rgba(150,150,150,0.5)"),
                showlegend=False,
                hovertemplate=f"{trial_id}<br>%{{x}}<br>Path length: %{{y:.3f}}<extra></extra>",
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

    sub = agg.loc[tracker].reindex(condition_order)
    means = sub["mean"].to_numpy()
    stds  = sub["std"].to_numpy()

    fig.add_trace(
        go.Scatter(
            x=display_x_short,
            y=means,
            mode="lines+markers",
            line=dict(color="black", width=2.5),
            marker=dict(color="black", size=7),
            showlegend=False,  # describe in caption instead
            error_y=dict(
                type="data", array=stds, visible=True,
                thickness=2.5, width=4,
            ),
            hovertemplate="%{x}<br>Mean: %{y:.3f}<br>SD: %{customdata:.3f}<extra></extra>",
            customdata=stds,
        ),
        row=1, col=col_for[tracker],
    )

fig.update_layout(
    height=500,           # taller for better aspect ratio
    width=1200,
    template="simple_white",
    margin=dict(l=80, r=20, t=30, b=70),
    font=dict(
        family="Arial",
        size=14,
    ),
)

# Subplot titles — bump up size
for ann in fig.layout.annotations:
    ann.update(
        font=dict(family="Arial", size=28),
        xanchor="center",
    )

fig.update_yaxes(
    title_text="<b>Path Length (mm)</b>",
    title_font=dict(size=28),
    tickfont=dict(size=22),
    row=1, col=1,
)


for c in range(1, len(TRACKERS) + 1):
    fig.update_xaxes(
        tickfont=dict(size=22),
        title_text="",  # drop redundant "Condition" labels
        row=1, col=c,
    )

fig.show()

fig.write_image(root_path / "com_path_length.svg", scale=3)

# fig.write_image(root_path / "com_path_length.png", scale=3)

# fig.write_image(root_path / "com_path_length.pdf")


