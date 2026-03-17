from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


CSV_PATH = Path(r"D:\validation\scaling_factors_by_trial_participant.csv")
save_root = Path(r"C:\Users\aaron\Documents\GitHub\dissertation\neu_coe_typst_starter\chapters\gait\figures")
save_root.mkdir(exist_ok=True, parents=True)

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows from {CSV_PATH}")
print(df)

tracker_colors = {
    "mediapipe": "#054f85",
    "rtmpose": "#d36600",
    "vitpose": "#029131",
}

tracker_display_names = {
    "rtmpose": "RTMPose",
    "mediapipe": "MediaPipe",
    "vitpose": "ViTPose",
}

tracker_symbols = {
    "rtmpose": "circle",
    "mediapipe": "square",
    "vitpose": "diamond",
}

participant_order = sorted(df["participant"].unique())
tracker_order = ["rtmpose", "mediapipe", "vitpose"]

# ---- Summaries ----
participant_summary = (
    df.groupby(["tracker", "participant"], as_index=False)["scaling_factor"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .reset_index()
)

overall_summary = (
    df.groupby("tracker", as_index=False)["scaling_factor"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .reset_index()
)

print("\nParticipant summary:")
print(participant_summary)

print("\nOverall summary:")
print(overall_summary)

# ---- Figure ----
fig = make_subplots(
    rows=2,
    cols=1,
    shared_yaxes=True,
    vertical_spacing=0.12,
    row_heights=[0.65, 0.35],
    subplot_titles=(
        "Participant-level scaling factors",
        "Overall scaling factor distribution by tracker",
    ),
)

# Top panel: participant-level
for tracker in tracker_order:
    tracker_df = df[df["tracker"] == tracker]

    fig.add_trace(
        go.Box(
            x=tracker_df["participant"],
            y=tracker_df["scaling_factor"],
            name=tracker_display_names[tracker],
            boxpoints="all",
            pointpos=0,
            jitter=0.25,
            marker_symbol=tracker_symbols[tracker],
            marker_color=tracker_colors[tracker],
            line_color=tracker_colors[tracker],
            opacity=0.6,
            legendgroup=tracker,
            showlegend=True,
        ),
        row=1,
        col=1,
    )

# Bottom panel: overall distributions
for tracker in tracker_order:
    tracker_df = df[df["tracker"] == tracker]

    fig.add_trace(
        go.Box(
            x=[tracker_display_names[tracker]] * len(tracker_df),
            y=tracker_df["scaling_factor"],
            name=tracker_display_names[tracker],
            boxpoints="all",
            pointpos=0,
            jitter=0.2,
            marker_symbol=tracker_symbols[tracker],
            opacity=0.75,
            legendgroup=f"overall_{tracker}",
            marker_color=tracker_colors[tracker],
            line_color=tracker_colors[tracker],
            showlegend=False,
        ),
        row=2,
        col=1,
    )

fig.add_hline(y=1.0, line_dash="dash", row=1, col=1)
fig.add_hline(y=1.0, line_dash="dash", row=2, col=1)

fig.update_xaxes(categoryorder="array", categoryarray=participant_order, row=1, col=1)
fig.update_yaxes(title_text="Scaling factor", row=1, col=1)
fig.update_yaxes(title_text="Scaling factor", row=2, col=1)
fig.update_xaxes(title_text="Participant", row=1, col=1)
fig.update_xaxes(title_text="Tracker", row=2, col=1)

fig.update_layout(
    title="Scaling factor comparison across trackers",
    height=900,
    width=1100,
    template="plotly_white",
)

fig.show()

# ---- Standalone publication figure: overall scaling factor distribution ----
fig_pub = go.Figure()

for tracker in tracker_order:
    tracker_df = df[df["tracker"] == tracker]

    fig_pub.add_trace(
        go.Box(
            x=[tracker_display_names[tracker]] * len(tracker_df),
            y=tracker_df["scaling_factor"],
            name=tracker_display_names[tracker],
            boxpoints="all",
            pointpos=0,
            jitter=0.3,
            marker=dict(
                symbol=tracker_symbols[tracker],
                size=7,
                opacity=0.6,
                color=tracker_colors[tracker],
            ),
            line=dict(color=tracker_colors[tracker]),
            marker_color=tracker_colors[tracker],
            opacity=0.75,
            showlegend=False,
        )
    )

fig_pub.add_hline(
    y=1.0,
    line_dash="dash",
    line_color="black",
    line_width=1,
)

fig_pub.update_yaxes(title_text="Scaling Factor")
fig_pub.update_xaxes(title_text="Tracker")

fig_pub.update_layout(
    height=450,
    width=600,
    template="plotly_white",
    font=dict(size=13),
    margin=dict(l=60, r=30, t=50, b=60),
)

fig_pub.show()

fig_pub.write_image(save_root / "scaling_factor_boxplot.svg", scale=3)