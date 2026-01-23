from pathlib import Path
from skellymodels.managers.human import Human


recording_path = Path(r"D:\2025_07_31_JSM_pilot\freemocap\2025-07-31_16-35-10_GMT-4_jsm_treadmill_trial_1")

tracker_list = ["qualisys", "vitpose_25", "vitpose_wholebody",]

skeleton_models = {}
for tracker in tracker_list:
    skeleton_models[tracker] = Human.from_data(recording_path/"validation"/tracker)

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

JOINTS = [
    ("left_foot_index",  "Left foot index"),
    ("left_heel",        "Left heel"),
    ("right_foot_index", "Right foot index"),
    ("right_heel",       "Right heel"),
]
DIMS = [("x", 0), ("y", 1), ("z", 2)]

TRACKER_COLORS = {
    "qualisys": "black",
    "vitpose_25": "#1f77b4",         # blue
    "vitpose_wholebody": "#ff7f0e",  # orange
}

def plot_feet_xyz_all_trackers(skeleton_models, title="Feet markers (XYZ) — all trackers"):
    fig = make_subplots(
        rows=len(JOINTS),
        cols=len(DIMS),
        shared_xaxes=True,
        vertical_spacing=0.06,
        horizontal_spacing=0.04,
        row_titles=[label for _, label in JOINTS],
        column_titles=[d.upper() for d, _ in DIMS],
    )

    # only show legend once, in the first subplot
    legend_shown = set()

    for r, (joint_key, joint_label) in enumerate(JOINTS, start=1):
        for c, (dim_name, dim_idx) in enumerate(DIMS, start=1):

            for tracker_name, human_model in skeleton_models.items():
                xyz_dict = human_model.body.xyz.as_dict
                arr = np.asarray(xyz_dict[joint_key])  # (F, 3)
                frames = np.arange(arr.shape[0])

                showlegend = False
                if tracker_name not in legend_shown and r == 1 and c == 1:
                    showlegend = True
                    legend_shown.add(tracker_name)

                fig.add_trace(
                    go.Scatter(
                        x=frames,
                        y=arr[:, dim_idx],
                        mode="lines",
                        name=tracker_name,
                        legendgroup=tracker_name,
                        showlegend=showlegend,
                        line=dict(color=TRACKER_COLORS.get(tracker_name, None), width=2),
                    ),
                    row=r,
                    col=c,
                )

            if c == 1:
                fig.update_yaxes(title_text="Position (units)", row=r, col=c)

    fig.update_xaxes(title_text="Frame", row=len(JOINTS), col=2)
    fig.update_layout(
        title=title,
        height=1200,
        width=2200,
        template="plotly_white",
        legend_title_text="Tracker",
    )
    return fig

fig = plot_feet_xyz_all_trackers(
    skeleton_models,
    title="Feet markers (XYZ) — Qualisys vs ViTPose",
)
fig.show()


