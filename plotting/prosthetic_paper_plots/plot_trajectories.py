from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skellymodels.managers.human import Human

# -------------------------------------------------------
# Paths / basic config
# -------------------------------------------------------
recording_path = Path(
    r"D:\2023-06-07_TF01\1.0_recordings\four_camera"
    r"\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1\validation"
)

mediapipe_dlc = recording_path / "mediapipe_dlc"
qualisys = recording_path / "qualisys"
mediapipe = recording_path / "mediapipe"

trackers = {
    "qualisys": qualisys,
    "mediapipe_dlc": mediapipe_dlc,
    "mediapipe": mediapipe,
}

# Frame window to show (adjust as you like)
frame_range = (1050, 1200)
start, stop = frame_range
frames = np.arange(start, stop)

# Joint names in the Human model and their row labels
JOINTS = [
    ("right_knee",  "KNEE"),
    ("right_ankle", "ANKLE"),
    ("right_heel",  "HEEL"),
    ("right_foot_index", "TOE"),
]

# Axis labels with anatomical directions
AXES = [
    "X (medio-lateral)",
    "Y (anterior-posterior)",
    "Z (height)",
]

SYSTEM_STYLE = {
    "qualisys": {
        "color": "#000000",
        "label": "Qualisys",
        "width": 2.5,
    },
    "mediapipe_dlc": {
        "color": "#1f77b4",  # blue
        "label": "Mediapipe + DLC",
        "width": 2,
    },
    "mediapipe": {
        "color": "#d62728",  # red
        "label": "Mediapipe",
        "width": 1.5,
    },
}

# -------------------------------------------------------
# Load data into systems_dict[system][joint] -> (N,3)
# -------------------------------------------------------
systems_dict: dict[str, dict[str, np.ndarray]] = {}

for system_name, tracker_dir in trackers.items():
    human: Human = Human.from_data(tracker_dir)
    joint_dict: dict[str, np.ndarray] = {}

    for joint_key, _row_label in JOINTS:
        arr = human.body.xyz.as_dict[joint_key][start:stop]
        joint_dict[joint_key] = np.asarray(arr)

    systems_dict[system_name] = joint_dict

# -------------------------------------------------------
# Build subplot grid: rows = joints, cols = X/Y/Z
# -------------------------------------------------------
n_rows = len(JOINTS)
n_cols = 3  # X, Y, Z

fig = make_subplots(
    rows=n_rows,
    cols=n_cols,
    shared_xaxes=True,
    horizontal_spacing=0.03,
    vertical_spacing=0.03,
    column_titles=AXES,
    row_titles=[row_label for _, row_label in JOINTS],
)

for col_idx in range(1, n_cols + 1):
    axis_dim = col_idx - 1  # 0=X, 1=Y, 2=Z

    for row_idx, (joint_key, _row_label) in enumerate(JOINTS, start=1):
        for system_name, joint_dict in systems_dict.items():
            style = SYSTEM_STYLE[system_name]
            data = joint_dict[joint_key][:, axis_dim]

            # Only show legend once (top-left subplot)
            show_legend = (row_idx == 1 and col_idx == 1)

            fig.add_trace(
                go.Scatter(
                    x=frames,
                    y=data,
                    mode="lines",
                    name=style["label"] if show_legend else None,
                    legendgroup=system_name,
                    showlegend=show_legend,
                    line=dict(
                        color=style["color"],
                        width=style["width"],
                    ),
                ),
                row=row_idx,
                col=col_idx,
            )

# -------------------------------------------------------
# Layout tweaks
# -------------------------------------------------------
fig.update_layout(
    height=900,
    width=1600,
    template="simple_white",
    margin=dict(l=100, r=20, t=60, b=40),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1.0,
    ),
)

# Clear x titles, set y titles to "Position (mm)"
for i in range(1, n_rows * n_cols + 1):
    fig["layout"][f"xaxis{i}"]["title"] = ""
    fig["layout"][f"yaxis{i}"]["title"] = ""

# Add y-axis titles only to the LEFTMOST column
for row_idx in range(1, n_rows + 1):
    axis_index = (row_idx - 1) * n_cols + 1   # 1 → 1, 2 → 4, 3 → 7, 4 → 10
    fig["layout"][f"yaxis{axis_index}"]["title"] = "Position (mm)"

# Bottom-row x-axis labels
for col_idx in range(1, n_cols + 1):
    fig["layout"][f"xaxis{(n_rows - 1) * n_cols + col_idx}"]["title"] = "Frame"

# Big shared y-axis label
fig.add_annotation(
    text="Position (mm)",
    xref="paper", yref="paper",
    x=0.0, y=0.5,
    showarrow=False,
    textangle=-90,
    font=dict(size=16),
)

fig.show()
