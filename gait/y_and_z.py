from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from skellymodels.managers.human import Human
import plotly.colors as pc

# --------- inputs ---------
path_to_recording = Path(r'D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_12_36_TF01_flexion_pos_5_6_trial_1')
tracker = 'mediapipe_dlc'
active_frame_range = range(600, 2350)   # frames to search for steps in
path_to_gait_events = path_to_recording / 'validation' / 'qualisys' / 'gait_events.csv'

# --------- data ---------
qualisys_human: Human = Human.from_data(path_to_recording/'validation'/'qualisys')
freemocap_human: Human = Human.from_data(path_to_recording/'validation'/tracker)
gait_events = pd.read_csv(path_to_gait_events)

# Heel-strike slices (right foot)
mask = (gait_events["foot"]=="right") & (gait_events["event"]=="heel_strike")
heelstrikes = [f for f in gait_events.loc[mask, "frame"].to_list() if f in active_frame_range]
hs_slices = [slice(heelstrikes[i], heelstrikes[i+1]) for i in range(len(heelstrikes)-1) if heelstrikes[i+1] > heelstrikes[i] + 5]

# --------- helpers ---------
def get_track(h: Human, marker: str) -> np.ndarray:
    """Return (T,3) track for marker."""
    return h.body.xyz.as_dict[marker]

def cycles_from_slices(arr_1d: np.ndarray, slices: list[slice]) -> list[np.ndarray]:
    return [arr_1d[s] for s in slices if (s.stop - s.start) > 5]

def normalize_cycle(v: np.ndarray, n_points:int=101) -> np.ndarray:
    x = np.linspace(0, 1, num=len(v))
    x_new = np.linspace(0, 1, num=n_points)
    return np.interp(x_new, x, v)

def normalize_cycles(v_list: list[np.ndarray], n_points:int=101) -> np.ndarray:
    if not v_list:
        return np.empty((0, n_points))
    return np.vstack([normalize_cycle(v, n_points) for v in v_list])

def mean_std(cycles: list[np.ndarray], n_points:int=101) -> tuple[np.ndarray, np.ndarray]:
    X = normalize_cycles(cycles, n_points)
    if X.size == 0:
        return np.array([]), np.array([])
    return X.mean(axis=0), X.std(axis=0)

def to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a hex color like '#1f77b4' to an rgba string with given alpha."""
    r, g, b = pc.hex_to_rgb(hex_color)
    return f"rgba({r},{g},{b},{alpha})"

# --------- markers & axes ---------
markers = [
    ("right_foot_index", "Toe (Index)"),
    ("right_ankle",      "Ankle"),
    ("right_heel",       "Heel")
]
AX_Y, AX_Z = 1, 2  # your convention: x=ML, y=AP, z=vertical

# Collect per-marker, per-axis cycles for both systems
results = {}  # key: (marker_label, axis_label) -> dict(system->(mean, std))
for marker_name, label in markers:
    # FMC
    fmc_xyz = get_track(freemocap_human, marker_name)
    fmc_y_cycles = cycles_from_slices(fmc_xyz[:, AX_Y], hs_slices)
    fmc_z_cycles = cycles_from_slices(fmc_xyz[:, AX_Z], hs_slices)
    fmc_y_mean, fmc_y_std = mean_std(fmc_y_cycles)
    fmc_z_mean, fmc_z_std = mean_std(fmc_z_cycles)

    # QTM
    qtm_xyz = get_track(qualisys_human, marker_name)
    qtm_y_cycles = cycles_from_slices(qtm_xyz[:, AX_Y], hs_slices)
    qtm_z_cycles = cycles_from_slices(qtm_xyz[:, AX_Z], hs_slices)
    qtm_y_mean, qtm_y_std = mean_std(qtm_y_cycles)
    qtm_z_mean, qtm_z_std = mean_std(qtm_z_cycles)

    results[(label, "Y (AP)")] = {
        "FMC": (fmc_y_mean, fmc_y_std),
        "QTM": (qtm_y_mean, qtm_y_std),
    }
    results[(label, "Z (Vertical)")] = {
        "FMC": (fmc_z_mean, fmc_z_std),
        "QTM": (qtm_z_mean, qtm_z_std),
    }

# --------- plotting ---------
t_gc = np.linspace(0, 100, 101)
fig = make_subplots(
    rows=3, cols=2, shared_xaxes=True, shared_yaxes=False,
    vertical_spacing=0.08, horizontal_spacing=0.05,
    subplot_titles=[
        "<b>Toe – Y (AP)</b>", "<b>Toe – Z (Vertical)</b>",
        "<b>Ankle – Y (AP)</b>", "<b>Ankle – Z (Vertical)</b>",
        "<b>Heel – Y (AP)</b>", "<b>Heel – Z (Vertical)</b>"
    ]
)

# simple color picks
col_fmc = "blue"
col_qtm = "red"
alpha = 0.12

def add_mean_sd(fig, row, col, mean, std, system, hex_color, alpha=0.2):
    if mean.size == 0:
        return
    showlegend = True if (row == 1 and col == 1) else False  # only show once
    # mean line
    fig.add_trace(
        go.Scatter(
            x=t_gc, y=mean,
            mode="lines",
            name=system,                 # just 'FMC' or 'QTM'
            legendgroup=system,
            line=dict(color=hex_color),
            showlegend=showlegend,
        ),
        row=row, col=col
    )
    # ± SD band
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([t_gc, t_gc[::-1]]),
            y=np.concatenate([mean + std, (mean - std)[::-1]]),
            fill="toself",
            fillcolor=to_rgba(hex_color, alpha),
            line=dict(width=0),
            name=f"{system} ± SD",
            legendgroup=system,
            showlegend=False,  # never show these
        ),
        row=row, col=col
    )

col_fmc = "#1f77b4"  # blue
col_qtm = "#d62728"  # red

# Row/col mapping
panel_map = {
    ("Toe (Index)", "Y (AP)"): (1,1),
    ("Toe (Index)", "Z (Vertical)"): (1,2),
    ("Ankle", "Y (AP)"): (2,1),
    ("Ankle", "Z (Vertical)"): (2,2),
    ("Heel", "Y (AP)"): (3,1),
    ("Heel", "Z (Vertical)"): (3,2),
}

for key, (row, col) in panel_map.items():
    data = results[key]
    fmc_mean, fmc_std = data["FMC"]
    qtm_mean, qtm_std = data["QTM"]
    add_mean_sd(fig, row, col, fmc_mean, fmc_std, "FMC", col_fmc)
    add_mean_sd(fig, row, col, qtm_mean, qtm_std, "QTM", col_qtm)


fig.update_layout(
    title="Right Foot Y and Z Trajectories over Gait Cycle (Mean ± SD)",
    title_font=dict(size=22),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-.05,
        xanchor="center",
        x=0.5,
        bordercolor="black",
        borderwidth=1,
        font=dict(size=20),
    ),
    font=dict(size=24),  # axis tick labels
)

# bump subplot title font size
for ann in fig['layout']['annotations']:
    ann['font'] = dict(size=26)  

# Update axes titles
fig.update_xaxes(title_text="<b>% Gait Cycle</b>", row=3, col=1, title_font=dict(size=24))
fig.update_xaxes(title_text="<b>% Gait Cycle</b>", row=3, col=2, title_font=dict(size=24))
fig.update_yaxes(title_text="<b>Position (mm)</b>", row=1, col=1, title_font=dict(size=24))
fig.update_yaxes(title_text="<b>Position (mm)</b>", row=1, col=2, title_font=dict(size=24))
fig.update_yaxes(title_text="<b>Position (mm)</b>", row=2, col=1, title_font=dict(size=24))
fig.update_yaxes(title_text="<b>Position (mm)</b>", row=2, col=2, title_font=dict(size=24))
fig.update_yaxes(title_text="<b>Position (mm)</b>", row=3, col=1, title_font=dict(size=24))
fig.update_yaxes(title_text="<b>Position (mm)</b>", row=3, col=2, title_font=dict(size=24))

fig.show()
