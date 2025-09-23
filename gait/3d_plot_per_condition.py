from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skellymodels.managers.human import Human

# ----------------------- Inputs -----------------------
conditions = {
    "neutral": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1",
    "neg_2_8": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_03_15_TF01_flexion_neg_2_8_trial_1",
    "neg_5_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_11_55_05_TF01_flexion_neg_5_6_trial_1",
    "pos_2_8": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_09_05_TF01_flexion_pos_2_8_trial_1",
    "pos_5_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_12_36_TF01_flexion_pos_5_6_trial_1",
}
tracker = "mediapipe_dlc"
FOOT = "right"
NPTS = 101                      # normalized 0..100 %GC
SHOW_TRAIL = True               # toe trails per condition
ANCHOR_AT_ANKLE0 = True
active_frame_range = range(600, 2350)  # used for gait event slicing per condition

# Color map (5 distinct)
COND_COLORS = {
    "neg_5_6": "#2ca02c",
    "neg_2_8": "#ff7f0e",
    "neutral": "#111111",
    "pos_2_8": "#9467bd",
    "pos_5_6": "#1f77b4",
}



# ----------------------- Helpers ----------------------
def get_track_any(h: Human, candidates: list[str]) -> np.ndarray:
    D = h.body.xyz.as_dict
    for k in candidates:
        if k in D:
            return D[k]
    raise KeyError(f"None of the markers found: {candidates}")

def interp_1d(v, n):
    x = np.linspace(0, 1, len(v))
    return np.interp(np.linspace(0, 1, n), x, v)

def normalize_cycles_xyz(traj: np.ndarray, slices: list[slice], n_points: int) -> np.ndarray:
    chunks = []
    for s in slices:
        seg = traj[s]
        if len(seg) < 6:
            continue
        chunks.append(np.column_stack([interp_1d(seg[:, i], n_points) for i in range(3)]))
    return np.stack(chunks, axis=0) if chunks else np.empty((0, n_points, 3))

def mean_xyz(traj: np.ndarray, slices: list[slice], n_points: int) -> np.ndarray:
    X = normalize_cycles_xyz(traj, slices, n_points)
    return X.mean(axis=0) if X.size else np.zeros((n_points, 3))

def find_hs_slices(path_to_recording: Path, foot: str, active: range) -> list[slice]:
    ge = pd.read_csv(path_to_recording / "validation" / "qualisys" / "gait_events.csv")
    mask = (ge["foot"] == foot) & (ge["event"] == "heel_strike")
    heelstrikes = [f for f in ge.loc[mask, "frame"].to_list() if f in active]
    hs = [slice(heelstrikes[i], heelstrikes[i+1])
          for i in range(len(heelstrikes)-1) if heelstrikes[i+1] > heelstrikes[i] + 5]
    if not hs:
        raise RuntimeError(f"No valid heel-strike cycles in {path_to_recording}")
    return hs

def system_means(h: Human, slices: list[slice], n_points: int, anchor_at_ankle0: bool):
    knee  = mean_xyz(get_track_any(h,  [f"{FOOT}_knee"]), slices, n_points)
    ankle = mean_xyz(get_track_any(h,  [f"{FOOT}_ankle"]), slices, n_points)
    toe   = mean_xyz(get_track_any(h,  [f"{FOOT}_foot_index", f"{FOOT}_toe", f"{FOOT}_toe_tip"]), slices, n_points)
    heel  = mean_xyz(get_track_any(h,  [f"{FOOT}_heel", f"{FOOT}_foot_heel", f"{FOOT}_calcaneus"]), slices, n_points)
    if anchor_at_ankle0:
        offset = ankle[0].copy()
        knee  = knee  - offset
        ankle = ankle - offset
        toe   = toe   - offset
        heel  = heel  - offset
    return dict(knee=knee, ankle=ankle, toe=toe, heel=heel)

def seg_line(p1, p2, color, name=None, showlegend=False, width=5):
    return go.Scatter(
        x=[p1[1], p2[1]], y=[p1[2], p2[2]], mode="lines",
        line=dict(color=color, width=width),
        name=name, showlegend=showlegend
    )

def trail_points(P, color, name=None, showlegend=False):
    return go.Scatter(
        x=P[:,1], y=P[:,2], mode="markers",
        marker=dict(size=3, color=color, opacity=0.6),
        name=name, showlegend=showlegend
    )

# ----------------------- Load & build per-condition means ----------------------
FMC_means = {}  # cond -> dict(knee,ankle,toe,heel)
QTM_means = {}

# Collect global axis limits across all conditions + systems
all_yz = []

for cond, root in conditions.items():
    root = Path(root)

    # Heel-strike slices for this condition
    hs_slices = find_hs_slices(root, FOOT, active_frame_range)

    # Humans
    qtm = Human.from_data(root / "validation" / "qualisys")
    fmc = Human.from_data(root / "validation" / tracker)

    # Means
    FMC_means[cond] = system_means(fmc, hs_slices, NPTS, ANCHOR_AT_ANKLE0)
    QTM_means[cond] = system_means(qtm, hs_slices, NPTS, ANCHOR_AT_ANKLE0)

    # For axis ranges
    for sys_means in (FMC_means[cond], QTM_means[cond]):
        all_yz.append(sys_means['knee'][:,1:])
        all_yz.append(sys_means['ankle'][:,1:])
        all_yz.append(sys_means['toe'][:,1:])
        all_yz.append(sys_means['heel'][:,1:])

# Axis ranges (shared across both panes)
stack = np.vstack(all_yz)
mins = stack.min(axis=0); maxs = stack.max(axis=0)
pad = 0.05 * (maxs - mins + 1e-6)
(ymin, zmin) = (mins - pad); (ymax, zmax) = (maxs + pad)

# ----------------------- Figure & initial traces ----------------------
fig = make_subplots(
    rows=1, cols=2,
    column_widths=[0.5, 0.5],
    horizontal_spacing=0.08,
    specs=[[{"type":"xy"}, {"type":"xy"}]],
    subplot_titles=("FreeMoCap (tracker)", "Qualisys")
)

gc = np.linspace(0, 100, NPTS)
t0 = 0

# Trace order (keep stable in frames)
# Left panel (FMC): per condition -> [trail?, shank, foot, heel, ankle marker]
# Right panel (QTM): same order per condition

# ------------- Initial draw -------------
# Left (FMC)
for cond, color in COND_COLORS.items():
    name = f"FMC • {cond}"
    means = FMC_means[cond]
    if SHOW_TRAIL:
        fig.add_trace(trail_points(means['toe'][:t0+1], color, name=name, showlegend=False), row=1, col=1)
    else:
        # add a placeholder empty trace to keep indices stable
        fig.add_trace(trail_points(np.zeros((0,3)), color, name=name, showlegend=False), row=1, col=1)
    fig.add_trace(seg_line(means['ankle'][t0], means['knee'][t0], color), row=1, col=1)
    fig.add_trace(seg_line(means['ankle'][t0], means['toe'][t0],  color), row=1, col=1)
    fig.add_trace(seg_line(means['ankle'][t0], means['heel'][t0], color), row=1, col=1)
    fig.add_trace(go.Scatter(x=[means['ankle'][t0,1]], y=[means['ankle'][t0,2]], mode="markers",
                             marker=dict(size=7, color=color), showlegend=False,
                             name=name), row=1, col=1)

# Right (QTM)
for cond, color in COND_COLORS.items():
    name = f"QTM • {cond}"
    means = QTM_means[cond]
    if SHOW_TRAIL:
        fig.add_trace(trail_points(means['toe'][:t0+1], color, name=name, showlegend=False), row=1, col=2)
    else:
        fig.add_trace(trail_points(np.zeros((0,3)), color, name=name, showlegend=False), row=1, col=2)
    fig.add_trace(seg_line(means['ankle'][t0], means['knee'][t0], color), row=1, col=2)
    fig.add_trace(seg_line(means['ankle'][t0], means['toe'][t0],  color), row=1, col=2)
    fig.add_trace(seg_line(means['ankle'][t0], means['heel'][t0], color), row=1, col=2)
    fig.add_trace(go.Scatter(x=[means['ankle'][t0,1]], y=[means['ankle'][t0,2]], mode="markers",
                             marker=dict(size=7, color=color), showlegend=False,
                             name=name), row=1, col=2)

# Layout / axes
fig.update_layout(
    title="Gait-AVERAGED right leg in Y–Z per condition (Left: FMC, Right: QTM)",
)

fig.update_xaxes(title_text="Y (A–P)", range=[ymin, ymax], row=1, col=1)
fig.update_yaxes(title_text="Z (Vertical)", range=[zmin, zmax], row=1, col=1, scaleanchor="x1", scaleratio=1)
fig.update_xaxes(title_text="Y (A–P)", range=[ymin, ymax], row=1, col=2)
fig.update_yaxes(title_text="Z (Vertical)", range=[zmin, zmax], row=1, col=2, scaleanchor="x2", scaleratio=1)

# ----------------------- Frames ----------------------
frames = []
for i in range(NPTS):
    data_list = []

    # Left (FMC)
    for cond, color in COND_COLORS.items():
        m = FMC_means[cond]
        # trail (or empty)
        data_list.append(trail_points(m['toe'][:i+1] if SHOW_TRAIL else np.zeros((0,3)), color))
        # shank, foot, heel, ankle marker
        data_list.append(seg_line(m['ankle'][i], m['knee'][i], color))
        data_list.append(seg_line(m['ankle'][i], m['toe'][i],  color))
        data_list.append(seg_line(m['ankle'][i], m['heel'][i], color))
        data_list.append(go.Scatter(x=[m['ankle'][i,1]], y=[m['ankle'][i,2]], mode="markers",
                                    marker=dict(size=7, color=color), showlegend=False))

    # Right (QTM)
    for cond, color in COND_COLORS.items():
        m = QTM_means[cond]
        data_list.append(trail_points(m['toe'][:i+1] if SHOW_TRAIL else np.zeros((0,3)), color))
        data_list.append(seg_line(m['ankle'][i], m['knee'][i], color))
        data_list.append(seg_line(m['ankle'][i], m['toe'][i],  color))
        data_list.append(seg_line(m['ankle'][i], m['heel'][i], color))
        data_list.append(go.Scatter(x=[m['ankle'][i,1]], y=[m['ankle'][i,2]], mode="markers",
                                    marker=dict(size=7, color=color), showlegend=False))

    frames.append(go.Frame(name=str(i), data=data_list))

fig.frames = frames

# ----------------------- Controls ----------------------
steps = [dict(method="animate",
              args=[[str(i)], {"mode":"immediate",
                               "frame":{"duration":0, "redraw":True},
                               "transition":{"duration":0}}],
              label=f"{i}") for i in range(NPTS)]

fig.update_layout(
    updatemenus=[{
        "type": "buttons",
        "buttons": [
            {"label":"▶ Play", "method":"animate",
             "args":[None, {"frame":{"duration":30,"redraw":True},
                            "fromcurrent":True, "transition":{"duration":0},}]},
            {"label":"⏸ Pause", "method":"animate",
             "args":[[None], {"mode":"immediate","frame":{"duration":0},"transition":{"duration":0}}]}
        ],
        "showactive": False, "x":0.02, "y":1.12, "xanchor":"left", "yanchor":"top"
    }],
    sliders=[{
        "active": 0, "steps": steps, "x":0.18, "y":1.08, "xanchor":"left", "yanchor":"top",
        "len": 0.75, "currentvalue":{"prefix":"%GC: ", "visible":True}
    }]
)

def add_condition_legend(fig, cond_colors):
    """
    Add one invisible trace per condition, so legend shows each condition once.
    """
    for cond, color in cond_colors.items():
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="lines",
                line=dict(color=color, width=6),
                name=cond,
                showlegend=True,
                hoverinfo="skip"
            )
        )

# Call this once before showing the fig:
add_condition_legend(fig, COND_COLORS)

# Move legend to bottom center
fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="top", y=-0.15,
        xanchor="center", x=0.5,
        bgcolor="rgba(255,255,255,0.7)"
    )
)



fig.show()
