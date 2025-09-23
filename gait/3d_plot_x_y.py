from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skellymodels.managers.human import Human

# ----------------------- Inputs -----------------------
path_to_recording = Path(r'D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1')
tracker = 'mediapipe_dlc'
active_frame_range = range(600, 2350)
path_to_gait_events = path_to_recording / 'validation' / 'qualisys' / 'gait_events.csv'

NPTS = 101                     # normalized frames (0..100 %GC)
FOOT = "right"
SHOW_TRAIL = True              # toe trails only
ANCHOR_AT_ANKLE0 = False

# Colors
COL_FMC = "#1f77b4"
COL_QTM = "#d62728" 

# Which labels appear in the angles CSV
FMC_SYSTEM = tracker           # e.g., "mediapipe_dlc"
QTM_SYSTEM = "qualisys"

# ----------------------- Load -------------------------
qualisys_human: Human  = Human.from_data(path_to_recording/'validation'/'qualisys')
freemocap_human: Human = Human.from_data(path_to_recording/'validation'/tracker)
gait_events = pd.read_csv(path_to_gait_events)

# Heel-strike slices (use Qualisys events as reference)
mask = (gait_events["foot"]==FOOT) & (gait_events["event"]=="heel_strike")
heelstrikes = [f for f in gait_events.loc[mask, "frame"].to_list() if f in active_frame_range]
hs_slices = [slice(heelstrikes[i], heelstrikes[i+1])
             for i in range(len(heelstrikes)-1) if heelstrikes[i+1] > heelstrikes[i] + 5]
if not hs_slices:
    raise RuntimeError("No valid heel-strike cycles found in the requested range.")

# ----------------------- Helpers ----------------------
def get_track_any(h: Human, candidates: list[str]) -> np.ndarray:
    D = h.body.xyz.as_dict
    for key in candidates:
        if key in D:
            return D[key]
    raise KeyError(f"None of the markers found: {candidates}. Available: {list(D.keys())[:10]} ...")

def interp_1d(v, n):
    x = np.linspace(0,1,len(v))
    return np.interp(np.linspace(0,1,n), x, v)

def normalize_cycles_xyz(traj: np.ndarray, slices: list[slice], n_points:int) -> np.ndarray:
    chunks = []
    for s in slices:
        seg = traj[s]
        if len(seg) < 6:
            continue
        chunks.append(np.column_stack([interp_1d(seg[:,i], n_points) for i in range(3)]))
    return np.stack(chunks, axis=0) if chunks else np.empty((0,n_points,3))

def mean_xyz(traj: np.ndarray, slices: list[slice], n_points:int) -> np.ndarray:
    X = normalize_cycles_xyz(traj, slices, n_points)
    return X.mean(axis=0) if X.size else np.zeros((n_points,3))

def system_means(h: Human, slices: list[slice], n_points:int, anchor_at_ankle0: bool):
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

def load_angles_from_single_csv(csv_path: Path, system_name: str, n_points: int) -> np.ndarray:
    """
    Load ankle_dorsi_plantar_r from joint_angle_by_stride.csv for a given system
    and resample to n_points over 0–100 % gait cycle.
    """
    df = pd.read_csv(csv_path)

    required_cols = {"system", "angle", "percent_gait_cycle", "value"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV {csv_path} missing required columns. "
            f"Expected at least {required_cols}, got {list(df.columns)}"
        )

    sub = df.loc[
        (df["system"].astype(str).str.lower() == system_name.lower()) &
        (df["angle"].astype(str) == "ankle_dorsi_plantar_r"),
        ["percent_gait_cycle", "value"],
    ]
    if sub.empty:
        raise ValueError(
            f"No rows for system='{system_name}' and angle='ankle_dorsi_plantar_r' in {csv_path}"
        )

    g = (
        sub.groupby("percent_gait_cycle", as_index=False)["value"]
           .mean()
           .sort_values("percent_gait_cycle")
    )

    x = g["percent_gait_cycle"].to_numpy(dtype=float)
    y = g["value"].to_numpy(dtype=float)

    xi = np.linspace(0, 100, n_points)
    yi = np.interp(xi, x, y)
    return yi

# ----------------------- Build means & load angles ----------------------
FMC = system_means(freemocap_human, hs_slices, NPTS, ANCHOR_AT_ANKLE0)
QTM = system_means(qualisys_human,  hs_slices, NPTS, ANCHOR_AT_ANKLE0)

angles_csv = path_to_recording/'validation'/tracker/f'{tracker}_joint_angle_by_stride.csv'
FMC_angle = load_angles_from_single_csv(angles_csv, system_name=FMC_SYSTEM, n_points=NPTS)
QTM_angle = load_angles_from_single_csv(angles_csv, system_name=QTM_SYSTEM, n_points=NPTS)
angle_source = f"angles from '{angles_csv.name}' (system column)"

# ----------------------- Axis limits (X–Y, top plot) -------------------
# Use columns 0 (X: mediolateral) and 1 (Y: anterior–posterior)
stack = np.vstack([
    FMC['knee'][:,0:2], FMC['ankle'][:,0:2], FMC['toe'][:,0:2], FMC['heel'][:,0:2],
    QTM['knee'][:,0:2], QTM['ankle'][:,0:2], QTM['toe'][:,0:2], QTM['heel'][:,0:2]
])
mins = stack.min(axis=0); maxs = stack.max(axis=0)
pad = 0.05*(maxs - mins + 1e-6)
(xmin, ymin) = (mins - pad); (xmax, ymax) = (maxs + pad)

# Angle subplot limits (bottom plot)
all_angles = np.concatenate([FMC_angle, QTM_angle])
aymin = float(np.nanmin(all_angles))
aymax = float(np.nanmax(all_angles))
apad  = 0.05*(aymax - aymin + 1e-6)
aymin -= apad
aymax += apad

# ----------------------- Figure with subplots ----------------------
fig = make_subplots(
    rows=2, cols=1,
    row_heights=[0.68, 0.32],
    vertical_spacing=0.12,
    specs=[[{"type":"xy"}],
           [{"type":"xy"}]],
    subplot_titles=(None, None)
)

def xy_line_segment(p1, p2, color, name, width=5, showlegend=False):
    return go.Scatter(
        x=[p1[0], p2[0]], y=[p1[1], p2[1]],
        mode="lines",
        line=dict(color=color, width=width),
        name=name, showlegend=showlegend
    )

def xy_points(P, color, name, size=6, showlegend=False, opacity=0.6):
    return go.Scatter(
        x=P[:,0], y=P[:,1], mode="markers",
        marker=dict(size=size, color=color, opacity=opacity),
        name=name, showlegend=showlegend
    )

def angle_annotation(i):
    theta_f = FMC_angle[i]
    theta_q = QTM_angle[i]
    d = theta_f - theta_q
    txt = (
        f"<b>Ankle angle (deg)</b> — {angle_source}"
        f"<br><b>FMC:</b> {theta_f:+.2f} | <b>QTM:</b> {theta_q:+.2f} | <b>Δ:</b> {d:+.2f}"
    )
    return [dict(
        text=txt,
        xref="paper", yref="paper",
        x=0.98, y=0.5,             # center-right
        xanchor="right", yanchor="middle",
        showarrow=False,
        align="right",
        font=dict(size=20),
        bgcolor="rgba(255,255,255,0.6)",
        bordercolor="rgba(0,0,0,0.2)", borderwidth=1
    )]

# ---------- BOTTOM PLOT (row=2): add the two static curves FIRST ----------
gc = np.linspace(0, 100, NPTS)
t0 = 0

# 1) FMC angle line
fig.add_trace(
    go.Scatter(x=gc, y=FMC_angle, mode="lines",
               line=dict(width=3, color=COL_FMC),
               name="FMC ankle angle"),
    row=2, col=1
)

# 2) QTM angle line
fig.add_trace(
    go.Scatter(x=gc, y=QTM_angle, mode="lines",
               line=dict(width=3, color=COL_QTM),
               name="QTM ankle angle"),
    row=2, col=1
)

# 3) Cursor (black dashed)
cursor_x = gc[t0]
fig.add_trace(
    go.Scatter(x=[cursor_x, cursor_x], y=[aymin, aymax], mode="lines",
               line=dict(width=2, dash="dash", color="black"),
               name="%GC cursor", showlegend=False),
    row=2, col=1
)

# ---------- TOP PLOT (row=1): then everything else, keeping indices stable ----------
# 4) FMC toe trail (or a placeholder empty trace if SHOW_TRAIL=False)
fig.add_trace(
    xy_points(FMC['toe'][:t0+1] if SHOW_TRAIL else np.zeros((0,3)),
              COL_FMC, "FMC toe trail", size=3, showlegend=True),
    row=1, col=1
)

# 5) QTM toe trail
fig.add_trace(
    xy_points(QTM['toe'][:t0+1] if SHOW_TRAIL else np.zeros((0,3)),
              COL_QTM, "QTM toe trail", size=3, showlegend=True),
    row=1, col=1
)

# 6) FMC shank
fig.add_trace(xy_line_segment(FMC['ankle'][t0], FMC['knee'][t0], COL_FMC, "FMC shank", showlegend=True), row=1, col=1)
# 7) FMC foot
fig.add_trace(xy_line_segment(FMC['ankle'][t0], FMC['toe'][t0],  COL_FMC, "FMC foot"), row=1, col=1)
# 8) FMC heel (same color)
fig.add_trace(xy_line_segment(FMC['ankle'][t0], FMC['heel'][t0], COL_FMC, "FMC heel"), row=1, col=1)
# 9) FMC ankle marker
fig.add_trace(go.Scatter(x=[FMC['ankle'][t0,0]], y=[FMC['ankle'][t0,1]], mode="markers",
                         marker=dict(size=7, color=COL_FMC), name="FMC ankle", showlegend=False), row=1, col=1)

# 10) QTM shank
fig.add_trace(xy_line_segment(QTM['ankle'][t0], QTM['knee'][t0], COL_QTM, "QTM shank", showlegend=True), row=1, col=1)
# 11) QTM foot
fig.add_trace(xy_line_segment(QTM['ankle'][t0], QTM['toe'][t0],  COL_QTM, "QTM foot"), row=1, col=1)
# 12) QTM heel
fig.add_trace(xy_line_segment(QTM['ankle'][t0], QTM['heel'][t0], COL_QTM, "QTM heel"), row=1, col=1)
# 13) QTM ankle marker
fig.add_trace(go.Scatter(x=[QTM['ankle'][t0,0]], y=[QTM['ankle'][t0,1]], mode="markers",
                         marker=dict(size=7, color=COL_QTM), name="QTM ankle", showlegend=False), row=1, col=1)

# Layout / axes
fig.update_layout(
    title="Gait-AVERAGED right leg top-down (X–Y) + ankle flex/ext over %GC (blue=FMC, red=QTM)",
    annotations=angle_annotation(0),
    legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1.0),
)
fig.update_xaxes(title_text="X (Mediolateral)", range=[xmin, xmax], row=1, col=1)
fig.update_yaxes(title_text="Y (Anterior–Posterior)", range=[ymin, ymax], row=1, col=1, scaleanchor="x1", scaleratio=1)
fig.update_xaxes(title_text="% Gait Cycle", range=[0, 100], row=2, col=1)
fig.update_yaxes(title_text="Ankle dorsiflexion (+) [deg]", range=[aymin, aymax], row=2, col=1)

# ---------- Frames (exactly 13 traces in this order EVERY frame) ----------
def trail_points_xy(P, color):
    return go.Scatter(x=P[:,0], y=P[:,1], mode="markers",
                      marker=dict(size=3, color=color, opacity=0.6), showlegend=False)

def seg_line_xy(p1, p2, color):
    return go.Scatter(x=[p1[0], p2[0]], y=[p1[1], p2[1]],
                      mode="lines", line=dict(color=color, width=5), showlegend=False)

frames = []
for i in range(NPTS):
    cursor_x = gc[i]
    data_list = [
        # 1 FMC angle (static)
        go.Scatter(x=gc, y=FMC_angle, mode="lines",
                   line=dict(width=3, color=COL_FMC), showlegend=False),
        # 2 QTM angle (static)
        go.Scatter(x=gc, y=QTM_angle, mode="lines",
                   line=dict(width=3, color=COL_QTM), showlegend=False),
        # 3 Cursor (black dashed)
        go.Scatter(x=[cursor_x, cursor_x], y=[aymin, aymax], mode="lines",
                   line=dict(width=2, dash="dash", color="black"), showlegend=False),
        # 4 FMC toe trail
        trail_points_xy(FMC['toe'][:i+1] if SHOW_TRAIL else np.zeros((0,3)), COL_FMC),
        # 5 QTM toe trail
        trail_points_xy(QTM['toe'][:i+1] if SHOW_TRAIL else np.zeros((0,3)), COL_QTM),
        # 6–9 FMC shank/foot/heel/ankle marker
        seg_line_xy(FMC['ankle'][i], FMC['knee'][i], COL_FMC),
        seg_line_xy(FMC['ankle'][i], FMC['toe'][i],  COL_FMC),
        seg_line_xy(FMC['ankle'][i], FMC['heel'][i], COL_FMC),
        go.Scatter(x=[FMC['ankle'][i,0]], y=[FMC['ankle'][i,1]], mode="markers",
                   marker=dict(size=7, color=COL_FMC), showlegend=False),
        # 10–13 QTM shank/foot/heel/ankle marker
        seg_line_xy(QTM['ankle'][i], QTM['knee'][i], COL_QTM),
        seg_line_xy(QTM['ankle'][i], QTM['toe'][i],  COL_QTM),
        seg_line_xy(QTM['ankle'][i], QTM['heel'][i], COL_QTM),
        go.Scatter(x=[QTM['ankle'][i,0]], y=[QTM['ankle'][i,1]], mode="markers",
                   marker=dict(size=7, color=COL_QTM), showlegend=False),
    ]
    frames.append(go.Frame(name=str(i), data=data_list, layout=dict(annotations=angle_annotation(i))))

fig.frames = frames

# ---------- Controls ----------
steps = [dict(method="animate",
              args=[[str(i)],
                    {"mode":"immediate",
                     "frame":{"duration":0, "redraw":True},
                     "transition":{"duration":0}}],
              label=f"{i}") for i in range(NPTS)]

fig.update_layout(
    updatemenus=[{
        "type":"buttons",
        "buttons":[
            {"label":"▶ Play","method":"animate",
             "args":[None, {"frame":{"duration":30,"redraw":True},"fromcurrent":True,"transition":{"duration":0}}]},
            {"label":"⏸ Pause","method":"animate",
             "args":[[None], {"mode":"immediate","frame":{"duration":0},"transition":{"duration":0}}]}
        ],
        "showactive":False, "x":0.02, "y":1.18, "xanchor":"left", "yanchor":"top"
    }],
    sliders=[{
        "active":0, "steps":steps, "x":0.20, "y":1.14, "xanchor":"left", "yanchor":"top",
        "len":0.75, "currentvalue":{"prefix":"%GC: ", "visible":True}
    }]
)

fig.show()
