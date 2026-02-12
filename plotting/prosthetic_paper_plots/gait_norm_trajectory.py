from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =========================
# Inputs
# =========================
path_to_neutral_recordings = {
    Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1"),
    Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_28_46_TF01_toe_angle_neutral_trial_1"),
    Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_46_54_TF01_leg_length_neutral_trial_1"),
}

reference = ["qualisys"]
trackers = ["mediapipe_dlc"]  # <-- only this

ALL_SYSTEMS = reference + trackers

# If you want a clean “paper intro” figure, I strongly recommend curating markers:
CURATED_RIGHT_MARKERS = [
    "right_knee",
    "right_ankle",
    "right_heel",
    "right_foot_index",
]
# Set to None to include all right-side markers found
USE_CURATED = True


# =========================
# Layout / style (tighter)
# =========================
SUBPLOT_WIDTH_IN = 1.7
SUBPLOT_HEIGHT_IN = 1.1
DPI = 120

MARGIN_LEFT_IN = 1.3
MARGIN_RIGHT_IN = 0.15
MARGIN_TOP_IN = 0.6
MARGIN_BOTTOM_IN = 0.55

# BIG FIX: reduce spacing for many rows
V_SPACING = 0.015
H_SPACING = 0.035

AXES_TO_PLOT = ["x", "y", "z"]
AXIS_LABEL = {"x": "X", "y": "Y", "z": "Z"}

LINE_WIDTH = 2.2

TRACKER_STYLE = {
    "qualisys":      dict(name="Qualisys", dash="solid", color="#7f7f7f", fill_opacity=0.12),
    "mediapipe_dlc": dict(name="MediaPipe+DLC", dash="solid", color="#1f77b4", fill_opacity=0.12),
}
DRAW_ORDER = ["qualisys", "mediapipe_dlc"]


def rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def traj_csv_path(recording_path: Path, system: str) -> Path:
    return recording_path / "validation" / system / "trajectories" / "trajectories_per_stride.csv"


def load_one(recording_path: Path, system: str) -> pd.DataFrame:
    p = traj_csv_path(recording_path, system)
    if not p.exists():
        raise FileNotFoundError(f"Missing:\n  {p}")

    df = pd.read_csv(p)

    required = {"marker", "x", "y", "z", "cycle", "percent_gait_cycle"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{p} missing columns: {missing}")

    df["recording"] = recording_path.name
    df["tracker"] = system
    df["percent_gait_cycle"] = df["percent_gait_cycle"].astype(int)
    return df


def is_right_marker(m: str) -> bool:
    m = (m or "").lower().strip()
    return (
        m.startswith("right") or
        m.startswith("r_") or
        m.startswith("r ") or
        m.startswith("rt_") or
        (" right" in m) or
        m.endswith("_r")
    )


def trial_mean(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["recording", "tracker", "marker", "percent_gait_cycle"], as_index=False)[["x", "y", "z"]]
        .mean(numeric_only=True)
    )


def across_trials_mean_sd(trial_means: pd.DataFrame) -> pd.DataFrame:
    agg = (
        trial_means.groupby(["tracker", "marker", "percent_gait_cycle"], as_index=False)
        .agg(
            mean_x=("x", "mean"), sd_x=("x", "std"),
            mean_y=("y", "mean"), sd_y=("y", "std"),
            mean_z=("z", "mean"), sd_z=("z", "std"),
        )
        .sort_values(["tracker", "marker", "percent_gait_cycle"])
    )
    for c in ["sd_x", "sd_y", "sd_z"]:
        agg[c] = agg[c].fillna(0.0)
    return agg


def to_long(mean_sd: pd.DataFrame) -> pd.DataFrame:
    out = []
    for axis in ["x", "y", "z"]:
        out.append(
            mean_sd.rename(columns={f"mean_{axis}": "mean_value", f"sd_{axis}": "sd_value"})[
                ["tracker", "marker", "percent_gait_cycle", "mean_value", "sd_value"]
            ].assign(axis=axis)
        )
    return pd.concat(out, ignore_index=True)


# =========================
# Load all neutral trials
# =========================
raw = pd.concat(
    [load_one(rec, system) for rec in sorted(path_to_neutral_recordings) for system in ALL_SYSTEMS],
    ignore_index=True,
)

# Right-side only
raw = raw[raw["marker"].astype(str).map(is_right_marker)].copy()

# Optional: curated marker list for a clean intro figure
if USE_CURATED:
    raw = raw[raw["marker"].isin(CURATED_RIGHT_MARKERS)].copy()

if raw.empty:
    raise RuntimeError("After right-side filtering, no rows remain. Check marker naming.")

trial_means = trial_mean(raw)
mean_sd = across_trials_mean_sd(trial_means)
mean_long = to_long(mean_sd)

# Marker order (use curated order if requested)
if USE_CURATED:
    MARKER_ORDER = [m for m in CURATED_RIGHT_MARKERS if m in mean_long["marker"].unique()]
else:
    # prefer qualisys order
    q = mean_long[(mean_long["tracker"] == "qualisys") & (mean_long["axis"] == "x")]["marker"].drop_duplicates().tolist()
    MARKER_ORDER = q if q else sorted(mean_long["marker"].unique().tolist())

n_rows = len(MARKER_ORDER)
n_cols = len(AXES_TO_PLOT)

FIG_WIDTH_IN = MARGIN_LEFT_IN + (n_cols * SUBPLOT_WIDTH_IN) + MARGIN_RIGHT_IN
FIG_HEIGHT_IN = MARGIN_TOP_IN + (n_rows * SUBPLOT_HEIGHT_IN) + MARGIN_BOTTOM_IN

FIG_WIDTH_PX = int(FIG_WIDTH_IN * DPI)
FIG_HEIGHT_PX = int(FIG_HEIGHT_IN * DPI)

MARGIN_LEFT_PX = int(MARGIN_LEFT_IN * DPI)
MARGIN_RIGHT_PX = int(MARGIN_RIGHT_IN * DPI)
MARGIN_TOP_PX = int(MARGIN_TOP_IN * DPI)
MARGIN_BOTTOM_PX = int(MARGIN_BOTTOM_IN * DPI)


# =========================
# Plot
# =========================
fig = make_subplots(
    rows=n_rows,
    cols=n_cols,
    shared_xaxes=True,
    shared_yaxes=False,  # <-- keep False (this does NOT force equal y)
    vertical_spacing=V_SPACING,
    horizontal_spacing=H_SPACING,
    column_titles=[f"<b>{AXIS_LABEL[a]}</b>" for a in AXES_TO_PLOT],
)

tickvals = list(range(0, 101, 25))

for r, marker in enumerate(MARKER_ORDER, start=1):
    for c, axis in enumerate(AXES_TO_PLOT, start=1):
        for tracker in DRAW_ORDER:
            sub = mean_long[
                (mean_long["tracker"] == tracker) &
                (mean_long["marker"] == marker) &
                (mean_long["axis"] == axis)
            ].sort_values("percent_gait_cycle")

            if sub.empty:
                continue

            xgc = sub["percent_gait_cycle"].to_numpy()
            mean = sub["mean_value"].to_numpy()
            sd = sub["sd_value"].to_numpy()
            lower, upper = mean - sd, mean + sd

            style = TRACKER_STYLE[tracker]
            color = style["color"]
            fill_color = rgba(color, style["fill_opacity"])

            # SD ribbon (optional—keep if you like)
            fig.add_trace(
                go.Scatter(
                    x=xgc, y=upper, mode="lines",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=tracker,
                ),
                row=r, col=c
            )
            fig.add_trace(
                go.Scatter(
                    x=xgc, y=lower, mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=fill_color,
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=tracker,
                ),
                row=r, col=c
            )

            # Mean line (now solid for both)
            showleg = (r == 1 and c == 1)
            fig.add_trace(
                go.Scatter(
                    x=xgc, y=mean, mode="lines",
                    line=dict(color=color, width=LINE_WIDTH, dash="solid"),
                    name=style["name"],
                    legendgroup=tracker,
                    showlegend=showleg,
                    hovertemplate=(
                        f"{style['name']}<br>"
                        f"{marker}<br>{axis.upper()}<br>"
                        "Value: %{y:.1f}<br>"
                        "Gait cycle: %{x:.0f}%<extra></extra>"
                    ),
                ),
                row=r, col=c
            )
for marker in MARKER_ORDER:
    rr = MARKER_ORDER.index(marker) + 1
    fig.add_annotation(
        x=-0.035, xref="paper",
        y=1 - (rr - 0.5) / n_rows, yref="paper",
        text=f"<b>{marker.replace('_',' ')}</b>",
        showarrow=False, xanchor="right",
        font=dict(size=12, color="#222"),
        align="right",
    )

# Axes cosmetics
for r in range(1, n_rows + 1):
    for c in range(1, n_cols + 1):
        fig.update_xaxes(
            title_text="<b>Gait cycle (%)</b>" if r == n_rows else None,
            tickvals=tickvals,
            tickfont=dict(size=10),
            showgrid=False, zeroline=False,
            showline=True, linecolor="#333", mirror=True,
            row=r, col=c
        )
        fig.update_yaxes(
            showticklabels=True,   # <-- since y ranges differ, it’s helpful to show y ticks everywhere
            tickfont=dict(size=10),
            showgrid=False, zeroline=False,
            showline=True, linecolor="#333", mirror=True,
            row=r, col=c
        )

fig.update_layout(
    template="plotly_white",
    height=FIG_HEIGHT_PX,
    width=FIG_WIDTH_PX,
    margin=dict(l=MARGIN_LEFT_PX, r=MARGIN_RIGHT_PX, t=MARGIN_TOP_PX, b=MARGIN_BOTTOM_PX),
    title=dict(
        text="<b>Neutral alignment (TF01): right-side trajectories (mean ± SD across neutral trials)</b>",
        font=dict(size=14),
        y=0.98, x=0.5, xanchor="center", yanchor="top",
    ),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.14,
        xanchor="center",
        x=0.5,
        font=dict(size=12),
    ),
    paper_bgcolor="white",
    plot_bgcolor="white",
)

fig.show()