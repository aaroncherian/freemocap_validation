import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# -----------------------
# paths
# -----------------------
root_dir = Path(r"D:\validation\joint_angles")

angle_summary = pd.read_csv(root_dir / "joint_angles_summary.csv")
spm_clusters  = pd.read_csv(root_dir / "spm_paired_ttest_clusters.csv")
spm_curves    = pd.read_csv(root_dir / "spm_paired_ttest_curves.csv")

# -----------------------
# normalize strings
# -----------------------
for df in (angle_summary, spm_clusters, spm_curves):
    for col in ["condition", "joint", "component", "tracker", "reference"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower()

# -----------------------
# helpers
# -----------------------
def speed_key(cond: str) -> float:
    m = re.search(r"speed_(\d+)[_\.](\d+)", str(cond))
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    m2 = re.search(r"speed_(\d+)", str(cond))
    if m2:
        return float(m2.group(1))
    return float("inf")

def speed_label(cond: str) -> str:
    k = speed_key(cond)
    return "?" if not np.isfinite(k) else f"{k:g} m/s"

def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"

def contiguous_true_runs(mask: np.ndarray):
    """Return list of (start_idx, end_idx) inclusive runs where mask is True."""
    if mask.size == 0:
        return []
    m = mask.astype(bool)
    d = np.diff(m.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends   = np.where(d == -1)[0]
    if m[0]:
        starts = np.r_[0, starts]
    if m[-1]:
        ends = np.r_[ends, m.size - 1]
    return list(zip(starts, ends))

def add_suprathreshold_fill(fig, *, x, y, ythr, color_rgba, row, col, showlegend, name):
    """
    Shade area between ythr and y wherever y > ythr (piecewise).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ythr = float(ythr)

    mask = np.isfinite(x) & np.isfinite(y) & (y > ythr)
    runs = contiguous_true_runs(mask)

    # legend proxy (once)
    if showlegend:
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=10, color=color_rgba),
                name=name,
                showlegend=True,
            ),
            row=row, col=col
        )

    for (i0, i1) in runs:
        xs = x[i0:i1+1]
        ys = y[i0:i1+1]
        ybase = np.full_like(ys, ythr, dtype=float)

        xp = np.r_[xs, xs[::-1]]
        yp = np.r_[ys, ybase[::-1]]

        fig.add_trace(
            go.Scatter(
                x=xp, y=yp,
                mode="lines",
                line=dict(width=0),
                fill="toself",
                fillcolor=color_rgba,
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row, col=col
        )
def add_subthreshold_fill(fig, *, x, y, ythr, color_rgba, row, col, showlegend, name):
    """
    Shade area between ythr and y wherever y < ythr (piecewise).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ythr = float(ythr)

    mask = np.isfinite(x) & np.isfinite(y) & (y < ythr)
    runs = contiguous_true_runs(mask)

    if showlegend:
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=10, color=color_rgba),
                name=name,
                showlegend=True,
            ),
            row=row, col=col
        )

    for (i0, i1) in runs:
        xs = x[i0:i1+1]
        ys = y[i0:i1+1]
        ybase = np.full_like(ys, ythr, dtype=float)

        xp = np.r_[xs, xs[::-1]]
        yp = np.r_[ys, ybase[::-1]]

        fig.add_trace(
            go.Scatter(
                x=xp, y=yp,
                mode="lines",
                line=dict(width=0),
                fill="toself",
                fillcolor=color_rgba,
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row, col=col
        )

# -----------------------
# plot config
# -----------------------
REFERENCE = "qualisys"
COMPARE   = ["mediapipe", "rtmpose", "vitpose"] 

COLORS = {
    "qualisys": "#313131",
    "mediapipe": "#0072B2",
    "rtmpose": "#D55E00",
    "vitpose": "#006D43",
}

JOINT_ORDER = ["hip", "knee", "ankle"]
COMPONENT_BY_JOINT = {"hip": "flex_ext", "knee": "flex_ext", "ankle": "dorsi_plantar"}
COMP_LABEL = {"flex_ext": "Flex/Ext", "dorsi_plantar": "Dorsi/Plantar"}

SPEEDS = sorted(angle_summary["condition"].unique().tolist(), key=speed_key)
n_cols = len(SPEEDS)

# -----------------------
# sizing (inches -> px)
# -----------------------
SUBPLOT_WIDTH_IN = 1.5
SUBPLOT_HEIGHT_IN = 1.5            # angle panels
SPM_HEIGHT_IN = SUBPLOT_HEIGHT_IN / 1.5
SPACER_HEIGHT_IN = 0.1            # controls "gap between joint blocks"

DPI = 100

MARGIN_LEFT_IN = 2.0
MARGIN_RIGHT_IN = 0.2
MARGIN_TOP_IN = 0.8
MARGIN_BOTTOM_IN = 1.15

# rows: (angle, spm, spacer) per joint, except no spacer after last
row_kinds = []
for j in range(len(JOINT_ORDER)):
    row_kinds += ["angle", "spm"]
    if j < len(JOINT_ORDER) - 1:
        row_kinds += ["spacer"]

n_rows = len(row_kinds)

FIG_WIDTH_IN  = MARGIN_LEFT_IN + n_cols * SUBPLOT_WIDTH_IN + MARGIN_RIGHT_IN
FIG_HEIGHT_IN = (
    MARGIN_TOP_IN
    + len(JOINT_ORDER) * (SUBPLOT_HEIGHT_IN + SPM_HEIGHT_IN)
    + (len(JOINT_ORDER) - 1) * SPACER_HEIGHT_IN
    + MARGIN_BOTTOM_IN
)

FIG_WIDTH_PX = int(FIG_WIDTH_IN * DPI)
FIG_HEIGHT_PX = int(FIG_HEIGHT_IN * DPI)

# row heights (relative)
row_heights = []
for kind in row_kinds:
    if kind == "angle":
        row_heights.append(SUBPLOT_HEIGHT_IN)
    elif kind == "spm":
        row_heights.append(SPM_HEIGHT_IN)
    else:
        row_heights.append(SPACER_HEIGHT_IN)
row_heights = (np.array(row_heights) / np.sum(row_heights)).tolist()

fig = make_subplots(
    rows=n_rows,
    cols=n_cols,
    shared_xaxes=True,
    shared_yaxes=False,
    vertical_spacing=0.02,          # keep overall tight; spacer rows create the real separation
    horizontal_spacing=0.02,
    column_titles=[speed_label(s) for s in SPEEDS],
    row_heights=row_heights,
)

# map joint -> (angle_row, spm_row)
joint_to_rows = {}
r = 1
for joint in JOINT_ORDER:
    angle_row = r
    spm_row = r + 1
    joint_to_rows[joint] = (angle_row, spm_row)
    r += 2
    if r <= n_rows and row_kinds[r-1] == "spacer":
        r += 1

# -----------------------
# traces
# -----------------------
for c, cond in enumerate(SPEEDS, start=1):
    for joint in JOINT_ORDER:
        comp = COMPONENT_BY_JOINT[joint]
        row_angles, row_spm = joint_to_rows[joint]

        # ---- ANGLES: mean ± SD
        sub = angle_summary[
            (angle_summary["condition"] == cond) &
            (angle_summary["joint"] == joint) &
            (angle_summary["component"] == comp)
        ].copy()

        for trk in [REFERENCE] + COMPARE:
            s = sub[sub["tracker"] == trk].sort_values("percent_gait_cycle")
            if s.empty:
                continue

            x = s["percent_gait_cycle"].to_numpy(float)
            m = s["mean_angle"].to_numpy(float)
            sd = s["std_angle"].to_numpy(float)
            if np.all(np.isnan(sd)):
                sd = np.zeros_like(m)

            fill_alpha = 0.12 if trk == REFERENCE else 0.18
            line_alpha = 0.70 if trk == REFERENCE else 0.90
            lw = 1.6 if trk == REFERENCE else 2.2

            fig.add_trace(
                go.Scatter(x=x, y=m+sd, mode="lines", line=dict(width=0),
                           showlegend=False, hoverinfo="skip"),
                row=row_angles, col=c
            )
            fig.add_trace(
                go.Scatter(
                    x=x, y=m-sd, mode="lines", line=dict(width=0),
                    fill="tonexty",
                    fillcolor=rgba(COLORS[trk], fill_alpha),
                    showlegend=False, hoverinfo="skip",
                ),
                row=row_angles, col=c
            )

            showleg = (row_angles == joint_to_rows["hip"][0] and c == 1)
            fig.add_trace(
                go.Scatter(
                    x=x, y=m, mode="lines",
                    line=dict(color=rgba(COLORS[trk], line_alpha), width=lw),
                    name=trk.capitalize(),
                    showlegend=showleg,
                    hovertemplate=f"{trk.capitalize()}<br>%{{x:.0f}}% GC<br>%{{y:.1f}}°<extra></extra>"
                ),
                row=row_angles, col=c
            )

        # ---- SPM: curves + suprathreshold fill + t*
        sub_curves = spm_curves[
            (spm_curves["condition"] == cond) &
            (spm_curves["joint"] == joint) &
            (spm_curves["component"] == comp) &
            (spm_curves["reference"] == REFERENCE) &
            (spm_curves["tracker"].isin(COMPARE))
        ].copy()

        if not sub_curves.empty:
            for trk in COMPARE:
                sc = sub_curves[sub_curves["tracker"] == trk].sort_values("percent_gait_cycle")
                if sc.empty:
                    continue

                x = sc["percent_gait_cycle"].to_numpy(float)
                z = sc["spm_t"].to_numpy(float)
                zstar = float(sc["t_star"].iloc[0])

                fig.add_trace(
                    go.Scatter(
                        x=x, y=z, mode="lines",
                        line=dict(color=rgba(COLORS[trk], 0.90), width=2),
                        name=f"{trk.capitalize()} SPM{{t}}",
                        showlegend=False,
                        hovertemplate=f"{trk.capitalize()}<br>%{{x:.0f}}% GC<br>SPM(t): %{{y:.2f}}<extra></extra>"
                    ),
                    row=row_spm, col=c
                )

                add_suprathreshold_fill(
                    fig,
                    x=x, y=z, ythr=zstar,
                    color_rgba=rgba(COLORS[trk], 0.18),
                    row=row_spm, col=c,
                    showlegend=False,
                    name=f"{trk.capitalize()} significant",
                )

                fig.add_hline(
                    y=zstar,
                    line=dict(color=rgba(COLORS[trk], 0.90), width=1.2, dash="dash"),
                    opacity=0.6,
                    row=row_spm, col=c
                )

                add_subthreshold_fill(
                fig,
                x=x, y=z, ythr=-zstar,
                color_rgba=rgba(COLORS[trk], 0.18),
                row=row_spm, col=c,
                showlegend=False,
                name=f"{trk.capitalize()} significant (neg)",
            )

                fig.add_hline(
                    y=-zstar,
                    line=dict(color=rgba(COLORS[trk], 0.90), width=1.2, dash="dash"),
                    opacity=0.6,
                    row=row_spm, col=c
                )


# -----------------------
# spacer rows: hide everything (axes & frames)
# -----------------------
for rr, kind in enumerate(row_kinds, start=1):
    if kind != "spacer":
        continue
    for cc in range(1, n_cols + 1):
        fig.update_xaxes(visible=False, row=rr, col=cc)
        fig.update_yaxes(visible=False, row=rr, col=cc)

# -----------------------
# Left-side row labels (centered per joint block)
# -----------------------
for joint in JOINT_ORDER:
    comp = COMPONENT_BY_JOINT[joint]
    label = COMP_LABEL[comp]
    row_angles, row_spm = joint_to_rows[joint]

    # center between angle+spm rows in paper coords (approx)
    y_center = 1 - (((row_angles - 0.5) + (row_spm - 0.5)) / (2 * n_rows))

    fig.add_annotation(
        x=-0.10, xref="paper",
        y=y_center, yref="paper",
        text=f"<b>{joint.title()}</b><br>{label} (°)",
        showarrow=False,
        xanchor="right",
        align="right",
        font=dict(size=12, color="#333"),
    )

# -----------------------
# Axes formatting
# - Only bottom row shows x tick labels + x title
# - All non-bottom non-spacer rows: hide x tick labels
# -----------------------
tickvals = list(range(0, 101, 20))

# bottom row = last *non-spacer* row
bottom_row = max(i for i, k in enumerate(row_kinds, start=1) if k != "spacer")

for rr, kind in enumerate(row_kinds, start=1):
    if kind == "spacer":
        continue
    for cc in range(1, n_cols + 1):
        show_xticks = (rr == bottom_row)

        fig.update_xaxes(
            tickvals=tickvals,
            tickfont=dict(size=9),
            showticklabels=show_xticks,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="#333",
            mirror=True,
            row=rr, col=cc
        )
        fig.update_yaxes(
            showticklabels=(cc == 1),
            tickfont=dict(size=9),
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="#333",
            mirror=True,
            row=rr, col=cc
        )

# x title on bottom row, all columns
for cc in range(1, n_cols + 1):
    fig.update_xaxes(
        title_text="<b>Gait cycle (%)</b>",
        title_font=dict(size=12, color="#333"),
        title_standoff=6,
        row=bottom_row, col=cc
    )

# y-axis titles (only first column) for angle/spm rows
for joint in JOINT_ORDER:
    row_angles, row_spm = joint_to_rows[joint]
    fig.update_yaxes(title_text="Angle (deg)", row=row_angles, col=1)
    fig.update_yaxes(title_text="SPM(t)",      row=row_spm, col=1)

# Bold column titles (speed labels)
for ann in fig.layout.annotations:
    if "m/s" in ann.text:
        ann.text = f"<b>{ann.text}</b>"
        ann.font = dict(size=11, color="#333")

# -----------------------
# Shared y-axes: compute range per joint for angle & SPM rows
# -----------------------
Y_PAD_FRAC = 0.05  # 5% padding on each side

for joint in JOINT_ORDER:
    comp = COMPONENT_BY_JOINT[joint]
    row_angles, row_spm = joint_to_rows[joint]

    # --- Angle range across all speeds ---
    sub_ang = angle_summary[
        (angle_summary["joint"] == joint) &
        (angle_summary["component"] == comp)
    ]
    if not sub_ang.empty:
        ang_lo = (sub_ang["mean_angle"] - sub_ang["std_angle"].fillna(0)).min()
        ang_hi = (sub_ang["mean_angle"] + sub_ang["std_angle"].fillna(0)).max()
        ang_pad = (ang_hi - ang_lo) * Y_PAD_FRAC
        ang_range = [ang_lo - ang_pad, ang_hi + ang_pad]

        for cc in range(1, n_cols + 1):
            fig.update_yaxes(range=ang_range, row=row_angles, col=cc)

    # --- SPM{t} range across all speeds ---
    sub_spm = spm_curves[
        (spm_curves["joint"] == joint) &
        (spm_curves["component"] == comp) &
        (spm_curves["reference"] == REFERENCE) &
        (spm_curves["tracker"].isin(COMPARE))
    ]
    if not sub_spm.empty:
        spm_lo = sub_spm["spm_t"].min()
        spm_hi = sub_spm["spm_t"].max()
        # also include t* thresholds so dashed lines aren't clipped
        tstar_max = sub_spm["t_star"].max()
        spm_lo = min(spm_lo, -tstar_max)
        spm_hi = max(spm_hi,  tstar_max)
        spm_hi = max(spm_hi, tstar_max)
        spm_pad = (spm_hi - spm_lo) * Y_PAD_FRAC
        max_abs = max(abs(spm_lo), abs(spm_hi))
        spm_range = [-max_abs, max_abs]

        for cc in range(1, n_cols + 1):
            fig.update_yaxes(range=spm_range, row=row_spm, col=cc)

# -----------------------
# Layout
# -----------------------
fig.update_layout(
    template="plotly_white",
    width=FIG_WIDTH_PX,
    height=FIG_HEIGHT_PX,
    margin=dict(
        l=int(MARGIN_LEFT_IN * DPI),
        r=int(MARGIN_RIGHT_IN * DPI),
        t=int(MARGIN_TOP_IN * DPI),
        b=int(MARGIN_BOTTOM_IN * DPI)
    ),
    title=dict(
        text="<b>Sagittal Plane Joint Angles Across Treadmill Speeds (Mean ± SD) with SPM{t}</b>",
        x=0.5, xanchor="center"
    ),
    legend=dict(
        orientation="h",
        x=0.5, y=-0.10,
        xanchor="center", yanchor="top",
        font=dict(size=11),
    ),
    paper_bgcolor="white",
    plot_bgcolor="white",
)

fig.add_annotation(
    text="SPM panels: dashed line = critical threshold (t*), shaded regions = significant (p < 0.05)",
    xref="paper", yref="paper",
    x=0.5, y=-0.07,
    xanchor="center", yanchor="top",
    showarrow=False,
    font=dict(size=12, color="#666"),
)


fig.show()


# Optional export
fig.write_image(str(root_dir / "joint_angles_with_spm.png"), scale=3)
