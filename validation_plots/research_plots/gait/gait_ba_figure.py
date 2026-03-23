"""
Bland-Altman figure: Stride Length across three trackers, colored by speed.

Single-column figure for main results section.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from gait_ba_utils import (
    load_paired_gait_data, ba_stats,
    TRACKERS, TRACKER_LABELS,
    SPEED_ORDER, SPEED_STYLE,
    inches_to_px, style_paperish,
)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
METRIC_KEY = "stride_length"
METRIC_LABEL = "Stride Length"
Y_SCALE = 1.0      # data already in mm
Y_UNITS = "mm"
X_UNITS = "mm"

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
paired_df = load_paired_gait_data("validation.db")


# ------------------------------------------------------------------
# Build figure — 3 rows (trackers) × 1 col
# ------------------------------------------------------------------
nrows = len(TRACKERS)
ncols = 1

subplot_titles = [TRACKER_LABELS[t] for t in TRACKERS]

fig = make_subplots(
    rows=nrows, cols=ncols,
    shared_xaxes=True,
    shared_yaxes=True,
    vertical_spacing=0.06,
    subplot_titles=subplot_titles,
)

all_y = []

for row_idx, tracker in enumerate(TRACKERS, start=1):
    tracker_label = TRACKER_LABELS[tracker]

    df_met = paired_df.query("metric == @METRIC_KEY and tracker == @tracker")
    if df_met.empty:
        continue

    diffs_scaled = df_met["ba_diff"].to_numpy() * Y_SCALE
    means_scaled = df_met["ba_mean"].to_numpy()
    all_y.append(diffs_scaled)

    # Compute BA lines on ALL data
    stats = ba_stats(diffs_scaled)

    # Reference lines
    ax_idx = row_idx
    xref = f"x{ax_idx} domain" if ax_idx > 1 else "x domain"
    yref = f"y{ax_idx}" if ax_idx > 1 else "y"
    line_kw = dict(xref=xref, x0=0, x1=1, yref=yref)

    fig.add_shape(type="line", y0=stats["bias"], y1=stats["bias"],
                  line=dict(color="rgba(0,0,0,0.5)", width=1, dash="dash"), **line_kw)
    fig.add_shape(type="line", y0=stats["loa_upper"], y1=stats["loa_upper"],
                  line=dict(color="black", width=1, dash="dashdot"), **line_kw)
    fig.add_shape(type="line", y0=stats["loa_lower"], y1=stats["loa_lower"],
                  line=dict(color="black", width=1, dash="dashdot"), **line_kw)

    # Plot each speed as separate trace
    for spd in SPEED_ORDER:
        ds = df_met[df_met["condition"] == spd]
        if ds.empty:
            continue
        sty = SPEED_STYLE[spd]
        fig.add_trace(
            go.Scatter(
                x=ds["ba_mean"].values,
                y=ds["ba_diff"].values * Y_SCALE,
                mode="markers",
                name=sty["label"],
                legendgroup=spd,
                showlegend=(row_idx == 1),  # legend only from first row
                marker=dict(
                    size=7, opacity=0.5, color=sty["color"],
                    line=dict(width=0.5, color="rgba(0,0,0,0.35)"),
                ),
            ),
            row=row_idx, col=1,
        )

    print(
        f"[{tracker_label}] "
        f"Bias={stats['bias']:+.2f} {Y_UNITS}, "
        f"LoA=[{stats['loa_lower']:+.2f}, {stats['loa_upper']:+.2f}] {Y_UNITS}"
    )


# ------------------------------------------------------------------
# Y-range: symmetric from data
# ------------------------------------------------------------------
y_all = np.concatenate(all_y) if all_y else np.array([0])
y_finite = y_all[np.isfinite(y_all)]
y_absmax = float(np.max(np.abs(y_finite))) if len(y_finite) > 0 else 10.0
y_pad = y_absmax * 1.05

for r in range(1, nrows + 1):
    fig.update_yaxes(range=[-y_pad, y_pad], autorange=False, row=r, col=1)
    fig.update_yaxes(title_text=f"<b>Difference ({Y_UNITS})</b>", row=r, col=1)

# X-axis label only on bottom row
fig.update_xaxes(title_text=f"<b>Mean ({X_UNITS})</b>", row=nrows, col=1)

# Bold subplot titles
for ann in fig.layout.annotations:
    if hasattr(ann, "text") and ann.text in [TRACKER_LABELS[t] for t in TRACKERS]:
        ann.font = dict(size=14, weight="bold")

# ------------------------------------------------------------------
# Styling
# ------------------------------------------------------------------
FIG_W_IN = 2.0
FIG_H_IN = 1 * nrows

style_paperish(fig, width_px=inches_to_px(FIG_W_IN), height_px=inches_to_px(FIG_H_IN))

fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="top", y=-0.07,
        xanchor="center", x=0.5,
        font=dict(size=14),
        tracegroupgap=2,
    ),
    margin=dict(l=80, r=10, t=35, b=75),
)

# Force y-range after styling
fig.update_yaxes(range=[-y_pad, y_pad], autorange=False)

fig.show()

from pathlib import Path
save_root = Path(r"C:\Users\aaron\Documents\GitHub\dissertation\neu_coe_typst_starter\chapters\gait\figures")
save_root.mkdir(exist_ok=True, parents=True)
fig.write_image(save_root / "ba_stride_length.png", scale=3)
print(f"\nFigure saved to: {save_root / 'ba_stride_length.png'}")
