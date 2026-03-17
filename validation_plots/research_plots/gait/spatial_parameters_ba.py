import sqlite3
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots

conn = sqlite3.connect("validation.db")

query = """
SELECT t.participant_code,
        t.trial_name,
        a.path,
        a.component_name,
        a.condition,
        a.tracker
FROM artifacts a
JOIN trials t ON a.trial_id = t.id
WHERE t.trial_type = "treadmill"
    AND a.category = "gait_metrics"
    AND a.tracker IN ("mediapipe", "rtmpose", "vitpose", "qualisys")
    AND a.file_exists = 1
    AND a.component_name LIKE "%gait_metrics"
ORDER BY t.trial_name, a.path
"""

reference_system = "qualisys"
TRACKERS = ["mediapipe", "rtmpose", "vitpose"]

METRICS = [
    ("stride_length", "Stride Length"),
    ("step_length", "Step Length"),
]

TRACKER_LABELS = {
    "mediapipe": "MediaPipe",
    "rtmpose":   "RTMPose",
    "vitpose":   "ViTPose",
}

# Speed conditions — viridis sequential palette (perceptually uniform, colorblind-safe)
# Dark purple (slow) → teal → yellow-green (fast)
SPEED_ORDER = ["speed_0_5", "speed_1_0", "speed_1_5", "speed_2_0", "speed_2_5"]
SPEED_STYLE = {
    "speed_0_5": dict(label="0.5 m/s", color="#440154"),  # viridis 0.0
    "speed_1_0": dict(label="1.0 m/s", color="#3b528b"),  # viridis 0.25
    "speed_1_5": dict(label="1.5 m/s", color="#21918c"),  # viridis 0.5
    "speed_2_0": dict(label="2.0 m/s", color="#5ec962"),  # viridis 0.75
    "speed_2_5": dict(label="2.5 m/s", color="#fde725"),  # viridis 1.0
}

Y_SCALE = 1.0      # data already in mm
Y_UNITS = "mm"

# ------------------------------------------------------------------
# Load & pivot
# ------------------------------------------------------------------
path_df = pd.read_sql_query(query, conn)

dfs = []
for _, row in path_df.iterrows():
    sub = pd.read_csv(row["path"])
    sub["participant_code"] = row["participant_code"]
    sub["trial_name"] = row["trial_name"].lower()
    sub["condition"] = row["condition"] or "none"
    dfs.append(sub)

df = pd.concat(dfs, ignore_index=True)

id_cols = [
    "participant_code", "trial_name", "condition",
    "side", "metric", "event_index",
]

wide = (
    df.pivot_table(index=id_cols, columns="system", values="value", aggfunc="first")
    .reset_index()
)
wide = wide.rename(columns={reference_system: "reference_value"})

tracker_cols_present = [t for t in TRACKERS if t in wide.columns]

paired_df = wide.melt(
    id_vars=id_cols + ["reference_value"],
    value_vars=tracker_cols_present,
    var_name="tracker",
    value_name="tracker_value",
)

# Combine left/right — side is retained in the dataframe but not used for grouping
paired_df["ba_mean"] = (paired_df["tracker_value"] + paired_df["reference_value"]) / 2
paired_df["ba_diff"] = paired_df["tracker_value"] - paired_df["reference_value"]

# ------------------------------------------------------------------
# Quick outlier report
# ------------------------------------------------------------------
THRESH_MM = 50
spatial_metrics = [m[0] for m in METRICS]

out = paired_df.loc[
    (paired_df["ba_diff"].abs() * Y_SCALE > THRESH_MM) &
    (paired_df["metric"].isin(spatial_metrics))
].copy()
out["diff_mm"] = out["ba_diff"] * Y_SCALE
out["mean_mm"]  = out["ba_mean"]

if not out.empty:
    cols = ["participant_code","trial_name","condition","tracker","metric","side","event_index",
            "reference_value","tracker_value","mean_mm","diff_mm"]
    cols = [c for c in cols if c in out.columns]
    print("=== Outliers (|diff| > {} mm) ===".format(THRESH_MM))
    print(out[cols].sort_values("diff_mm", key=lambda s: s.abs(), ascending=False).to_string(index=False))
else:
    print(f"No outliers beyond {THRESH_MM} mm threshold.")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def inches_to_px(inches, dpi=300):
    return int(inches * dpi)


def add_bland_altman_subplot(
    fig, df_plot, *, row, col, ncols, phase_label,
    x_col="ba_mean", y_col="ba_diff",
    y_scale=1.0, y_units="mm",
    point_alpha=0.5, point_size=7,
    tracker_name="", show_legend=False,
):
    d = df_plot.copy()

    d["_x"] = d[x_col].astype(float)
    d["_y"] = d[y_col].astype(float) * y_scale

    # --- Statistics computed on ALL data ---
    bias = float(np.mean(d["_y"]))
    sd = float(np.std(d["_y"], ddof=1))
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd

    # axis index for this subplot
    ax_idx = (row - 1) * ncols + col
    xref = f"x{ax_idx} domain" if ax_idx > 1 else "x domain"
    yref = f"y{ax_idx}" if ax_idx > 1 else "y"

    line_kw = dict(xref=xref, x0=0, x1=1, yref=yref)
    fig.add_shape(type="line", y0=bias, y1=bias,
                  line=dict(color="rgba(0,0,0,0.5)", width=1, dash="dash"), **line_kw)
    fig.add_shape(type="line", y0=loa_upper, y1=loa_upper,
                  line=dict(color="black", width=1, dash="dashdot"), **line_kw)
    fig.add_shape(type="line", y0=loa_lower, y1=loa_lower,
                  line=dict(color="black", width=1, dash="dashdot"), **line_kw)

    # Plot each speed condition as a separate trace
    for spd in SPEED_ORDER:
        ds = d[d["condition"] == spd]
        if ds.empty:
            continue
        sty = SPEED_STYLE[spd]
        fig.add_trace(
            go.Scatter(
                x=ds["_x"], y=ds["_y"], mode="markers",
                name=sty["label"],
                legendgroup=spd,
                showlegend=show_legend,
                marker=dict(size=point_size, opacity=point_alpha, color=sty["color"],
                            line=dict(width=0.5, color="rgba(0,0,0,0.35)")),
            ),
            row=row, col=col,
        )

    print(
        f"[{phase_label} | {tracker_name}] "
        f"Bias={bias:+.2f} {y_units}, "
        f"Upper LoA={loa_upper:+.2f} {y_units}, "
        f"Lower LoA={loa_lower:+.2f} {y_units}"
    )


def style_paperish(fig, *, width_px, height_px):
    BASE, TICK = 14, 12
    fig.update_layout(
        template="simple_white",
        width=width_px, height=height_px,
        font=dict(family="Arial", size=BASE, color="black"),
        margin=dict(l=90, r=10, t=32, b=58),
    )
    fig.update_xaxes(
        tickfont=dict(size=TICK), title_font=dict(size=BASE),
        showline=True, linecolor="black", mirror=True,
        ticks="outside", ticklen=3, showgrid=False, zeroline=False,
    )
    fig.update_yaxes(
        tickfont=dict(size=TICK), title_font=dict(size=BASE),
        showline=True, linecolor="black", mirror=True,
        ticks="outside", ticklen=3, showgrid=False, zeroline=False,
    )
    return fig


# ------------------------------------------------------------------
# Build figure
# ------------------------------------------------------------------
nrows = len(TRACKERS)
ncols = len(METRICS)  # one BA panel per metric

subplot_titles = []
for r in range(nrows):
    for m_key, m_label in METRICS:
        subplot_titles.append(m_label if r == 0 else "")

fig = make_subplots(
    rows=nrows, cols=ncols,
    shared_yaxes=True,
    horizontal_spacing=0.08,
    vertical_spacing=0.07,
    subplot_titles=subplot_titles,
)

# ------------------------------------------------------------------
# Populate panels
# ------------------------------------------------------------------
all_y = []

for row_idx, tracker in enumerate(TRACKERS, start=1):
    tracker_label = TRACKER_LABELS.get(tracker, tracker)

    for met_idx, (m_key, m_label) in enumerate(METRICS):
        col = met_idx + 1

        df_met = paired_df.query("metric == @m_key and tracker == @tracker")
        if df_met.empty:
            continue

        all_y.append(df_met["ba_diff"].to_numpy() * Y_SCALE)

        # Show legend only on the first panel (row 1, col 1)
        is_first = (row_idx == 1 and col == 1)

        add_bland_altman_subplot(
            fig, df_met,
            row=row_idx, col=col, ncols=ncols,
            phase_label=m_label,
            y_scale=Y_SCALE, y_units=Y_UNITS,
            tracker_name=tracker_label,
            show_legend=is_first,
        )

# ------------------------------------------------------------------
# Y-range: symmetric autorange from data
# ------------------------------------------------------------------
y_all = np.concatenate(all_y) if all_y else np.array([0])
y_finite = y_all[np.isfinite(y_all)]
y_absmax = float(np.max(np.abs(y_finite))) if len(y_finite) > 0 else 10.0
y_pad = y_absmax * 1.05  # 5% padding

ylo = -y_pad
yhi =  y_pad

print(f"\n--- Y-axis range ---")
print(f"  [{ylo:.1f}, {yhi:.1f}] {Y_UNITS}  (symmetric around 0)")
print(f"  Total points: {len(y_finite)}")

for r in range(1, nrows + 1):
    for c in range(1, ncols + 1):
        fig.update_yaxes(range=[ylo, yhi], autorange=False, row=r, col=c)

# ------------------------------------------------------------------
# Axis labels & formatting
# ------------------------------------------------------------------
for row_idx, tracker in enumerate(TRACKERS, start=1):
    fig.update_yaxes(title_text=f"<b>Difference ({Y_UNITS})</b>", row=row_idx, col=1)

    for met_idx in range(len(METRICS)):
        col = met_idx + 1
        if row_idx == nrows:
            fig.update_xaxes(title_text="<b>Mean (mm)</b>", row=row_idx, col=col)
        else:
            fig.update_xaxes(title_text="", row=row_idx, col=col)

# Hide empty subplot title annotations, bold the real ones
for ann in list(fig.layout.annotations):
    if hasattr(ann, "text") and ann.text.strip() == "":
        ann.visible = False
    elif ann.text in [m[1] for m in METRICS]:
        ann.font = dict(size=14, weight="bold")

FIG_W_IN = 4.5
FIG_H_IN = 1.15 * nrows

style_paperish(fig, width_px=inches_to_px(FIG_W_IN), height_px=inches_to_px(FIG_H_IN))

fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="center", x=0.5,
        font=dict(size=11),
        tracegroupgap=2,
    ),
    margin=dict(l=90, r=10, t=55, b=58),
)

# Force shared y-range AFTER all styling
fig.update_yaxes(range=[ylo, yhi], autorange=False)

fig.show()

from pathlib import Path
save_root = Path(r"C:\Users\aaron\Documents\GitHub\dissertation\neu_coe_typst_starter\chapters\gait\figures")
save_root.mkdir(exist_ok=True, parents=True)

fig.write_image(save_root / "ba_gait_spatial.svg", scale=3)