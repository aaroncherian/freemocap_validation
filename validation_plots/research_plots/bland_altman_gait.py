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
    AND a.tracker IN ("mediapipe", "rtmpose", "qualisys")
    AND a.file_exists = 1
    AND a.component_name LIKE "%gait_metrics"
ORDER BY t.trial_name, a.path
"""

reference_system = "qualisys"
TRACKERS = ["mediapipe", "rtmpose"]

METRICS = [
    ("stance_duration", "Stance Duration"),
    ("swing_duration", "Swing Duration"),
]

TRACKER_STYLE = {
    "mediapipe": dict(label="MediaPipe", color="steelblue"),
    "rtmpose":   dict(label="RTMPose",   color="darkorange"),
    "vitpose":   dict(label="ViTPose",   color="seagreen"),
}

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

paired_df["ba_mean"] = (paired_df["tracker_value"] + paired_df["reference_value"]) / 2
paired_df["ba_diff"] = paired_df["tracker_value"] - paired_df["reference_value"]

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def inches_to_px(inches, dpi=300):
    return int(inches * dpi)


def add_bland_altman_subplot(
    fig, df_plot, *, row, col, ncols, phase_label,
    x_col="ba_mean", y_col="ba_diff", y_units="ms",
    point_alpha=0.4, point_size=8, marker_color="steelblue",
):
    d = df_plot.copy()
    y_scale = 1000.0 if y_units == "ms" else 1.0

    d["_x"] = d[x_col].astype(float)
    d["_y"] = d[y_col].astype(float) * y_scale

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

    fig.add_trace(
        go.Scatter(
            x=d["_x"], y=d["_y"], mode="markers", name=phase_label,
            marker=dict(size=point_size, opacity=point_alpha, color=marker_color,
                        line=dict(width=0.5, color="rgba(0,0,0,0.35)")),
            showlegend=False,
        ),
        row=row, col=col,
    )

    def fmt(v):
        return f"{v:+.1f} {y_units}"

    xref_dom = f"x{ax_idx} domain" if ax_idx > 1 else "x domain"
    yref_dom = f"y{ax_idx} domain" if ax_idx > 1 else "y domain"

    # fig.add_annotation(
    #     x=0.02, xref=xref_dom,          # <-- left
    #     y=0.98, yref=yref_dom,          # <-- top
    #     text=f"Bias: {fmt(bias)}<br>Upper LoA: {fmt(loa_upper)}<br>Lower LoA: {fmt(loa_lower)}",
    #     showarrow=False,
    #     xanchor="left", yanchor="top",  # <-- anchor to upper-left
    #     align="left",
    #     font=dict(size=10),
    #     bgcolor="rgba(255,255,255,0.75)",
    #     bordercolor="rgba(0,0,0,0.25)", borderwidth=0.5,
    # )
    print(
    f"[{phase_label} | tracker={tracker}] "
    f"Bias={bias:+.1f} ms, "
    f"Upper LoA={loa_upper:+.1f} ms, "
    f"Lower LoA={loa_lower:+.1f} ms"
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
ncols = 4  # [BA_stance, hist_stance, BA_swing, hist_swing]

subplot_titles = []
for r in range(nrows):
    for m_key, m_label in METRICS:
        subplot_titles.append(m_label if r == 0 else "")
        subplot_titles.append("")

fig = make_subplots(
    rows=nrows, cols=ncols,
    column_widths=[0.42, 0.08, 0.42, 0.08],
    shared_yaxes=True,
    horizontal_spacing=0.02,  # placeholder, overridden below
    vertical_spacing=0.07,
    subplot_titles=subplot_titles,
)

# ------------------------------------------------------------------
# Override x-axis domains for asymmetric spacing:
#   small gap between BA and its histogram,
#   larger gap between the two metric groups
# ------------------------------------------------------------------
BA_W = 0.38       # width of each BA panel
HIST_W = 0.065    # width of each histogram panel
INNER_GAP = 0.01  # gap between BA and its histogram
OUTER_GAP = 0.06  # gap between stance group and swing group

# Stance group
s_ba_x0 = 0.0
s_ba_x1 = s_ba_x0 + BA_W
s_hi_x0 = s_ba_x1 + INNER_GAP
s_hi_x1 = s_hi_x0 + HIST_W

# Swing group
w_ba_x0 = s_hi_x1 + OUTER_GAP
w_ba_x1 = w_ba_x0 + BA_W
w_hi_x0 = w_ba_x1 + INNER_GAP
w_hi_x1 = w_hi_x0 + HIST_W

col_domains = [
    (s_ba_x0, s_ba_x1),
    (s_hi_x0, s_hi_x1),
    (w_ba_x0, w_ba_x1),
    (w_hi_x0, w_hi_x1),
]

for r in range(nrows):
    for c in range(ncols):
        ax_idx = r * ncols + c + 1
        ax_name = f"xaxis{ax_idx}" if ax_idx > 1 else "xaxis"
        fig.layout[ax_name].domain = col_domains[c]

# (subplot title repositioning done at the end)

# ------------------------------------------------------------------
# Populate panels
# ------------------------------------------------------------------
nbinsy = 25
all_y = []

for row_idx, tracker in enumerate(TRACKERS, start=1):
    style = TRACKER_STYLE.get(tracker, dict(label=tracker, color="gray"))

    for met_idx, (m_key, m_label) in enumerate(METRICS):
        ba_col = met_idx * 2 + 1
        hist_col = met_idx * 2 + 2

        df_met = paired_df.query("metric == @m_key and tracker == @tracker")
        if df_met.empty:
            continue

        all_y.append(df_met["ba_diff"].to_numpy() * 1000.0)

        add_bland_altman_subplot(
            fig, df_met,
            row=row_idx, col=ba_col, ncols=ncols,
            phase_label=m_label, y_units="ms",
            marker_color=style["color"],
        )

        fig.add_trace(
            go.Histogram(
                y=df_met["ba_diff"] * 1000.0,
                nbinsy=nbinsy,
                marker_color=style["color"],
                opacity=0.45,
                marker_line_width=0,
                showlegend=False,
            ),
            row=row_idx, col=hist_col,
        )

# ------------------------------------------------------------------
# Shared y-range + aligned bins
# ------------------------------------------------------------------
y_all = np.concatenate(all_y) if all_y else np.array([0])
ymin, ymax = float(np.min(y_all)), float(np.max(y_all))
pad = 0.08 * (ymax - ymin + 1e-9)
ylo, yhi = ymin - pad, ymax + pad
bin_size = (yhi - ylo) / nbinsy

for r in range(1, nrows + 1):
    for c in range(1, ncols + 1):
        fig.update_yaxes(range=[ylo, yhi], row=r, col=c)

fig.update_traces(
    selector=dict(type="histogram"),
    autobiny=False,
    ybins=dict(start=ylo, end=yhi, size=bin_size),
)
BA_COLS = [1, 3]

for r in range(1, nrows + 1):
    for c in BA_COLS:
        fig.update_yaxes(matches="y", row=r, col=c)

# ------------------------------------------------------------------
# Axis labels, row labels, formatting
# ------------------------------------------------------------------
for row_idx, tracker in enumerate(TRACKERS, start=1):
    style = TRACKER_STYLE.get(tracker, dict(label=tracker, color="gray"))

    fig.update_yaxes(title_text="<b>Difference (ms)</b>", row=row_idx, col=1)

    for met_idx in range(len(METRICS)):
        ba_col = met_idx * 2 + 1
        hist_col = met_idx * 2 + 2

        # x-axis titles only on bottom row
        if row_idx == nrows:
            fig.update_xaxes(title_text="<b>Mean (s)</b>", row=row_idx, col=ba_col)
            fig.update_xaxes(title_text="<b>Count</b>", row=row_idx, col=hist_col)
        else:
            fig.update_xaxes(title_text="", row=row_idx, col=ba_col)
            fig.update_xaxes(title_text="", row=row_idx, col=hist_col)

        # Histogram column formatting
        fig.update_xaxes(
            showticklabels=(row_idx == nrows),
            tickfont=dict(size=10),
            showline=True, linecolor="black", mirror=False,
            ticks="outside", ticklen=3, showgrid=False,
            row=row_idx, col=hist_col,
        )
        fig.update_yaxes(
            showticklabels=False, ticks="", showline=False,
            mirror=False, showgrid=False, title_text="",
            row=row_idx, col=hist_col,
        )

# Hide empty subplot title annotations, bold the real ones
for ann in list(fig.layout.annotations):
    if hasattr(ann, "text") and ann.text.strip() == "":
        ann.visible = False
    elif ann.text in [m[1] for m in METRICS]:
        ann.font = dict(size=14, weight="bold")

# Reposition subplot title annotations to center over BA panels.
# make_subplots creates nrows*ncols title annotations in row-major order.
ba_centers = [(s_ba_x0 + s_ba_x1) / 2, None, (w_ba_x0 + w_ba_x1) / 2, None]
n_subplot_titles = nrows * ncols
for i in range(min(n_subplot_titles, len(fig.layout.annotations))):
    col_in_row = i % ncols
    if ba_centers[col_in_row] is not None:
        fig.layout.annotations[i].x = ba_centers[col_in_row]

fig.update_layout(bargap=0.15)

FIG_W_IN = 4
FIG_H_IN = 1.15 * nrows

style_paperish(fig, width_px=inches_to_px(FIG_W_IN), height_px=inches_to_px(FIG_H_IN))

# Force shared y-range AFTER all styling so nothing overrides it
fig.update_yaxes(range=[ylo, yhi], autorange=False)

fig.show()

from pathlib import Path
save_root = Path(r"D:/validation/gait_parameters")
save_root.mkdir(exist_ok=True, parents=True)

fig.write_image(save_root / "ba_gait_phases.png", scale=3)