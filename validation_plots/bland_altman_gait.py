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
    AND a.tracker IN ("mediapipe", "qualisys")
    AND a.file_exists = 1
    AND a.component_name LIKE "%gait_metrics"
ORDER BY t.trial_name, a.path
"""

reference_system = "qualisys"
TRACKERS = ["mediapipe", "rtmpose", "vitpose"]

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
    "participant_code",
    "trial_name",
    "condition",
    "side",
    "metric",
    "event_index",
]

wide = (
    df.pivot_table(
        index=id_cols,
        columns="system",
        values="value",
        aggfunc="first",
    )
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


def inches_to_px(inches, dpi=300):
    return int(inches * dpi)


def add_bland_altman_subplot(
    fig,
    df_plot,
    *,
    row: int,
    col: int,
    phase_label: str,
    x_col="ba_mean",
    y_col="ba_diff",
    y_units="ms",
    point_alpha=0.4,
    point_size=8,
    color_by=None,
    show_legend=False,
):
    d = df_plot.copy()

    y_scale = 1000.0 if y_units == "ms" else 1.0
    y_suffix = " (ms)" if y_units == "ms" else " (s)"

    d["_x"] = d[x_col].astype(float)
    d["_y"] = d[y_col].astype(float) * y_scale

    bias = float(np.mean(d["_y"]))
    sd = float(np.std(d["_y"], ddof=1))
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd

    # keep lines local to subplot width
    xref = f"x{col} domain" if col > 1 else "x domain"
    # shared y-axis across all columns
    yref = "y"

    line_kw = dict(xref=xref, x0=0, x1=1, yref=yref)

    fig.add_shape(
        type="line",
        y0=bias, y1=bias,
        line=dict(color="rgba(0,0,0,0.5)", width=1, dash="dash"),
        **line_kw
    )
    fig.add_shape(
        type="line",
        y0=loa_upper, y1=loa_upper,
        line=dict(color="black", width=1, dash="dashdot"),
        **line_kw
    )
    fig.add_shape(
        type="line",
        y0=loa_lower, y1=loa_lower,
        line=dict(color="black", width=1, dash="dashdot"),
        **line_kw
    )

    def fmt(v):
        return f"{v:+.1f} {y_units}"

    # points
    fig.add_trace(
        go.Scatter(
            x=d["_x"],
            y=d["_y"],
            mode="markers",
            name=phase_label,
            marker=dict(
                size=point_size,
                opacity=point_alpha,
                line=dict(width=0.5, color="rgba(0,0,0,0.35)"),
            ),
            showlegend=False,
        ),
        row=row, col=col
    )

    xref = f"x{col} domain" if col > 1 else "x domain"  
    yref_dom = "y domain"  # row=1 only; shared y still OK for domain coords

    label_text = (
    f"Bias: {fmt(bias)}<br>"
    f"Upper LoA: {fmt(loa_upper)}<br>"
    f"Lower LoA: {fmt(loa_lower)}"
    )

    fig.add_annotation(
    x=0.98, xref=xref,              # right side of THIS subplot
    y=0.98, yref=yref_dom,          # near top of THIS subplot
    text=label_text,
    showarrow=False,
    xanchor="right",
    yanchor="top",
    align="left",
    font=dict(size=12),
    bgcolor="rgba(255,255,255,0.75)",  # “paper-ish” readability
    bordercolor="rgba(0,0,0,0.25)",
    borderwidth=0.5,
    )   
    
    fig.update_xaxes(
        title_text="<b>Phase duration mean (s)</b>",
        row=row, col=col
    )


def style_paperish(fig, *, width_px, height_px):
    BASE = 16
    TICK = 14
    TITLE = 16

    fig.update_layout(
        template="simple_white",
        width=width_px,
        height=height_px,
        font=dict(family="Arial", size=BASE, color="black"),
        margin=dict(l=58, r=8, t=28, b=62),
    )

    for ann in fig.layout.annotations:
        if ann.text in ("Stance", "Swing"):
            ann.font.size = TITLE
            ann.font.weight = "bold"

    fig.update_xaxes(
        tickfont=dict(size=TICK),
        title_font=dict(size=BASE),
        showline=True,
        linecolor="black",
        mirror=True,
        ticks="outside",
        ticklen=3,
        showgrid=False,
        zeroline=False,
    )
    fig.update_yaxes(
        tickfont=dict(size=TICK),
        title_font=dict(size=BASE),
        showline=True,
        linecolor="black",
        mirror=True,
        ticks="outside",
        ticklen=3,
        showgrid=False,
        zeroline=False,
    )
    return fig


# ------------------- Build the figure -------------------

df_stance = paired_df.query("metric == 'stance_duration' and tracker == 'mediapipe'")
df_swing  = paired_df.query("metric == 'swing_duration'  and tracker == 'mediapipe'")

FIG_W_IN = 3
FIG_H_IN = 1.5

fig = make_subplots(
    rows=1, cols=4,
    column_widths=[0.42, 0.08, 0.42, 0.08],
    shared_yaxes=True,
    horizontal_spacing=0.02,
    subplot_titles=["Stance Duration", "", "Swing Duration", ""],
)

# BA panels
add_bland_altman_subplot(fig, df_stance, row=1, col=1, phase_label="Stance duration", y_units="ms")
add_bland_altman_subplot(fig, df_swing,  row=1, col=3, phase_label="Swing duration",  y_units="ms")

# Marginal histograms (counts on x)
nbinsy = 25

fig.add_trace(
    go.Histogram(
        y=df_stance["ba_diff"] * 1000.0,
        nbinsy=nbinsy,
        marker_color="steelblue",
        opacity=0.45,
        marker_line_width=0,
        showlegend=False,
    ),
    row=1, col=2
)

fig.add_trace(
    go.Histogram(
        y=df_swing["ba_diff"] * 1000.0,
        nbinsy=nbinsy,
        marker_color="darkorange",
        opacity=0.45,
        marker_line_width=0,
        showlegend=False,
    ),
    row=1, col=4
)

for hist_col in [2, 4]:
    fig.update_xaxes(
        tickvals=[0, 500, 1000],
        tickangle=0,  # horizontal
        tickfont=dict(size=10),
        showline=True,
        linecolor="black",
        mirror=True,
        ticks="outside",
        ticklen=3,
        showgrid=False,
        row=1, col=hist_col
    )

# ------------------- Shared y-range + aligned bins -------------------

y_all = np.concatenate([
    (df_stance["ba_diff"].to_numpy() * 1000.0),
    (df_swing["ba_diff"].to_numpy()  * 1000.0),
])
ymin, ymax = float(np.min(y_all)), float(np.max(y_all))
pad = 0.08 * (ymax - ymin + 1e-9)
ylo, yhi = ymin - pad, ymax + pad

for c in [1, 2, 3, 4]:
    fig.update_yaxes(range=[ylo, yhi], row=1, col=c)

bin_size = (yhi - ylo) / nbinsy
fig.update_traces(
    selector=dict(type="histogram"),
    autobiny=False,
    ybins=dict(start=ylo, end=yhi, size=bin_size),
)

# ------------------- Make marginal panels informative -------------------
# Show "Count" on x-axis for histogram columns, hide their y tick labels

for hist_col in [2, 4]:
    fig.update_xaxes(
        title_text="<b>Count</b>",
        showticklabels=True,
        ticks="outside",
        ticklen=3,
        showline=True,
        linecolor="black",
        mirror=False,           # don't box it like a full panel
        showgrid=False,
        row=1, col=hist_col
    )
    fig.update_yaxes(
        showticklabels=False,   # y-axis already shown on BA panel
        ticks="",
        showline=False,
        mirror=False,
        showgrid=False,
        title_text="",
        row=1, col=hist_col
    )

# Make the bars read cleanly
fig.update_layout(bargap=0.15)

# Hide empty subplot titles
for ann in list(fig.layout.annotations):
    if ann.text.strip() == "":
        ann.visible = False

# Global styling
style_paperish(fig, width_px=inches_to_px(FIG_W_IN), height_px=inches_to_px(FIG_H_IN))
fig.update_yaxes(
    title_text="<b>Phase duration difference (ms)</b>",
    row=1, col=1
)

fig.show()
