import sqlite3
import pandas as pd
import plotly.graph_objects as go
import numpy as np 

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
    sub["participant_code"] = row['participant_code']
    sub["trial_name"] = row["trial_name"].lower()
    sub["condition"]  = row["condition"] or "none"
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

wide = (df.pivot_table(
    index = id_cols,
    columns = "system",
    values = "value",
    aggfunc="first"
    ).reset_index()
)

wide = wide.rename(columns = {reference_system: "reference_value"})

tracker_cols_present = [t for t in TRACKERS if t in wide.columns]

paired_df = wide.melt(
    id_vars = id_cols + ["reference_value"],
    value_vars = tracker_cols_present,
    var_name = "tracker",
    value_name = "tracker_value"
)

paired_df["ba_mean"] = (paired_df["tracker_value"] + paired_df["reference_value"]) / 2
paired_df["ba_diff"] = paired_df["tracker_value"] - paired_df["reference_value"]


group_cols = ["tracker", "metric",]

ba_summary = (
    paired_df
    .groupby(group_cols)
    .agg(
        n = ("ba_diff", "size"),
        bias = ("ba_diff", "mean"),
        sd = ("ba_diff", "std")
    )
    .reset_index()
)

ba_summary["loa_upper"] = ba_summary["bias"] + 1.96 * ba_summary["sd"]
ba_summary["loa_lower"] = ba_summary["bias"] - 1.96 * ba_summary["sd"]

def bland_altman_plot(
    df_plot,
    *,
    x_col="ba_mean",
    y_col="ba_diff",
    metric_label="Stance duration",
    tracker_label="mediapipe",
    width=900,
    height=550,
    y_units="ms",              # "ms" or "s"
    point_alpha=0.25,
    point_size=3,
    color_by=None,             # e.g. "condition" to color by speed
):
    """
    df_plot must contain columns: x_col, y_col (and optionally color_by).
    Assumes y_col is in seconds unless y_units="ms" converts it.
    """

    d = df_plot.copy()

    # Convert y to ms if desired (also converts bias/LoA)
    y_scale = 1000.0 if y_units == "ms" else 1.0
    y_suffix = " (ms)" if y_units == "ms" else " (s)"

    d["_x"] = d[x_col].astype(float)
    d["_y"] = d[y_col].astype(float) * y_scale

    bias = float(np.mean(d["_y"]))
    sd   = float(np.std(d["_y"], ddof=1))
    loa_upper = bias + 1.96 * sd
    loa_lower = bias - 1.96 * sd

    fig = go.Figure()

    # --- BA lines (as shapes so they look crisp) ---
    line_kw = dict(xref="paper", x0=0, x1=1)

    fig.add_shape(
        type="line",
        y0=bias, y1=bias,
        line=dict(color="rgba(0,0,0,0.5)", width=1, dash = "dash"),
        **line_kw
    )      
    fig.add_shape(type="line", y0=loa_upper, y1=loa_upper, line=dict(width=1, color="black", dash="dashdot"), **line_kw)
    fig.add_shape(type="line", y0=loa_lower, y1=loa_lower, line=dict(width=1, color="black", dash="dashdot"), **line_kw)

    # --- annotations (put them inside the plotting area) ---
    def fmt(v):
        return f"{v:+.1f} {y_units}"

    
    # --- points ---
    if color_by and color_by in d.columns:
        for grp, sub in d.groupby(color_by, sort=True):
            fig.add_trace(
                go.Scatter(
                    x=sub["_x"],
                    y=sub["_y"],
                    mode="markers",
                    name=str(grp),
                    marker=dict(
                    size=point_size,
                    opacity=point_alpha,
                    line=dict(
                        width=0.5,
                        color="rgba(0,0,0,0.35)"
                    ))
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=d["_x"],
                y=d["_y"],
                mode="markers",
                name="events",
                marker=dict(
                    size=point_size,
                    opacity=point_alpha,
                    line=dict(
                        width=0.5,
                        color="rgba(0,0,0,0.35)"
                    )),                
                    showlegend=False,
            )
        )

        
    fig.add_annotation(x=0.01, xref="paper", y=bias,      yref="y",
                       text=f"Bias: {fmt(bias)}", showarrow=False, xanchor="left", yanchor="bottom",
                       font=dict(size=12))
    fig.add_annotation(x=0.01, xref="paper", y=loa_upper, yref="y",
                       text=f"Upper LoA: {fmt(loa_upper)}", showarrow=False, xanchor="left", yanchor="bottom",
                       font=dict(size=12))
    fig.add_annotation(x=0.01, xref="paper", y=loa_lower, yref="y",
                       text=f"Lower LoA: {fmt(loa_lower)}", showarrow=False, xanchor="left", yanchor="bottom",
                       font=dict(size=12))

    # --- layout: paper style ---
    fig.update_layout(
        width=width,
        height=height,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        title_font=dict(size=16),
        font=dict(family = "Arial", size=14, color = "black"),
        margin=dict(l=80, r=30, t=60, b=65),
        legend=dict(
            title=color_by if color_by else None,
            orientation="v",
            x=1.02, xanchor="left",
            y=1.0,  yanchor="top",
            font=dict(size=12),
        ),
    )

    # Axis styling: “matplotlib-like box”
    fig.update_xaxes(
        title=f"Stance Duration Mean (s)",
        showline=True, linewidth=1.5, linecolor="black",
        mirror=True,
        ticks="outside", ticklen=6, tickwidth=1, tickcolor="black",
        showgrid=False, gridcolor="rgba(0,0,0,0.08)",
        zeroline=False,
    )
    fig.update_yaxes(
        title=f"Stance Duration Difference {y_suffix}",
        showline=True, linewidth=1.5, linecolor="black",
        mirror=True,
        ticks="outside", ticklen=6, tickwidth=1, tickcolor="black",
        showgrid=False, gridcolor="rgba(0,0,0,0.08)",
        zeroline=False, zerolinecolor="rgba(0,0,0,0.25)", zerolinewidth=1,
    )

    return fig

df_plot = paired_df.query(
    "metric == 'stance_duration' and tracker == 'mediapipe'"
)

def inches_to_px(inches, dpi=300):
    return int(inches * dpi)

fig = bland_altman_plot(
    df_plot,                         # already filtered to metric+tracker
    metric_label="Stance duration",
    tracker_label="mediapipe",
    height=inches_to_px(1.5),
    width =inches_to_px(2),
    y_units="ms",
    color_by=None,            # optional: show speed bins
)
fig.show()



f = 2
