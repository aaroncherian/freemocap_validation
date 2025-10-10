import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_trajectory_cycles_grid(cycles: pd.DataFrame, marker_order=None):
    """
    Rows = Left/Right per joint (2 rows per joint)
    Cols = X/Y/Z
    Systems overlaid (solid colors), single legend.
    """
    # ---- checks ----
    required = {'marker','x','y','z','cycle','percent_gait_cycle','system'}
    missing = required - set(cycles.columns)
    if missing:
        raise ValueError(f"cycles DataFrame missing columns: {missing}")

    axes  = ['x','y','z']
    sides = ['left','right']

    all_markers = cycles['marker'].astype(str).unique().tolist()
    if marker_order is None:
        joints = sorted({m.replace('left_','').replace('right_','') for m in all_markers})
    else:
        joints = list(marker_order)

    n_rows = len(joints) * 2
    n_cols = len(axes)

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        shared_xaxes=True, shared_yaxes=False,
        vertical_spacing=0.025,
        horizontal_spacing=0.09,      # was 0.05 → more room between columns
        column_titles=[ax.upper() for ax in axes],
    )

    # ---- visuals: solid colors per system, single legend ----
    systems = list(cycles['system'].unique())
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]
    sys_color = {sys: palette[i % len(palette)] for i, sys in enumerate(systems)}

    def ax_key(kind, row, col):
        idx = (row - 1) * n_cols + col
        return f"{kind}axis" + ("" if idx == 1 else str(idx))

    # collect y-range per (joint,axis) to sync left/right
    y_minmax = {(j,a): [np.inf, -np.inf] for j in joints for a in axes}

    # ---- traces ----
    row_idx = 0
    for j_idx, joint in enumerate(joints):
        for side in sides:
            row_idx += 1
            marker_name = f"{side}_{joint}"
            if marker_name not in all_markers:
                continue
            df_m = cycles[cycles['marker'] == marker_name]

            for c, axis in enumerate(axes, start=1):
                if not df_m.empty:
                    vals = df_m[axis].to_numpy()
                    if vals.size:
                        y_minmax[(joint, axis)][0] = min(y_minmax[(joint, axis)][0], np.nanmin(vals))
                        y_minmax[(joint, axis)][1] = max(y_minmax[(joint, axis)][1], np.nanmax(vals))

                for sys in systems:
                    df_s = df_m[df_m['system'] == sys]
                    if df_s.empty:
                        continue
                    for cyc_id, df_cyc in df_s.groupby('cycle', sort=True):
                        fig.add_trace(
                            go.Scatter(
                                x=df_cyc['percent_gait_cycle'],
                                y=df_cyc[axis],
                                mode='lines',
                                line=dict(color=sys_color[sys], width=1.3),
                                opacity=0.55,
                                name=sys,
                                legendgroup=sys,
                                # single legend entry (your tip)
                                showlegend=(cyc_id == 1 and row_idx == 1 and c == 1),
                                hovertemplate=(
                                    f"{marker_name} | {axis}<br>"
                                    "system=%{meta}<br>"
                                    "cycle=%{customdata[0]}<br>"
                                    "pct=%{x}<br>"
                                    "value=%{y}<extra></extra>"
                                ),
                                meta=sys,
                                customdata=df_cyc[['cycle']].to_numpy(),
                            ),
                            row=row_idx, col=c
                        )

    # ---- sync y-ranges for left/right per joint/axis ----
    for j_idx, joint in enumerate(joints):
        left_row  = j_idx*2 + 1
        right_row = j_idx*2 + 2
        for c, axis in enumerate(axes, start=1):
            lo, hi = y_minmax[(joint, axis)]
            if np.isfinite(lo) and np.isfinite(hi):
                pad = (hi - lo) * 0.05 if hi > lo else 1.0
                rng = [lo - pad, hi + pad]
                fig.update_yaxes(range=rng, row=left_row, col=c)
                fig.update_yaxes(range=rng, row=right_row, col=c)

    # ---- ticks/labels ----
    tickvals = list(range(0, 101, 20))
    # X-ticks on every column’s bottom row
    for c in range(1, n_cols + 1):
        fig.update_xaxes(
            showticklabels=True,
            tickmode='array', tickvals=tickvals, ticks='outside', automargin=True,
            title=dict(text="Percent gait cycle", standoff=6),
            row=n_rows, col=c
        )
    # Show y-ticks on ALL subplots
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            fig.update_yaxes(showticklabels=True, ticks='outside', automargin=True, row=r, col=c)

    # ---- alternating row backgrounds (Left vs Right) ----
    fig.update_layout(margin=dict(l=120, r=40, t=70, b=90))  # a bit more left room
    for r in range(1, n_rows + 1):
        y0, y1 = fig.layout[ax_key("y", r, 1)].domain
        fill = "rgba(0,0,0,0.03)" if (r % 2 == 1) else "rgba(0,0,0,0.015)"
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=0.0, x1=1.0, y0=y0, y1=y1,
            layer="below", line=dict(width=0), fillcolor=fill
        )

    # ---- Left/Right labels like y-axis tags for EVERY subplot ----
    # place for each (row, col) just left of that subplot's y-axis
    side_text = {True: "Left", False: "Right"}
    for r in range(1, n_rows + 1):
        is_left_row = (r % 2 == 1)
        for c in range(1, n_cols + 1):
            xdom = fig.layout[ax_key("x", r, c)].domain
            ydom = fig.layout[ax_key("y", r, c)].domain
            x_pos = xdom[0] - 0.04   # small gutter before each subplot
            y_pos = 0.5 * (ydom[0] + ydom[1])
            fig.add_annotation(
                x=x_pos, xref="paper",
                y=y_pos, yref="paper",
                text=side_text[is_left_row],
                textangle=-90,
                showarrow=False,
                xanchor="right",  # hug the left edge
                yanchor="middle",
                font=dict(size=13, color="#444")
            )

    # ---- joint titles centered between Left/Right rows ----
    for j_idx, joint in enumerate(joints):
        top_row = j_idx*2 + 1
        bot_row = j_idx*2 + 2
        y_top = fig.layout[ax_key("y", top_row, 1)].domain[1]
        y_bot = fig.layout[ax_key("y", bot_row, 1)].domain[0]
        y_mid = 0.5 * (y_top + y_bot)
        fig.add_annotation(
            x=0.5, xref="paper",
            y=y_mid, yref="paper",
            text=f"<b>{joint.upper()}</b>",
            showarrow=False,
            xanchor="center", yanchor="middle",
            font=dict(size=13, color="#000"),
            bgcolor="rgba(255,255,255,0.96)", borderpad=2
        )

    for j_idx, joint in enumerate(joints[:-1]):  # skip after the last one
        bot_row = j_idx * 2 + 2
        y_bot = fig.layout[ax_key("y", bot_row, 1)].domain[0]

        fig.add_shape(
            type="line",
            x0=0.0, x1=1.0, xref="paper",
            y0=y_bot - 0.012, y1=y_bot - 0.012, yref="paper",
            line=dict(width=1, color="rgba(0,0,0,0.65)")
        )
    # ---- single Y-axis label on first column ----
    y_top = fig.layout[ax_key("y", 1, 1)].domain[1]
    y_bot = fig.layout[ax_key("y", n_rows, 1)].domain[0]
    y_center = (y_top + y_bot) / 2
    x_left = fig.layout[ax_key("x", 1, 1)].domain[0]
    fig.add_annotation(
        x=x_left - 0.070, xref="paper",
        y=y_center, yref="paper",
        text="<b>Position (mm)</b>",
        textangle=-90,
        showarrow=False,
        xanchor="center", yanchor="middle",
        font=dict(size=13, color="#000")
    )

    # ---- final layout ----
    fig.update_layout(
        height=max(440, 120 * n_rows),
        width=1120,
        template="plotly_white",
        title="Trajectory cycles per marker (FreeMoCap vs Qualisys)",
        legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1.0, orientation="h"),
    )
    fig.update_xaxes(showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=True, zeroline=False)

    return fig
