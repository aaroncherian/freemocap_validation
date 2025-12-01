import numpy as np
import plotly.graph_objects as go

import numpy as np
import plotly.graph_objects as go

def plot_gait_events_over_time(
    q_hs: np.ndarray,
    q_to: np.ndarray,
    fmc_hs: np.ndarray,
    fmc_to: np.ndarray,
    sampling_rate: float,
    title: str = "Gait events over time (one foot)",
    xlim: tuple | None = None,
    separation: float = 0.03,
):
    frame_interval = 1.0 / sampling_rate
    q_hs, q_to = np.asarray(q_hs), np.asarray(q_to)
    fmc_hs, fmc_to = np.asarray(fmc_hs), np.asarray(fmc_to)

    # consistent colors
    qual_color = "#d62728"   # red
    fmc_color  = "#1f77b4"   # blue

    # y positions
    q_hs_y   = np.full(q_hs.shape, 0.3 + separation/2)
    fmc_hs_y = np.full(fmc_hs.shape, 0.3 - separation/2)
    q_to_y   = np.full(q_to.shape, 0.0 + separation/2)
    fmc_to_y = np.full(fmc_to.shape, 0.0 - separation/2)

    # convert to seconds
    q_hs_t, q_to_t = q_hs * frame_interval, q_to * frame_interval
    fmc_hs_t, fmc_to_t = fmc_hs * frame_interval, fmc_to * frame_interval

    fig = go.Figure()

    # heel strikes
    fig.add_trace(go.Scatter(
        x=q_hs_t, y=q_hs_y, mode="markers",
        marker=dict(symbol="x", size=8, color=qual_color),
        name="Qualisys HS"
    ))
    fig.add_trace(go.Scatter(
        x=fmc_hs_t, y=fmc_hs_y, mode="markers",
        marker=dict(symbol="x", size=8, color=fmc_color),
        name="FreeMoCap HS"
    ))

    # toe offs – same color as each system’s HS
    fig.add_trace(go.Scatter(
        x=q_to_t, y=q_to_y, mode="markers",
        marker=dict(symbol="circle-open", size=8, color=qual_color,
                    line=dict(width=1.5, color=qual_color)),
        name="Qualisys TO"
    ))
    fig.add_trace(go.Scatter(
        x=fmc_to_t, y=fmc_to_y, mode="markers",
        marker=dict(symbol="circle-open", size=8, color = fmc_color,
                    line=dict(width=1.5, color=fmc_color)),
        name="FreeMoCap TO"
    ))

    # axes / layout
    fig.update_yaxes(
        tickmode="array",
        tickvals=[0.3, 0.0],
        ticktext=["Heel Strike", "Toe Off"],
        range=[-0.3, 0.6],
        showgrid=False
    )
    fig.update_xaxes(title="Time (seconds)", showgrid=True, gridcolor="rgba(0,0,0,0.15)")
    if xlim:
        fig.update_xaxes(range=list(xlim))

    fig.update_layout(
        title=title,
        legend=dict(orientation="h", y=1.08, yanchor="bottom", x=1.0, xanchor="right"),
        margin=dict(l=40, r=20, t=60, b=40),
        height=420
    )

    return fig
def plot_gait_event_diagnostics(
    heel_pos: np.ndarray,
    toe_pos: np.ndarray,
    heel_strikes: np.ndarray,
    toe_offs: np.ndarray,
    sampling_rate: float,
    title: str = "Gait event diagnostics (position)",
    hs_short_cluster_flags: np.ndarray | None = None,
    to_short_cluster_flags: np.ndarray | None = None,
):
    n = heel_pos.shape[0]
    t = np.arange(n) / sampling_rate

    z_heel = heel_pos[:, 2]
    z_toe = toe_pos[:, 2]

    hs_in_cluster = np.zeros(heel_strikes.size, dtype=bool)
    if hs_short_cluster_flags is not None and heel_strikes.size == hs_short_cluster_flags.size:
        hs_in_cluster = hs_short_cluster_flags

    to_in_cluster = np.zeros(toe_offs.size, dtype=bool)
    if to_short_cluster_flags is not None and toe_offs.size == to_short_cluster_flags.size:
        to_in_cluster = to_short_cluster_flags

    fig = go.Figure()

    # base lines
    fig.add_trace(go.Scatter(x=t, y=z_heel, mode="lines", name="Heel z"))
    fig.add_trace(go.Scatter(x=t, y=z_toe, mode="lines", name="Toe z"))

    # HS: normal vs cluster
    if heel_strikes.size:
        normal_mask = ~hs_in_cluster
        fig.add_trace(go.Scatter(
            x=t[heel_strikes[normal_mask]],
            y=z_heel[heel_strikes[normal_mask]],
            mode="markers",
            name="HS (normal)",
            marker=dict(color="red", symbol="circle-open", size=9),
        ))
        fig.add_trace(go.Scatter(
            x=t[heel_strikes[hs_in_cluster]],
            y=z_heel[heel_strikes[hs_in_cluster]],
            mode="markers",
            name="HS (short-interval cluster)",
            marker=dict(color="orange", symbol="circle", size=11),
        ))

    # TO: normal vs cluster
    if toe_offs.size:
        normal_mask = ~to_in_cluster
        fig.add_trace(go.Scatter(
            x=t[toe_offs[normal_mask]],
            y=z_toe[toe_offs[normal_mask]],
            mode="markers",
            name="TO (normal)",
            marker=dict(color="blue", symbol="x", size=9),
        ))
        fig.add_trace(go.Scatter(
            x=t[toe_offs[to_in_cluster]],
            y=z_toe[toe_offs[to_in_cluster]],
            mode="markers",
            name="TO (short-interval cluster)",
            marker=dict(color="orange", symbol="x", size=12),
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Vertical position",
        width=1200,
        height=400,
    )

    return fig
