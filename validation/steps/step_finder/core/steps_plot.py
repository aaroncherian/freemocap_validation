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
