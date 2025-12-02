import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


def plot_gait_events_over_time_debug(
    # left foot
    q_left_hs: np.ndarray,
    q_left_to: np.ndarray,
    fmc_left_hs: np.ndarray,
    fmc_left_to: np.ndarray,
    # right foot
    q_right_hs: np.ndarray,
    q_right_to: np.ndarray,
    fmc_right_hs: np.ndarray,
    fmc_right_to: np.ndarray,
    sampling_rate: float,
    # optional FreeMoCap cluster flags
    fmc_left_hs_cluster_flags: np.ndarray | None = None,
    fmc_left_to_cluster_flags: np.ndarray | None = None,
    fmc_right_hs_cluster_flags: np.ndarray | None = None,
    fmc_right_to_cluster_flags: np.ndarray | None = None,
    title: str = "Gait events over time (debug, left & right)",
    xlim: tuple | None = None,
    separation: float = 0.03,
):
    """
    Debug plot:
      - Two subplots: row 1 = left foot, row 2 = right foot
      - Qualisys events in red
      - FreeMoCap events in blue
      - FreeMoCap events that belong to short-interval clusters in orange
    """

    frame_interval = 1.0 / sampling_rate

    # cast to arrays
    q_left_hs   = np.asarray(q_left_hs)
    q_left_to   = np.asarray(q_left_to)
    fmc_left_hs = np.asarray(fmc_left_hs)
    fmc_left_to = np.asarray(fmc_left_to)

    q_right_hs   = np.asarray(q_right_hs)
    q_right_to   = np.asarray(q_right_to)
    fmc_right_hs = np.asarray(fmc_right_hs)
    fmc_right_to = np.asarray(fmc_right_to)

    # colors (same as main plot)
    qual_color = "#d62728"   # red
    fmc_color  = "#1f77b4"   # blue
    cluster_color = "orange"

    def _foot_row(fig, row, q_hs, q_to, fmc_hs, fmc_to,
                  hs_cluster_flags, to_cluster_flags, foot_label: str):

        # y positions
        q_hs_y   = np.full(q_hs.shape, 0.3 + separation/2)
        fmc_hs_y = np.full(fmc_hs.shape, 0.3 - separation/2)
        q_to_y   = np.full(q_to.shape, 0.0 + separation/2)
        fmc_to_y = np.full(fmc_to.shape, 0.0 - separation/2)

        # times in seconds
        q_hs_t, q_to_t = q_hs * frame_interval, q_to * frame_interval
        fmc_hs_t, fmc_to_t = fmc_hs * frame_interval, fmc_to * frame_interval

        # --- Qualisys HS/TO (always single color) ---
        fig.add_trace(go.Scatter(
            x=q_hs_t, y=q_hs_y, mode="markers",
            marker=dict(symbol="x", size=8, color=qual_color),
            name="Qualisys HS" if row == 1 else "",
            showlegend=(row == 1),
        ), row=row, col=1)

        fig.add_trace(go.Scatter(
            x=q_to_t, y=q_to_y, mode="markers",
            marker=dict(symbol="circle-open", size=8, color=qual_color,
                        line=dict(width=1.5, color=qual_color)),
            name="Qualisys TO" if row == 1 else "",
            showlegend=(row == 1),
        ), row=row, col=1)

        # --- FreeMoCap HS/TO: normal vs clustered ---
        # default flags: all False
        hs_flags = np.zeros(fmc_hs.shape[0], dtype=bool)
        to_flags = np.zeros(fmc_to.shape[0], dtype=bool)

        if hs_cluster_flags is not None and hs_cluster_flags.shape == fmc_hs.shape:
            hs_flags = hs_cluster_flags.astype(bool)
        if to_cluster_flags is not None and to_cluster_flags.shape == fmc_to.shape:
            to_flags = to_cluster_flags.astype(bool)

        hs_normal = ~hs_flags
        to_normal = ~to_flags

        # FreeMoCap HS (normal)
        fig.add_trace(go.Scatter(
            x=fmc_hs_t[hs_normal],
            y=fmc_hs_y[hs_normal],
            mode="markers",
            marker=dict(symbol="x", size=8, color=fmc_color),
            name="FreeMoCap HS" if row == 1 else "",
            showlegend=(row == 1),
        ), row=row, col=1)

        # FreeMoCap HS (cluster)
        fig.add_trace(go.Scatter(
            x=fmc_hs_t[hs_flags],
            y=fmc_hs_y[hs_flags],
            mode="markers",
            marker=dict(symbol="x", size=10, color=cluster_color),
            name="FreeMoCap HS (cluster)" if row == 1 else "",
            showlegend=(row == 1),
        ), row=row, col=1)

        # FreeMoCap TO (normal)
        fig.add_trace(go.Scatter(
            x=fmc_to_t[to_normal],
            y=fmc_to_y[to_normal],
            mode="markers",
            marker=dict(symbol="circle-open", size=8, color=fmc_color,
                        line=dict(width=1.5, color=fmc_color)),
            name="FreeMoCap TO" if row == 1 else "",
            showlegend=(row == 1),
        ), row=row, col=1)

        # FreeMoCap TO (cluster)
        fig.add_trace(go.Scatter(
            x=fmc_to_t[to_flags],
            y=fmc_to_y[to_flags],
            mode="markers",
            marker=dict(symbol="circle-open", size=10, color=cluster_color,
                        line=dict(width=1.5, color=cluster_color)),
            name="FreeMoCap TO (cluster)" if row == 1 else "",
            showlegend=(row == 1),
        ), row=row, col=1)

        # y-axis formatting for this row
        fig.update_yaxes(
            row=row, col=1,
            tickmode="array",
            tickvals=[0.3, 0.0],
            ticktext=[f"{foot_label} Heel Strike", f"{foot_label} Toe Off"],
            range=[-0.3, 0.6],
            showgrid=False,
        )

    # --- create subplots ---
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Left foot", "Right foot"),
    )

    # row 1: left foot
    _foot_row(
        fig, row=1,
        q_hs=q_left_hs, q_to=q_left_to,
        fmc_hs=fmc_left_hs, fmc_to=fmc_left_to,
        hs_cluster_flags=fmc_left_hs_cluster_flags,
        to_cluster_flags=fmc_left_to_cluster_flags,
        foot_label="Left",
    )

    # row 2: right foot
    _foot_row(
        fig, row=2,
        q_hs=q_right_hs, q_to=q_right_to,
        fmc_hs=fmc_right_hs, fmc_to=fmc_right_to,
        hs_cluster_flags=fmc_right_hs_cluster_flags,
        to_cluster_flags=fmc_right_to_cluster_flags,
        foot_label="Right",
    )

    # x-axis and layout
    fig.update_xaxes(
        title="Time (seconds)",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.15)",
        row=2, col=1,  # only label bottom axis
    )
    if xlim:
        fig.update_xaxes(range=list(xlim), row=1, col=1)
        fig.update_xaxes(range=list(xlim), row=2, col=1)

    fig.update_layout(
        title=title,
        legend=dict(orientation="h", y=1.1, yanchor="bottom",
                    x=1.0, xanchor="right"),
        margin=dict(l=40, r=20, t=80, b=40),
        height=650,
    )

    return fig