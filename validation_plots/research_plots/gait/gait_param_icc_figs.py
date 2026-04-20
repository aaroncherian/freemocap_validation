"""
Generate paper-quality ICC figures for gait spatiotemporal parameters.

Figure 1: Pooled ICC (grouped bar chart) — one bar per tracker per metric, with 95% CI error bars
Figure 2: ICC by speed (line plot grid) — rows = metrics, columns = trackers, speed on x-axis

Aesthetic matched to trajectory RMSE grid figures.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────
TRACKERS = ["mediapipe", "rtmpose", "vitpose"]
TRACKER_LABELS = {
    "mediapipe": "MediaPipe",
    "rtmpose": "RTMPose",
    "vitpose": "ViTPose",
}
TRACKER_COLORS = {
    "mediapipe": "#0072B2",
    "rtmpose":   "#D55E00",
    "vitpose":   "#006D43",
}

METRIC_ORDER = ["Stride Length", "Step Length", "Stance Duration", "Swing Duration"]

SPEED_ORDER = ["speed_0_5", "speed_1_0", "speed_1_5", "speed_2_0", "speed_2_5"]
SPEED_LABELS = {
    "speed_0_5": "0.5", "speed_1_0": "1.0", "speed_1_5": "1.5",
    "speed_2_0": "2.0", "speed_2_5": "2.5",
}
SPEED_FLOATS = [0.5, 1.0, 1.5, 2.0, 2.5]


# ── Helpers ───────────────────────────────────────────────────────────
def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Figure sizing ─────────────────────────────────────────────────────
DPI = 100


# =====================================================================
# Figure 1: Pooled ICC — Grouped Bar Chart
# =====================================================================
def generate_pooled_icc_figure(
    pooled_rows: list[dict],
    title: str = "ICC (2,1) for spatiotemporal gait parameters — pooled across speeds",
    show: bool = True,
    save_path: str | Path | None = None,
) -> go.Figure:
    """
    Grouped bar chart: x = metric, grouped bars = tracker, y = ICC with 95% CI.
    """
    fig = go.Figure()

    for tracker in TRACKERS:
        label = TRACKER_LABELS[tracker]
        color = TRACKER_COLORS[tracker]

        tracker_rows = [r for r in pooled_rows if r["tracker"] == label]
        # Sort by METRIC_ORDER
        metric_map = {r["metric"]: r for r in tracker_rows}

        metrics = []
        iccs = []
        ci_lower = []
        ci_upper = []

        for m in METRIC_ORDER:
            if m in metric_map:
                r = metric_map[m]
                metrics.append(f"{m}<br>({r['units']})")
                iccs.append(r["icc"])
                ci_lower.append(r["icc"] - r["icc_lb"])
                ci_upper.append(r["icc_ub"] - r["icc"])
            else:
                metrics.append(m)
                iccs.append(np.nan)
                ci_lower.append(0)
                ci_upper.append(0)

        fig.add_trace(go.Bar(
            name=label,
            x=metrics,
            y=iccs,
            error_y=dict(
                type="data",
                array=ci_upper,
                arrayminus=ci_lower,
                visible=True,
                color=rgba(color, 0.6),
                thickness=1.5,
                width=4,
            ),
            marker=dict(
                color=rgba(color, 0.8),
                line=dict(width=0.5, color=color),
            ),
            hovertemplate=(
                f"<b>{label}</b><br>"
                "ICC: %{y:.3f}<br>"
                "<extra></extra>"
            ),
        ))

    # Reference lines for ICC interpretation
    for threshold, label_text, dash in [(0.75, "Good", "dot"), (0.90, "Excellent", "dash")]:
        fig.add_hline(
            y=threshold,
            line=dict(color="#999", width=1, dash=dash),
            annotation_text=label_text,
            annotation_position="top left",
            annotation_font=dict(size=10, color="#888"),
        )

    width_px = int(8.5 * DPI)
    height_px = int(4.5 * DPI)

    fig.update_layout(
        template="plotly_white",
        title=dict(text=f"<b>{title}</b>", x=0.5, font=dict(size=14, color="#333")),
        barmode="group",
        width=width_px,
        height=height_px,
        margin=dict(l=70, r=30, t=80, b=80),
        paper_bgcolor="white",
        plot_bgcolor="white",
        yaxis=dict(
            title=dict(text="<b>ICC (2,1)</b>", font=dict(size=12, color="#333")),
            range=[0, 1.05],
            dtick=0.1,
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="#333",
            mirror=True,
            tickfont=dict(size=11),
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            linecolor="#333",
            mirror=True,
            tickfont=dict(size=11),
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            font=dict(size=14),
        ),
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        fig.write_image(str(save_path.with_suffix(".svg")), scale=3)
        fig.write_image(str(save_path.with_suffix(".png")), scale=3)
        print(f"Saved: {save_path.with_suffix('.svg')}")
        print(f"Saved: {save_path.with_suffix('.png')}")

    if show:
        fig.show()

    return fig


# =====================================================================
# Figure 2: ICC by Speed — Line Plot Grid
# =====================================================================
def generate_speed_icc_figure(
    speed_rows: list[dict],
    title: str = "ICC (2,1) by walking speed",
    show: bool = True,
    save_path: str | Path | None = None,
) -> go.Figure:
    """
    4×3 grid: rows = metrics, columns = trackers.
    Each panel shows ICC vs speed with 95% CI error bars.
    Shared y-axis [0, 1.05] for direct comparison.
    """
    n_rows = len(METRIC_ORDER)
    n_cols = len(TRACKERS)

    SUBPLOT_WIDTH_IN = 2.2
    SUBPLOT_HEIGHT_IN = 1.6
    MARGIN_LEFT_IN = 1.8
    MARGIN_RIGHT_IN = 0.3
    MARGIN_TOP_IN = 0.7
    MARGIN_BOTTOM_IN = 0.8

    fig_w = int((MARGIN_LEFT_IN + n_cols * SUBPLOT_WIDTH_IN + MARGIN_RIGHT_IN) * DPI)
    fig_h = int((MARGIN_TOP_IN + n_rows * SUBPLOT_HEIGHT_IN + MARGIN_BOTTOM_IN) * DPI)

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_xaxes=False,
        shared_yaxes=True,
        vertical_spacing=0.06,
        horizontal_spacing=0.02,
        column_titles=[f"<b>{TRACKER_LABELS[t]}</b>" for t in TRACKERS],
    )

    for r_idx, metric in enumerate(METRIC_ORDER):
        row = r_idx + 1
        for c_idx, tracker in enumerate(TRACKERS):
            col = c_idx + 1
            label = TRACKER_LABELS[tracker]
            color = TRACKER_COLORS[tracker]

            # Filter rows for this metric + tracker
            subset = [
                r for r in speed_rows
                if r["metric"] == metric and r["tracker"] == label
            ]

            # Build arrays ordered by speed
            speed_map = {r["speed"]: r for r in subset}
            xs, ys, err_up, err_down = [], [], [], []

            for spd_key, spd_float in zip(SPEED_ORDER, SPEED_FLOATS):
                spd_label = SPEED_LABELS[spd_key]
                if spd_label in speed_map:
                    r = speed_map[spd_label]
                    xs.append(spd_float)
                    ys.append(r["icc"])
                    err_up.append(r["icc_ub"] - r["icc"])
                    err_down.append(r["icc"] - r["icc_lb"])

            if not xs:
                continue

            show_legend = (r_idx == 0 and c_idx == 0)

            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    error_y=dict(
                        type="data",
                        array=err_up,
                        arrayminus=err_down,
                        visible=True,
                        color=rgba(color, 0.45),
                        thickness=1.2,
                        width=3,
                    ),
                    mode="lines+markers",
                    name=label,
                    legendgroup=tracker,
                    showlegend=False,  # tracker names are column headers
                    line=dict(color=rgba(color, 0.85), width=2),
                    marker=dict(color=color, size=6, symbol="square", line=dict(width=0)),
                    hovertemplate=(
                        f"<b>{label} — {metric}</b><br>"
                        f"Speed: %{{x:.1f}} m/s<br>"
                        f"ICC: %{{y:.3f}}"
                        f"<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

    # ── Reference line at 0.75 ────────────────────────────────────
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            fig.add_shape(
                type="line",
                x0=0.3, x1=2.7,
                y0=0.75, y1=0.75,
                line=dict(color="#bbb", width=0.8, dash="dot"),
                row=r, col=c,
            )

    # ── Row labels ────────────────────────────────────────────────
    for r_idx, metric in enumerate(METRIC_ORDER):
        row = r_idx + 1
        fig.add_annotation(
            x=-0.09,
            xref="paper",
            y=1 - (row - 0.5) / n_rows,
            yref="paper",
            text=f"<b>{metric}</b>",
            showarrow=False,
            xanchor="right",
            font=dict(size=12, color="#333"),
            align="right",
        )

    # ── Centered ICC y-axis label ─────────────────────────────────
    fig.add_annotation(
        x=-0.14,
        xref="paper",
        y=0.5,
        yref="paper",
        text="<b>ICC (2,1)</b>",
        showarrow=False,
        textangle=-90,
        font=dict(size=12, color="#333"),
    )

    # ── Axis formatting ───────────────────────────────────────────
    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            is_bottom = (r == n_rows)
            is_left = (c == 1)

            fig.update_xaxes(
                title_text="<b>Speed (m/s)</b>" if is_bottom else None,
                title_font=dict(size=12, color="#333"),
                title_standoff=5,
                tickvals=SPEED_FLOATS,
                ticktext=[f"{s:g}" for s in SPEED_FLOATS],
                tickfont=dict(size=11),
                showticklabels=is_bottom,
                range=[0.3, 2.7],
                showgrid=False,
                zeroline=False,
                showline=True,
                linecolor="#333",
                mirror=True,
                row=r, col=c,
            )

            fig.update_yaxes(
                range=[0.25, 1.05],
                dtick=0.25,
                title_text=None,
                title_font=dict(size=12, color="#333"),
                showticklabels=is_left,
                tickfont=dict(size=11),
                showgrid=False,
                zeroline=False,
                showline=True,
                linecolor="#333",
                mirror=True,
                row=r, col=c,
            )

    # ── Layout ────────────────────────────────────────────────────
    fig.update_layout(
        template="plotly_white",
        title=dict(text=f"<b>{title}</b>", x=0.5, font=dict(size=14, color="#333")),
        width=fig_w,
        height=fig_h,
        margin=dict(
            l=int(MARGIN_LEFT_IN * DPI),
            r=int(MARGIN_RIGHT_IN * DPI),
            t=int(MARGIN_TOP_IN * DPI),
            b=int(MARGIN_BOTTOM_IN * DPI),
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    # Bold column titles
    for ann in fig.layout.annotations:
        if ann.text and any(t in ann.text for t in TRACKER_LABELS.values()):
            ann.font = dict(size=13, color="#333")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(exist_ok=True, parents=True)
        fig.write_image(str(save_path.with_suffix(".svg")), scale=3)
        fig.write_image(str(save_path.with_suffix(".png")), scale=3)
        print(f"Saved: {save_path.with_suffix('.svg')}")
        print(f"Saved: {save_path.with_suffix('.png')}")

    fig.show()

    return fig


# =====================================================================
# Direct execution
# =====================================================================
if __name__ == "__main__":
    from gait_ba_utils import (
        load_paired_gait_data, ba_stats, compute_icc_2_1,
        ALL_METRICS,
    )
    import pingouin as pg

    DB_PATH = "validation.db"
    FIGURE_OUT_DIR = Path(r"D:\validation\gait\figures")
    FIGURE_OUT_DIR.mkdir(exist_ok=True, parents=True)

    paired_df = load_paired_gait_data(DB_PATH)

    # ── Reuse compute_row from your BA script ─────────────────────
    def compute_row(df_slice, metric_key, units, scale, tracker):
        sub = df_slice.query("metric == @metric_key and tracker == @tracker").dropna(
            subset=["tracker_value", "reference_value"]
        )
        if sub.empty:
            return None

        diffs = sub["ba_diff"].values * scale
        stats = ba_stats(diffs)

        tracker_rows = sub[['participant_code', 'trial_name', 'condition', 'side', 'metric', 'event_index', 'tracker']].copy()
        tracker_rows['value'] = sub['tracker_value']
        ref_rows = sub[['participant_code', 'trial_name', 'condition', 'side', 'metric', 'event_index']].copy()
        ref_rows['tracker'] = 'qualisys'
        ref_rows['value'] = sub['reference_value']
        long_df = pd.concat([tracker_rows, ref_rows], ignore_index=True)
        long_df['target'] = (
            long_df['trial_name'] + '|' +
            long_df['condition'] + '|' +
            long_df['side'] + '|' +
            long_df['event_index'].astype(str)
        )
        icc_overall = pg.intraclass_corr(
            data=long_df, targets="target", raters="tracker", ratings="value"
        ).set_index("Type")
        icc_result = icc_overall.loc["ICC2", ["ICC", "CI95%"]]

        return dict(
            n=stats["n"],
            bias=stats["bias"],
            sd=stats["sd"],
            loa_lower=stats["loa_lower"],
            loa_upper=stats["loa_upper"],
            icc=icc_result["ICC"],
            icc_lb=icc_result["CI95%"][0],
            icc_ub=icc_result["CI95%"][1],
            units=units,
        )

    # ── Compute pooled stats ──────────────────────────────────────
    pooled_rows = []
    for m_key, m_label, m_units, m_scale in ALL_METRICS:
        for tracker in TRACKERS:
            row = compute_row(paired_df, m_key, m_units, m_scale, tracker)
            if row is None:
                continue
            row["metric"] = m_label
            row["tracker"] = TRACKER_LABELS[tracker]
            pooled_rows.append(row)

    generate_pooled_icc_figure(
        pooled_rows,
        save_path=FIGURE_OUT_DIR / "icc_pooled_gait",
    )

    # ── Compute by-speed stats ────────────────────────────────────
    speed_rows = []
    for m_key, m_label, m_units, m_scale in ALL_METRICS:
        for speed in SPEED_ORDER:
            df_speed = paired_df[paired_df["condition"] == speed]
            for tracker in TRACKERS:
                row = compute_row(df_speed, m_key, m_units, m_scale, tracker)
                if row is None:
                    continue
                row["metric"] = m_label
                row["speed"] = SPEED_LABELS[speed]
                row["tracker"] = TRACKER_LABELS[tracker]
                speed_rows.append(row)

    generate_speed_icc_figure(
        speed_rows,
        save_path=FIGURE_OUT_DIR / "icc_by_speed_gait",
    )

    print("\nDone!")