from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ----------------------- Inputs -----------------------

recordings = {
    "neg_5": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_38_16_TF01_leg_length_neg_5_trial_1"),
    "neg_25": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_43_15_TF01_leg_length_neg_25_trial_1"),
    "neutral": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_46_54_TF01_leg_length_neutral_trial_1"),
    "pos_25": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_50_56_TF01_leg_length_pos_25_trial_1"),
    "pos_5": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_55_21_TF01_leg_length_pos_5_trial_1"),
}

trackers = ["qualisys", "mediapipe_dlc", "rtmpose"]  # column order: left then right

# Plotting config (match your style)
COND_ORDER = ["neg_5", "neg_25", "neutral", "pos_25", "pos_5"]
COND_COLORS = {
    "neg_5": "#94342b",
    "neg_25": "#d39182",
    "neutral": "#524F4F",
    "pos_25": "#7bb6c6",
    "pos_5": "#447c8e",
}
COND_LABELS = {
    "neutral": "Neutral",
    "neg_5": "-0.5",
    "neg_25": "-0.25",
    "pos_25": "+0.25",
    "pos_5": "+0.5",
}

SYSTEM_LABELS = {
    "rtmpose": "RTMPose",
    "qualisys": "Qualisys",
    "mediapipe_dlc": "Mediapipe",
}

ERRORBAR_STEP = 10
MAX_JITTER = 1.0

JOINT = "pelvis"
COMPONENT = "obliquity"
SIDE_PREFERENCE = ("mid", "right", "left")  # choose whichever exists in file


# -------------------- Helpers --------------------

def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def load_stride_summary_csv(recording: Path, tracker: str) -> pd.DataFrame:
    csv_path = recording / "validation" / tracker / "joint_angles" / "joint_angles_per_stride_summary_stats.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing: {csv_path}")
    return pd.read_csv(csv_path)


def filter_pelvis_obliquity(df: pd.DataFrame) -> pd.DataFrame:
    # Filter down to pelvis/obliquity
    df = df[(df["joint"] == JOINT) & (df["component"] == COMPONENT)].copy()

    # Prefer a side if present (some pipelines store pelvis as right, some as mid)
    if "side" in df.columns and not df.empty:
        sides = [s for s in df["side"].dropna().unique().tolist()]
        chosen = None
        for s in SIDE_PREFERENCE:
            if s in sides:
                chosen = s
                break
        if chosen is not None:
            df = df[df["side"] == chosen]

    return df


def build_pelvis_summary(recordings: dict[str, Path], tracker: str) -> pd.DataFrame:
    """
    Returns concatenated wide DF with columns:
      ['system','condition','percent_gait_cycle','mean','std']
    """
    rows = []
    for cond, rec_path in recordings.items():
        df = load_stride_summary_csv(rec_path, tracker)
        df = filter_pelvis_obliquity(df)
        if df.empty:
            raise ValueError(f"No pelvis obliquity rows for {cond} / {tracker}")

        wide = (
            df.pivot(index="percent_gait_cycle", columns="stat", values="value")
              .reset_index()
        )
        if not {"mean", "std"}.issubset(wide.columns):
            raise ValueError(
                f"Expected stats mean+std for {cond}/{tracker}. Got: {wide.columns.tolist()}"
            )

        wide["system"] = tracker
        wide["condition"] = cond
        rows.append(wide[["system", "condition", "percent_gait_cycle", "mean", "std"]])

    return pd.concat(rows, ignore_index=True)


def make_pelvis_system_comparison_figure(
    summary: pd.DataFrame,
    trackers: list[str],
    errorbar_step: int = ERRORBAR_STEP,
    max_jitter: float = MAX_JITTER,
) -> go.Figure:
    """
    Two-column figure: tracker | Qualisys side-by-side.
    Mean curves for each condition + jittered SD error bars.
    """

    # --- match your FPA sizing ---
    FIG_W_IN = 1.5
    FIG_H_IN = 1.5
    DPI = 300
    W = int(FIG_W_IN * DPI)
    H = int(FIG_H_IN * DPI)

    BASE = 16
    TICK = 14
    LEG = 14
    TITLE = 14

    subplot_titles = [SYSTEM_LABELS.get(s, s) for s in trackers]

    fig = make_subplots(
        rows=1,
        cols=len(trackers),
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.05,
    )

    # Jitter offsets per condition
    n_cond = len(COND_ORDER)
    offsets = np.linspace(-max_jitter, max_jitter, n_cond) if n_cond > 1 else np.array([0.0])
    cond_offset = dict(zip(COND_ORDER, offsets))

    # Shared y-range across all systems/conditions
    ymin = (summary["mean"] - summary["std"]).min()
    ymax = (summary["mean"] + summary["std"]).max()
    pad = 0.08 * (ymax - ymin + 1e-9)
    ylo, yhi = float(ymin - pad), float(ymax + pad)

    for col_idx, system in enumerate(trackers, 1):
        # zero reference line (often nice for obliquity)
        fig.add_hline(
            y=0,
            line=dict(color="gray", width=0.75, dash="dot"),
            row=1,
            col=col_idx,
        )

        for condition in COND_ORDER:
            sub = summary[(summary["system"] == system) & (summary["condition"] == condition)].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("percent_gait_cycle")

            x_all = sub["percent_gait_cycle"].to_numpy()
            m_all = sub["mean"].to_numpy()
            sd_all = sub["std"].to_numpy()

            display_label = COND_LABELS.get(condition, condition)
            system_label = SYSTEM_LABELS.get(system, system)

            # mean line
            fig.add_trace(
                go.Scatter(
                    x=x_all,
                    y=m_all,
                    mode="lines",
                    name=display_label if col_idx == 1 else None,
                    line=dict(color=COND_COLORS[condition], width=1.75),
                    legendgroup=condition,
                    showlegend=(col_idx == 1),
                    hovertemplate=(
                        f"<b>{system_label} – {display_label}</b><br>"
                        "Gait cycle: %{x:.1f}%<br>"
                        "Obliquity: %{y:.2f}°<br>"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=col_idx,
            )

            # jittered SD error bars
            idx = np.arange(0, len(x_all), errorbar_step)
            x_base = x_all[idx]
            m_err = m_all[idx]
            sd_err = sd_all[idx]

            x_err = x_base + cond_offset[condition]
            if x_err.size > 0:
                x_err[0] = x_base[0]
                x_err[-1] = x_base[-1]

            r, g, b = hex_to_rgb(COND_COLORS[condition])

            fig.add_trace(
                go.Scatter(
                    x=x_err,
                    y=m_err,
                    mode="markers",
                    name=None,
                    legendgroup=condition,
                    showlegend=False,
                    marker=dict(
                        color=COND_COLORS[condition],
                        size=5,
                        symbol="circle-open",
                        opacity=0.5,
                    ),
                    error_y=dict(
                        type="data",
                        array=sd_err,
                        visible=True,
                        symmetric=True,
                        thickness=1.0,
                        width=2,
                        color=f"rgba({r},{g},{b},0.45)",
                    ),
                    hovertemplate=(
                        f"<b>{system_label} – {display_label}</b><br>"
                        "Gait cycle: %{x:.1f}%<br>"
                        "Mean: %{y:.2f}°<br>"
                        "SD: %{error_y.array:.2f}°<br>"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=col_idx,
            )

        fig.update_yaxes(range=[ylo, yhi], row=1, col=col_idx)

    # axis titles (match your figure)
    fig.update_yaxes(
        title_text="<b>Pelvic obliquity (°)</b>",
        title_font=dict(size=BASE),
        row=1,
        col=1,
    )

    for col_idx in range(1, len(trackers) + 1):
        fig.update_xaxes(
            title_text="<b>Gait cycle (%)</b>",
            title_font=dict(size=BASE),
            row=1,
            col=col_idx,
        )

    fig.update_layout(
        title=None,
        template="simple_white",
        width=W * len(trackers),   # keep same per-panel width
        height=H,
        font=dict(family="Arial", size=BASE, color="black"),
        legend=dict(
            orientation="h",
            x=0.5,
            y=-0.15,
            xanchor="center",
            yanchor="top",
            font=dict(size=LEG),
            tracegroupgap=4,
        ),
        margin=dict(l=58, r=8, t=28, b=62),
    )

    for ann in fig.layout.annotations:
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
        range=[0, 100],
    )
    fig.update_yaxes(
        tickfont=dict(size=TICK),
        title_font=dict(size=BASE),
        showline=True,
        linecolor="black",
        mirror=True,
        ticks="outside",
        ticklen=3,
    )

    return fig


# -------------------- Run script --------------------

if __name__ == "__main__":
    # build combined summary across trackers
    all_summ = []
    for trk in trackers:
        all_summ.append(build_pelvis_summary(recordings, trk))
    summary = pd.concat(all_summ, ignore_index=True)

    fig = make_pelvis_system_comparison_figure(summary, trackers=trackers)
    fig.show()
    # fig.write_html("pelvis_obliquity_system_comparison.html")
    # fig.write_image("pelvis_obliquity_system_comparison.pdf")
    # fig.show()
