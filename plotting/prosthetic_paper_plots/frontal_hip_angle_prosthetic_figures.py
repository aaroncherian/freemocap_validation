from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


# =============================================================================
# CONFIG
# =============================================================================

OUTPUT_DIR = Path(r"C:\Users\aaron\Documents\prosthetics_paper\frontal_hip_angle")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Main-paper style: systems in separate panels, conditions overlaid.
MAIN_SYSTEMS = ["qualisys", "rtmpose_dlc"]

# Supplementary style: conditions in separate panels, systems overlaid.
SUPPLEMENT_SYSTEMS = ["mediapipe", "rtmpose", "qualisys", "rtmpose_dlc"]

SYSTEM_LABELS = {
    "qualisys": "Qualisys",
    "rtmpose_dlc": "FMC-Hybrid",
    "rtmpose": "FMC-RTMPose",
    "mediapipe": "FMC-MediaPipe",
}

SYSTEM_STYLES = {
    "qualisys": {"color": "#4d4d4d", "dash": "solid"},
    "rtmpose_dlc": {"color": "#1f77b4", "dash": "solid"},
    "rtmpose": {"color": "#d62728", "dash": "solid"},
    "mediapipe": {"color": "#e69f00", "dash": "solid"},
}

CONDITION_COLORS = {
    "lowest": "#94342b",
    "low": "#d39182",
    "neutral": "#524F4F",
    "high": "#7bb6c6",
    "highest": "#447c8e",
}

# If you decide later that you want the displayed sign reversed, this is the
# only switch you need. With False, positive means the right thigh is directed
# more laterally relative to the pelvis (abduction-like); negative is more
# medial (adduction-like).
FLIP_SIGN = False


# =============================================================================
# RECORDINGS
# =============================================================================

LEG_LENGTH_RECORDINGS = {
    "neg_5": Path(
        r"D:\2023-06-07_TF01\1.0_recordings\four_camera"
        r"\sesh_2023-06-07_12_38_16_TF01_leg_length_neg_5_trial_1"
    ),
    "neg_25": Path(
        r"D:\2023-06-07_TF01\1.0_recordings\four_camera"
        r"\sesh_2023-06-07_12_43_15_TF01_leg_length_neg_25_trial_1"
    ),
    "neutral": Path(
        r"D:\2023-06-07_TF01\1.0_recordings\four_camera"
        r"\sesh_2023-06-07_12_46_54_TF01_leg_length_neutral_trial_1"
    ),
    "pos_25": Path(
        r"D:\2023-06-07_TF01\1.0_recordings\four_camera"
        r"\sesh_2023-06-07_12_50_56_TF01_leg_length_pos_25_trial_1"
    ),
    "pos_5": Path(
        r"D:\2023-06-07_TF01\1.0_recordings\four_camera"
        r"\sesh_2023-06-07_12_55_21_TF01_leg_length_pos_5_trial_1"
    ),
}

ANKLE_ALIGNMENT_RECORDINGS = {
    "neg_5_6": Path(
        r"D:\2023-06-07_TF01\1.0_recordings\four_camera"
        r"\sesh_2023-06-07_11_55_05_TF01_flexion_neg_5_6_trial_1"
    ),
    "neg_2_8": Path(
        r"D:\2023-06-07_TF01\1.0_recordings\four_camera"
        r"\sesh_2023-06-07_12_03_15_TF01_flexion_neg_2_8_trial_1"
    ),
    "neutral": Path(
        r"D:\2023-06-07_TF01\1.0_recordings\four_camera"
        r"\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1"
    ),
    "pos_2_8": Path(
        r"D:\2023-06-07_TF01\1.0_recordings\four_camera"
        r"\sesh_2023-06-07_12_09_05_TF01_flexion_pos_2_8_trial_1"
    ),
    "pos_5_6": Path(
        r"D:\2023-06-07_TF01\1.0_recordings\four_camera"
        r"\sesh_2023-06-07_12_12_36_TF01_flexion_pos_5_6_trial_1"
    ),
}

TOE_ANGLE_RECORDINGS = {
    "neg_6_0": Path(
        r"D:\2023-06-07_TF01\1.0_recordings\four_camera"
        r"\sesh_2023-06-07_12_20_59_TF01_toe_angle_neg_6_trial_1"
    ),
    "neg_3_0": Path(
        r"D:\2023-06-07_TF01\1.0_recordings\four_camera"
        r"\sesh_2023-06-07_12_25_38_TF01_toe_angle_neg_3_trial_1"
    ),
    "neutral": Path(
        r"D:\2023-06-07_TF01\1.0_recordings\four_camera"
        r"\sesh_2023-06-07_12_28_46_TF01_toe_angle_neutral_trial_1"
    ),
    "pos_3_0": Path(
        r"D:\2023-06-07_TF01\1.0_recordings\four_camera"
        r"\sesh_2023-06-07_12_31_49_TF01_toe_angle_pos_3_trial_1"
    ),
    "pos_6_0": Path(
        r"D:\2023-06-07_TF01\1.0_recordings\four_camera"
        r"\sesh_2023-06-07_12_34_37_TF01_toe_angle_pos_6_trial_1"
    ),
}

EXPERIMENTS = {
    "leg_length": {
        "recordings": LEG_LENGTH_RECORDINGS,
        "order": ["neg_5", "neg_25", "neutral", "pos_25", "pos_5"],
        "labels": {
            "neg_5": "-12.70 mm",
            "neg_25": "-6.35 mm",
            "neutral": "Neutral",
            "pos_25": "+6.35 mm",
            "pos_5": "+12.70 mm",
        },
        "color_keys": {
            "neg_5": "lowest",
            "neg_25": "low",
            "neutral": "neutral",
            "pos_25": "high",
            "pos_5": "highest",
        },
    },
    "ankle_alignment": {
        "recordings": ANKLE_ALIGNMENT_RECORDINGS,
        "order": ["neg_5_6", "neg_2_8", "neutral", "pos_2_8", "pos_5_6"],
        "labels": {
            "neg_5_6": "-5.6°",
            "neg_2_8": "-2.8°",
            "neutral": "Neutral",
            "pos_2_8": "+2.8°",
            "pos_5_6": "+5.6°",
        },
        "color_keys": {
            "neg_5_6": "lowest",
            "neg_2_8": "low",
            "neutral": "neutral",
            "pos_2_8": "high",
            "pos_5_6": "highest",
        },
    },
    "toe_angle": {
        "recordings": TOE_ANGLE_RECORDINGS,
        "order": ["neg_6_0", "neg_3_0", "neutral", "pos_3_0", "pos_6_0"],
        "labels": {
            "neg_6_0": "-6°",
            "neg_3_0": "-3°",
            "neutral": "Neutral",
            "pos_3_0": "+3°",
            "pos_6_0": "+6°",
        },
        "color_keys": {
            "neg_6_0": "lowest",
            "neg_3_0": "low",
            "neutral": "neutral",
            "pos_3_0": "high",
            "pos_6_0": "highest",
        },
    },
}


# =============================================================================
# ANGLE CALCULATION
# =============================================================================

def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, eps)


def calculate_frontal_thigh_to_pelvis_angle(
    left_hip: np.ndarray,
    right_hip: np.ndarray,
    right_knee: np.ndarray,
) -> np.ndarray:
    """
    Simplified frontal-plane thigh-to-pelvis angle.

    Pelvis mediolateral axis:
        left_hip -> right_hip

    Pelvis-local vertical:
        global vertical projected perpendicular to the full 3-D pelvis
        mediolateral axis

    Thigh vector:
        right_hip -> right_knee

    The thigh vector is projected into the plane spanned by the pelvis
    mediolateral axis and pelvis-local vertical. The signed angle is then
    calculated relative to pelvis-local DOWN.

    Positive values mean the right thigh points more toward the subject's
    right/lateral side relative to the pelvis (abduction-like).
    Negative values mean more medial positioning (adduction-like).

    This is intentionally NOT the repository's existing hip Cardan Y component.
    It is a simpler frontal-plane geometric measure that does not use the
    hips_center -> neck_center trunk vector.
    """
    up = np.zeros_like(left_hip)
    up[:, 2] = 1.0

    pelvis = right_hip - left_hip
    pelvis_hat = normalize(pelvis)

    pelvis_up = up - np.sum(up * pelvis_hat, axis=1, keepdims=True) * pelvis_hat
    pelvis_up = normalize(pelvis_up)
    pelvis_down = -pelvis_up

    frontal_normal = normalize(np.cross(pelvis_up, pelvis_hat))

    thigh = right_knee - right_hip
    thigh_frontal = (
        thigh
        - np.sum(thigh * frontal_normal, axis=1, keepdims=True) * frontal_normal
    )
    thigh_frontal = normalize(thigh_frontal)

    lateral_component = np.sum(thigh_frontal * pelvis_hat, axis=1)
    down_component = np.sum(thigh_frontal * pelvis_down, axis=1)

    angle = np.degrees(np.arctan2(lateral_component, down_component))

    if FLIP_SIGN:
        angle *= -1

    return angle


# =============================================================================
# DATA LOADING
# =============================================================================

def trajectory_path(recording: Path, system: str) -> Path:
    return (
        recording
        / "validation"
        / system
        / "trajectories"
        / "trajectories_per_stride.csv"
    )


def load_condition(
    experiment: str,
    condition: str,
    recording: Path,
    system: str,
) -> pd.DataFrame:
    path = trajectory_path(recording, system)

    if not path.exists():
        raise FileNotFoundError(f"Missing trajectory file: {path}")

    df = pd.read_csv(path)
    required = {"cycle", "percent_gait_cycle", "marker", "x", "y", "z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    points = (
        df[df["marker"].isin(["left_hip", "right_hip", "right_knee"])]
        .pivot_table(
            index=["cycle", "percent_gait_cycle"],
            columns="marker",
            values=["x", "y", "z"],
        )
        .dropna()
        .reset_index()
    )

    available = set(points.columns.get_level_values(1))
    for marker in ["left_hip", "right_hip", "right_knee"]:
        if marker not in available:
            raise ValueError(f"{path} is missing marker: {marker}")

    def xyz(marker: str) -> np.ndarray:
        return np.column_stack([
            points[("x", marker)].to_numpy(float),
            points[("y", marker)].to_numpy(float),
            points[("z", marker)].to_numpy(float),
        ])

    angle = calculate_frontal_thigh_to_pelvis_angle(
        left_hip=xyz("left_hip"),
        right_hip=xyz("right_hip"),
        right_knee=xyz("right_knee"),
    )

    return pd.DataFrame({
        "experiment": experiment,
        "system": system,
        "condition": condition,
        "cycle": points["cycle"].to_numpy(),
        "percent_gait_cycle": points["percent_gait_cycle"].to_numpy(),
        "frontal_hip_angle": angle,
    })


def load_experiment(experiment: str) -> pd.DataFrame:
    cfg = EXPERIMENTS[experiment]
    rows = []

    for condition, recording in cfg["recordings"].items():
        for system in SUPPLEMENT_SYSTEMS:
            print(f"Loading {experiment}: {condition} / {system}")
            rows.append(load_condition(experiment, condition, recording, system))

    return pd.concat(rows, ignore_index=True)


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby(
            ["experiment", "system", "condition", "percent_gait_cycle"],
            as_index=False,
        )["frontal_hip_angle"]
        .agg(mean="mean", std="std")
    )


# =============================================================================
# PLOT HELPERS
# =============================================================================

def rgba(hex_color: str, alpha: float) -> str:
    color = hex_color.lstrip("#")
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def add_sd_band(
    fig: go.Figure,
    *,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    color: str,
    row: int,
    col: int,
    legendgroup: str,
) -> None:
    std = np.nan_to_num(std)

    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean - std,
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
            legendgroup=legendgroup,
        ),
        row=row,
        col=col,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean + std,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=rgba(color, 0.14),
            hoverinfo="skip",
            showlegend=False,
            legendgroup=legendgroup,
        ),
        row=row,
        col=col,
    )


def overall_y_range(summary: pd.DataFrame) -> list[float]:
    low = (summary["mean"] - summary["std"].fillna(0)).min()
    high = (summary["mean"] + summary["std"].fillna(0)).max()
    pad = 0.08 * (high - low + 1e-9)
    return [float(low - pad), float(high + pad)]


# =============================================================================
# MAIN-PAPER STYLE
# Systems separated into columns; conditions overlaid within each system.
# This mirrors the current pelvic-obliquity style in the main prosthetic paper.
# =============================================================================

def make_main_style_figure(
    summary: pd.DataFrame,
    experiment: str,
) -> go.Figure:
    cfg = EXPERIMENTS[experiment]
    plot_data = summary[
        (summary["experiment"] == experiment)
        & (summary["system"].isin(MAIN_SYSTEMS))
    ].copy()

    DPI = 300
    FIG_W_IN = 2.0
    FIG_H_IN = 1.0
    W = int(FIG_W_IN * DPI)
    H = int(FIG_H_IN * DPI)

    BASE = 16
    TICK = 14
    LEG = 14
    TITLE = 14

    fig = make_subplots(
        rows=1,
        cols=len(MAIN_SYSTEMS),
        subplot_titles=[SYSTEM_LABELS[s] for s in MAIN_SYSTEMS],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.05,
    )

    y_range = overall_y_range(plot_data)

    for col_idx, system in enumerate(MAIN_SYSTEMS, start=1):
        fig.add_hline(
            y=0,
            line=dict(color="gray", width=0.75, dash="dot"),
            row=1,
            col=col_idx,
        )

        for condition in cfg["order"]:
            sub = plot_data[
                (plot_data["system"] == system)
                & (plot_data["condition"] == condition)
            ].sort_values("percent_gait_cycle")

            if sub.empty:
                continue

            color = CONDITION_COLORS[cfg["color_keys"][condition]]
            label = cfg["labels"][condition]
            x = sub["percent_gait_cycle"].to_numpy()
            mean = sub["mean"].to_numpy()
            std = sub["std"].fillna(0).to_numpy()

            add_sd_band(
                fig,
                x=x,
                mean=mean,
                std=std,
                color=color,
                row=1,
                col=col_idx,
                legendgroup=condition,
            )

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=mean,
                    mode="lines",
                    name=label,
                    legendgroup=condition,
                    showlegend=(col_idx == 1),
                    line=dict(color=color, width=1.75),
                    hovertemplate=(
                        f"<b>{SYSTEM_LABELS[system]} – {label}</b><br>"
                        "Gait cycle: %{x:.1f}%<br>"
                        "Frontal thigh-to-pelvis angle: %{y:.2f}°"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=col_idx,
            )

        fig.update_yaxes(range=y_range, row=1, col=col_idx)

    fig.update_yaxes(
        title_text="<b>Frontal thigh-to-pelvis angle (°)</b>",
        title_font=dict(size=BASE),
        row=1,
        col=1,
    )

    for col_idx in range(1, len(MAIN_SYSTEMS) + 1):
        fig.update_xaxes(
            title_text="<b>Gait cycle (%)</b>",
            title_font=dict(size=BASE),
            row=1,
            col=col_idx,
        )

    fig.update_layout(
        template="simple_white",
        width=W,
        height=H,
        font=dict(family="Arial", size=BASE, color="black"),
        legend=dict(
            orientation="h",
            x=0.5,
            y=-0.28,
            xanchor="center",
            yanchor="top",
            font=dict(size=LEG),
        ),
        margin=dict(l=62, r=8, t=28, b=62),
    )

    for annotation in fig.layout.annotations:
        annotation.font.size = TITLE
        annotation.font.weight = "bold"

    fig.update_xaxes(
        range=[0, 100],
        tickvals=[0, 50, 100],
        tickfont=dict(size=TICK),
        showline=True,
        linecolor="black",
        mirror=True,
        ticks="outside",
        ticklen=3,
    )

    fig.update_yaxes(
        tickfont=dict(size=TICK),
        showline=True,
        linecolor="black",
        mirror=True,
        ticks="outside",
        ticklen=3,
    )

    return fig


# =============================================================================
# SUPPLEMENTARY STYLE
# Conditions separated into columns; systems overlaid within each condition.
# Uses all four systems, matching the supplementary comparison figures.
# =============================================================================

def make_supplement_style_figure(
    summary: pd.DataFrame,
    experiment: str,
) -> go.Figure:
    cfg = EXPERIMENTS[experiment]
    plot_data = summary[
        (summary["experiment"] == experiment)
        & (summary["system"].isin(SUPPLEMENT_SYSTEMS))
    ].copy()

    DPI = 300
    FIG_W_IN = 4.5
    FIG_H_IN = 1.0
    W = int(FIG_W_IN * DPI)
    H = int(FIG_H_IN * DPI)

    BASE = 16
    TICK = 14
    LEG = 14
    TITLE = 14

    fig = make_subplots(
        rows=1,
        cols=len(cfg["order"]),
        subplot_titles=[cfg["labels"][c] for c in cfg["order"]],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.025,
    )

    y_range = overall_y_range(plot_data)

    for col_idx, condition in enumerate(cfg["order"], start=1):
        fig.add_hline(
            y=0,
            line=dict(color="gray", width=0.75, dash="dot"),
            row=1,
            col=col_idx,
        )

        # Qualisys first so markerless traces sit on top where they overlap.
        draw_order = ["qualisys", "rtmpose_dlc", "rtmpose", "mediapipe"]

        for system in draw_order:
            sub = plot_data[
                (plot_data["condition"] == condition)
                & (plot_data["system"] == system)
            ].sort_values("percent_gait_cycle")

            if sub.empty:
                continue

            style = SYSTEM_STYLES[system]
            x = sub["percent_gait_cycle"].to_numpy()
            mean = sub["mean"].to_numpy()
            std = sub["std"].fillna(0).to_numpy()

            add_sd_band(
                fig,
                x=x,
                mean=mean,
                std=std,
                color=style["color"],
                row=1,
                col=col_idx,
                legendgroup=system,
            )

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=mean,
                    mode="lines",
                    name=SYSTEM_LABELS[system],
                    legendgroup=system,
                    showlegend=(col_idx == 1),
                    line=dict(
                        color=style["color"],
                        width=2.0,
                        dash=style["dash"],
                    ),
                    hovertemplate=(
                        f"<b>{SYSTEM_LABELS[system]} – {cfg['labels'][condition]}</b><br>"
                        "Gait cycle: %{x:.1f}%<br>"
                        "Frontal thigh-to-pelvis angle: %{y:.2f}°"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=col_idx,
            )

        fig.update_yaxes(range=y_range, row=1, col=col_idx)

    fig.update_yaxes(
        title_text="<b>Frontal thigh-to-pelvis angle (°)</b>",
        title_font=dict(size=BASE),
        row=1,
        col=1,
    )

    for col_idx in range(1, len(cfg["order"]) + 1):
        fig.update_xaxes(
            title_text="<b>Gait cycle (%)</b>",
            title_font=dict(size=BASE),
            row=1,
            col=col_idx,
        )

    fig.update_layout(
        template="simple_white",
        width=W,
        height=H,
        font=dict(family="Arial", size=BASE, color="black"),
        legend=dict(
            orientation="h",
            x=0.5,
            y=-0.30,
            xanchor="center",
            yanchor="top",
            font=dict(size=LEG),
        ),
        margin=dict(l=65, r=10, t=30, b=65),
    )

    for annotation in fig.layout.annotations:
        annotation.font.size = TITLE
        annotation.font.weight = "bold"

    fig.update_xaxes(
        range=[0, 100],
        tickvals=[0, 50, 100],
        tickfont=dict(size=TICK),
        showline=True,
        linecolor="black",
        mirror=True,
        ticks="outside",
        ticklen=3,
    )

    fig.update_yaxes(
        tickfont=dict(size=TICK),
        showline=True,
        linecolor="black",
        mirror=True,
        ticks="outside",
        ticklen=3,
    )

    return fig


# =============================================================================
# SAVE / RUN
# =============================================================================

def save_figure(fig: go.Figure, base: Path) -> None:
    fig.write_html(base.with_suffix(".html"))
    fig.write_image(base.with_suffix(".pdf"))
    fig.write_image(base.with_suffix(".png"), scale=3)


if __name__ == "__main__":
    pio.kaleido.scope.mathjax = None

    all_stride_data = []
    all_summary_data = []

    for experiment in EXPERIMENTS:
        print(f"\n{'=' * 80}")
        print(f"{experiment}")
        print(f"{'=' * 80}")

        data = load_experiment(experiment)
        summary = summarize(data)

        all_stride_data.append(data)
        all_summary_data.append(summary)

        data.to_csv(
            OUTPUT_DIR / f"{experiment}_frontal_hip_angle_per_stride.csv",
            index=False,
        )
        summary.to_csv(
            OUTPUT_DIR / f"{experiment}_frontal_hip_angle_summary.csv",
            index=False,
        )

        main_fig = make_main_style_figure(summary, experiment)
        supplement_fig = make_supplement_style_figure(summary, experiment)

        save_figure(
            main_fig,
            OUTPUT_DIR / f"{experiment}_frontal_hip_angle_main_overlay",
        )
        save_figure(
            supplement_fig,
            OUTPUT_DIR / f"{experiment}_frontal_hip_angle_supplement_separated",
        )

        print("\nSaved:")
        print(f"  {experiment}_frontal_hip_angle_main_overlay.[pdf/png/html]")
        print(f"  {experiment}_frontal_hip_angle_supplement_separated.[pdf/png/html]")

    pd.concat(all_stride_data, ignore_index=True).to_csv(
        OUTPUT_DIR / "all_adjustments_frontal_hip_angle_per_stride.csv",
        index=False,
    )

    pd.concat(all_summary_data, ignore_index=True).to_csv(
        OUTPUT_DIR / "all_adjustments_frontal_hip_angle_summary.csv",
        index=False,
    )

    print(f"\nAll outputs saved to:\n{OUTPUT_DIR}")
