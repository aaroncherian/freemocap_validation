from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

CONDITION_ORDER = ["neg_5_6", "neg_2_8", "neutral", "pos_2_8", "pos_5_6"]

CONDITION_LABELS = {
    "neg_5_6": "-5.6°",
    "neg_2_8": "-2.8°",
    "neutral": "Neutral",
    "pos_2_8": "+2.8°",
    "pos_5_6": "+5.6°",
}

SYSTEMS = ["rtmpose_dlc", "qualisys"]

SYSTEM_LABELS = {
    "mediapipe": "FMC-MediaPipe",
    "rtmpose": "FMC-RTMPose",
    "rtmpose_dlc": "FMC-Hybrid",
    "qualisys": "Qualisys",
}

SYSTEM_STYLES = {
    "mediapipe": {"color": "#4E012B", "dash": "solid", "symbol": "diamond-open"},
    "rtmpose": {"color": "#a0f700", "dash": "solid", "symbol": "triangle-up-open"},
    "rtmpose_dlc": {"color": "#1f77b4", "dash": "solid", "symbol": "circle-open"},
    "qualisys": {"color": "#d62728", "dash": "solid", "symbol": "square-open"},
}

OUTPUT_DIR = Path(r"C:\Users\aaron\Documents\prosthetics_paper")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_HTML = OUTPUT_DIR / "ankle_adjustment_condition_overlay.html"
OUT_PDF = OUTPUT_DIR / "ankle_adjustment_condition_overlay.pdf"
OUT_PNG = OUTPUT_DIR / "ankle_adjustment_condition_overlay.png"


# -------------------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------------------

def load_angle_summary_for_tracker(
    conditions: dict[str, Path | str],
    tracker_dir: str,
    *,
    joint: str,
    side: str = "right",
    component: str = "dorsi_plantar",
) -> pd.DataFrame:
    all_summaries = []

    for condition, root in conditions.items():
        csv_path = (
            Path(root)
            / "validation"
            / tracker_dir
            / "joint_angles"
            / "joint_angles_per_stride_summary_stats.csv"
        )

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing summary CSV for condition '{condition}' and tracker '{tracker_dir}': {csv_path}"
            )

        df = pd.read_csv(csv_path)

        if "joint" in df.columns:
            df = df[df["joint"] == joint]
        if "side" in df.columns:
            df = df[df["side"] == side]
        if "component" in df.columns:
            df = df[df["component"] == component]

        if df.empty:
            raise ValueError(
                f"No rows in {csv_path} for joint={joint}, side={side}, component={component}"
            )

        wide = df.pivot(index="percent_gait_cycle", columns="stat", values="value").reset_index()

        if not {"mean", "std"}.issubset(wide.columns):
            raise ValueError(
                f"Expected mean + std for {condition}/{tracker_dir}, got {wide.columns.tolist()}"
            )

        wide["system"] = tracker_dir
        wide["condition"] = condition
        wide["joint"] = joint

        all_summaries.append(
            wide[["system", "condition", "joint", "percent_gait_cycle", "mean", "std"]]
        )

    return pd.concat(all_summaries, ignore_index=True)


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    color = hex_color.lstrip("#")
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def add_mean_and_band(
    fig: go.Figure,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    system: str,
    row: int,
    col: int,
    showlegend: bool,
    hover_label: str,
) -> None:
    style = SYSTEM_STYLES[system]
    upper = mean + std
    lower = mean - std

    fig.add_trace(
        go.Scatter(
            x=x,
            y=lower,
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
            legendgroup=system,
        ),
        row=row,
        col=col,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=upper,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=hex_to_rgba(style["color"], 0.14),
            hoverinfo="skip",
            showlegend=False,
            legendgroup=system,
        ),
        row=row,
        col=col,
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean,
            mode="lines",
            name=SYSTEM_LABELS[system],
            legendgroup=system,
            showlegend=showlegend,
            line=dict(color=style["color"], width=2.0, dash=style["dash"]),
            hovertemplate=(
                f"<b>{SYSTEM_LABELS[system]} – {hover_label}</b><br>"
                "Gait cycle: %{x:.1f}%<br>"
                "Angle: %{y:.1f}°<br>"
                "<extra></extra>"
            ),
        ),
        row=row,
        col=col,
    )


# -------------------------------------------------------------------
# FIGURE
# -------------------------------------------------------------------

def make_condition_overlay_figure(
    summary: pd.DataFrame,
    joints_in_rows: list[str] = ["knee", "ankle"],
    flip_sign_for: set[str] | None = None,
) -> go.Figure:
    flip_sign_for = flip_sign_for or set()

    DPI = 300
    FIG_W_IN = 7.0
    FIG_H_IN = 3.2
    W = int(FIG_W_IN * DPI)
    H = int(FIG_H_IN * DPI)

    BASE = 14
    TICK = 11
    LEG = 12
    TITLE = 12

    subplot_titles = CONDITION_LABELS.values()

    fig = make_subplots(
        rows=len(joints_in_rows),
        cols=len(CONDITION_ORDER),
        shared_xaxes=True,
        shared_yaxes="rows",
        subplot_titles=list(subplot_titles),
        vertical_spacing=0.10,
        horizontal_spacing=0.025,
    )

    for row_idx, joint in enumerate(joints_in_rows, start=1):
        joint_data = summary[summary["joint"] == joint].copy()

        if joint in flip_sign_for:
            joint_data["mean"] *= -1

        ymin = (joint_data["mean"] - joint_data["std"]).min()
        ymax = (joint_data["mean"] + joint_data["std"]).max()
        pad = 0.08 * (ymax - ymin + 1e-9)
        y_range = [float(ymin - pad), float(ymax + pad)]

        for col_idx, condition in enumerate(CONDITION_ORDER, start=1):
            if joint == "ankle":
                fig.add_hline(
                    y=0,
                    line=dict(color="gray", width=0.75, dash="dot"),
                    row=row_idx,
                    col=col_idx,
                )

            for system in SYSTEMS:
                sub = joint_data[
                    (joint_data["condition"] == condition)
                    & (joint_data["system"] == system)
                ].sort_values("percent_gait_cycle")

                if sub.empty:
                    continue

                add_mean_and_band(
                    fig=fig,
                    x=sub["percent_gait_cycle"].to_numpy(),
                    mean=sub["mean"].to_numpy(),
                    std=sub["std"].to_numpy(),
                    system=system,
                    row=row_idx,
                    col=col_idx,
                    showlegend=(row_idx == 1 and col_idx == 1),
                    hover_label=f"{joint.capitalize()} – {CONDITION_LABELS[condition]}",
                )

            fig.update_yaxes(range=y_range, row=row_idx, col=col_idx)

        joint_label = "Knee angle" if joint == "knee" else "Ankle angle"
        fig.update_yaxes(
            title_text=f"<b>{joint_label} (°)</b>",
            title_font=dict(size=BASE),
            row=row_idx,
            col=1,
        )

    for col_idx in range(1, len(CONDITION_ORDER) + 1):
        fig.update_xaxes(
            title_text="<b>Gait cycle (%)</b>",
            title_font=dict(size=BASE),
            row=len(joints_in_rows),
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
            y=-0.13,
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


# -------------------------------------------------------------------
# RUN
# -------------------------------------------------------------------

if __name__ == "__main__":
    conditions = {
        "neutral": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1",
        "neg_2_8": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_03_15_TF01_flexion_neg_2_8_trial_1",
        "neg_5_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_11_55_05_TF01_flexion_neg_5_6_trial_1",
        "pos_2_8": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_09_05_TF01_flexion_pos_2_8_trial_1",
        "pos_5_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_12_36_TF01_flexion_pos_5_6_trial_1",
    }

    joints_to_load = [
        dict(joint="knee", side="right", component="flex_ext"),
        dict(joint="ankle", side="right", component="dorsi_plantar"),
    ]

    all_rows = []
    for joint_config in joints_to_load:
        for system in SYSTEMS:
            all_rows.append(
                load_angle_summary_for_tracker(
                    conditions,
                    tracker_dir=system,
                    joint=joint_config["joint"],
                    side=joint_config["side"],
                    component=joint_config["component"],
                )
            )

    summary = pd.concat(all_rows, ignore_index=True)
    fig = make_condition_overlay_figure(summary, flip_sign_for={"knee"})

    fig.show()
    fig.write_html(OUT_HTML)

    pio.kaleido.scope.mathjax = None
    fig.write_image(OUT_PDF)
    fig.write_image(OUT_PNG, scale=3)

    print("\nSaved:")
    print(f"  {OUT_HTML}")
    print(f"  {OUT_PDF}")
    print(f"  {OUT_PNG}")
