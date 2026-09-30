from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

recordings = {
    "neg_5": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_38_16_TF01_leg_length_neg_5_trial_1"),
    "neg_25": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_43_15_TF01_leg_length_neg_25_trial_1"),
    "neutral": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_46_54_TF01_leg_length_neutral_trial_1"),
    "pos_25": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_50_56_TF01_leg_length_pos_25_trial_1"),
    "pos_5": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_55_21_TF01_leg_length_pos_5_trial_1"),
}

SYSTEMS = ["mediapipe", "rtmpose" , "qualisys", "rtmpose_dlc"]

SYSTEM_LABELS = {
    "rtmpose_dlc": "FMC-Hybrid",
    "qualisys": "Qualisys",
    "rtmpose": "FMC-RTMPose",
    "mediapipe": "FMC-MediaPipe",
}

SYSTEM_STYLES = {
    "rtmpose_dlc": {"color": "#1f77b4", "dash": "solid"},
    "qualisys": {"color": "#4d4d4d", "dash": "solid"},
    "rtmpose": {"color": "#d62728", "dash": "solid"},
    "mediapipe": {"color": "#e69f00", "dash": "solid"},
}

COND_ORDER = ["neg_5", "neg_25", "neutral", "pos_25", "pos_5"]
COND_LABELS = {
    "neg_5": "-12.70 mm",
    "neg_25": "-6.35 mm",
    "neutral": "Neutral",
    "pos_25": "+6.35 mm",
    "pos_5": "+12.70 mm",
}

JOINT = "pelvis"
COMPONENT = "obliquity"
SIDE_PREFERENCE = ("mid", "right", "left")

OUTPUT_DIR = Path(r"C:\Users\aaron\Documents\prosthetics_paper")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = OUTPUT_DIR / "pelvic_obliquity_condition_overlay.pdf"
OUT_PNG = OUTPUT_DIR / "pelvic_obliquity_condition_overlay.png"


# -------------------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------------------

def load_stride_summary_csv(recording: Path, tracker: str) -> pd.DataFrame:
    csv_path = recording / "validation" / tracker / "joint_angles" / "joint_angles_per_stride_summary_stats.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing: {csv_path}")
    return pd.read_csv(csv_path)


def filter_pelvis_obliquity(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df["joint"] == JOINT) & (df["component"] == COMPONENT)].copy()

    if "side" in df.columns and not df.empty:
        available_sides = df["side"].dropna().unique().tolist()
        for side in SIDE_PREFERENCE:
            if side in available_sides:
                df = df[df["side"] == side]
                break

    return df


def build_pelvis_summary(recordings: dict[str, Path], tracker: str) -> pd.DataFrame:
    rows = []

    for condition, recording in recordings.items():
        df = filter_pelvis_obliquity(load_stride_summary_csv(recording, tracker))

        if df.empty:
            raise ValueError(f"No pelvis obliquity rows for {condition} / {tracker}")

        wide = df.pivot(index="percent_gait_cycle", columns="stat", values="value").reset_index()

        if not {"mean", "std"}.issubset(wide.columns):
            raise ValueError(
                f"Expected mean and std for {condition}/{tracker}. Got: {wide.columns.tolist()}"
            )

        wide["system"] = tracker
        wide["condition"] = condition
        rows.append(wide[["system", "condition", "percent_gait_cycle", "mean", "std"]])

    return pd.concat(rows, ignore_index=True)


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    color = hex_color.lstrip("#")
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# -------------------------------------------------------------------
# FIGURE
# -------------------------------------------------------------------

def make_pelvic_obliquity_condition_overlay(summary: pd.DataFrame) -> go.Figure:
    DPI = 300
    FIG_W_IN = 4.5
    FIG_H_IN = 1
    W = int(FIG_W_IN * DPI)
    H = int(FIG_H_IN * DPI)


    BASE = 16
    TICK = 14
    LEG = 14
    TITLE = 14

    fig = make_subplots(
        rows=1,
        cols=len(COND_ORDER),
        subplot_titles=[COND_LABELS[c] for c in COND_ORDER],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.025,
    )

    ymin = (summary["mean"] - summary["std"].fillna(0)).min()
    ymax = (summary["mean"] + summary["std"].fillna(0)).max()
    pad = 0.08 * (ymax - ymin + 1e-9)
    y_range = [float(ymin - pad), float(ymax + pad)]

    for col_idx, condition in enumerate(COND_ORDER, start=1):
        fig.add_hline(
            y=0,
            line=dict(color="gray", width=0.75, dash="dot"),
            row=1,
            col=col_idx,
        )

        # Qualisys is added first so the reference is behind FMC-Hybrid.
        for system in SYSTEMS:
            sub = summary[
                (summary["condition"] == condition)
                & (summary["system"] == system)
            ].sort_values("percent_gait_cycle")

            if sub.empty:
                continue

            x = sub["percent_gait_cycle"].to_numpy()
            mean = sub["mean"].to_numpy()
            std = sub["std"].fillna(0).to_numpy()
            style = SYSTEM_STYLES[system]

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=mean - std,
                    mode="lines",
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=system,
                    opacity=0.7,

                ),
                row=1,
                col=col_idx,
            )

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=mean + std,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=hex_to_rgba(style["color"], 0.14),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=system,
                    
                ),
                row=1,
                col=col_idx,
            )

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=mean,
                    mode="lines",
                    name=SYSTEM_LABELS[system],
                    legendgroup=system,
                    showlegend=(col_idx == 1),
                    line=dict(color=style["color"], width=2.0, dash=style["dash"]),
                    hovertemplate=(
                        f"<b>{SYSTEM_LABELS[system]} – {COND_LABELS[condition]}</b><br>"
                        "Gait cycle: %{x:.1f}%<br>"
                        "Pelvic obliquity: %{y:.2f}°<br>"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=col_idx,
            )

        fig.update_yaxes(range=y_range, row=1, col=col_idx)

    fig.update_yaxes(
        title_text="<b>Pelvic obliquity (°)</b>",
        title_font=dict(size=BASE),
        row=1,
        col=1,
    )

    for col_idx in range(1, len(COND_ORDER) + 1):
        fig.update_xaxes(
            title_text="<b>Gait cycle (%)</b>",
            title_font=dict(size=BASE),
            row=1,
            col=col_idx,
        )

    fig.update_layout(
        title=None,
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
        margin=dict(l=62, r=10, t=32, b=62),
    )

    for annotation in fig.layout.annotations:
        annotation.font.size = TITLE
        annotation.font.weight = "bold"

    fig.update_xaxes(
        tickfont=dict(size=TICK),
        showline=True,
        linecolor="black",
        mirror=True,
        ticks="outside",
        ticklen=3,
        range=[0, 100],
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
    summary = pd.concat(
        [build_pelvis_summary(recordings, system) for system in SYSTEMS],
        ignore_index=True,
    )

    fig = make_pelvic_obliquity_condition_overlay(summary)
    fig.show()

    pio.kaleido.scope.mathjax = None
    fig.write_image(OUT_PDF)
    fig.write_image(OUT_PNG, scale=3)

    print("\nSaved:")
    print(f"  {OUT_PDF}")
    print(f"  {OUT_PNG}")
