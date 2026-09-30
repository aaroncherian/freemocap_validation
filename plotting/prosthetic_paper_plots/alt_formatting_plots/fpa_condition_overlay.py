from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------

conditions = {
    "neg_6_0": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_20_59_TF01_toe_angle_neg_6_trial_1",
    "neg_3_0": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_25_38_TF01_toe_angle_neg_3_trial_1",
    "neutral": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_28_46_TF01_toe_angle_neutral_trial_1",
    "pos_3_0": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_31_49_TF01_toe_angle_pos_3_trial_1",
    "pos_6_0": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_34_37_TF01_toe_angle_pos_6_trial_1",
}

SYSTEMS = [
        #     "mediapipe", 
        #    "rtmpose",
           "qualisys", 
           "rtmpose_dlc", 
           ]

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

COND_ORDER = ["neg_6_0", "neg_3_0", "neutral", "pos_3_0", "pos_6_0"]
COND_LABELS = {
    "neg_6_0": "-6°",
    "neg_3_0": "-3°",
    "neutral": "Neutral",
    "pos_3_0": "+3°",
    "pos_6_0": "+6°",
}

STANCE_SWING_BOUNDARY = 60

# Reference direction (line of progression) and ground normal
a = np.array([0, 1, 0], dtype=float)
n = np.array([0, 0, 1], dtype=float)

OUTPUT_DIR = Path(r"C:\Users\aaron\Documents\prosthetics_paper")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PDF = OUTPUT_DIR / "fpa_condition_overlay.pdf"
OUT_PNG = OUTPUT_DIR / "fpa_condition_overlay.png"


# -------------------------------------------------------------------
# FPA CALCULATION
# -------------------------------------------------------------------

def calculate_foot_progression_angle(
    foot_vector: np.ndarray,
    reference_vector: np.ndarray,
    axis_of_rotation: np.ndarray,
) -> float:
    n_axis = axis_of_rotation.astype(float)
    n_axis /= np.linalg.norm(n_axis) + 1e-12

    ref = reference_vector.astype(float)
    foot = foot_vector.astype(float)

    ref_proj = ref - np.dot(ref, n_axis) * n_axis
    foot_proj = foot - np.dot(foot, n_axis) * n_axis

    ref_hat = ref_proj / (np.linalg.norm(ref_proj) + 1e-12)
    foot_hat = foot_proj / (np.linalg.norm(foot_proj) + 1e-12)

    sin_theta = np.dot(np.cross(ref_hat, foot_hat), n_axis)
    cos_theta = np.dot(ref_hat, foot_hat)
    return np.degrees(np.arctan2(sin_theta, cos_theta))


def load_fpa_data() -> pd.DataFrame:
    rows = []

    for condition, root_path in conditions.items():
        for system in SYSTEMS:
            path_to_csv = (
                Path(root_path)
                / "validation"
                / system
                / "trajectories"
                / "trajectories_per_stride.csv"
            )

            if not path_to_csv.exists():
                raise FileNotFoundError(f"Missing: {path_to_csv}")

            traj_df = pd.read_csv(path_to_csv)
            foot_heel_df = traj_df.query(
                "marker in ['right_foot_index','right_heel']"
            ).pivot_table(
                index=["cycle", "percent_gait_cycle"],
                columns="marker",
                values=["x", "y", "z"],
            ).reset_index()

            vector_df = pd.DataFrame()
            for axis in ["x", "y", "z"]:
                vector_df[axis] = (
                    foot_heel_df[axis]["right_foot_index"]
                    - foot_heel_df[axis]["right_heel"]
                )

            vector_df["cycle"] = foot_heel_df["cycle"]
            vector_df["percent_gait_cycle"] = foot_heel_df["percent_gait_cycle"]

            fpa_df = vector_df[["cycle", "percent_gait_cycle"]].copy()
            fpa_df["fpa"] = vector_df.apply(
                lambda row: calculate_foot_progression_angle(
                    np.array([row["x"], row["y"], row["z"]]),
                    reference_vector=a,
                    axis_of_rotation=n,
                ),
                axis=1,
            )
            fpa_df["system"] = system
            fpa_df["condition"] = condition
            rows.append(fpa_df)

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

def make_fpa_condition_overlay_figure(fpas: pd.DataFrame) -> go.Figure:
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

    grouped_all = (
        fpas.groupby(["system", "condition", "percent_gait_cycle"])["fpa"]
        .agg(["mean", "std"])
        .reset_index()
    )

    ymin = (grouped_all["mean"] - grouped_all["std"].fillna(0)).min()
    ymax = (grouped_all["mean"] + grouped_all["std"].fillna(0)).max()
    pad = 0.08 * (ymax - ymin + 1e-9)
    y_range = [float(ymin - pad), float(ymax + pad)]

    for col_idx, condition in enumerate(COND_ORDER, start=1):
        fig.add_vline(
            x=STANCE_SWING_BOUNDARY,
            line=dict(color="gray", width=1, dash="dash"),
            row=1,
            col=col_idx,
        )

        fig.add_hline(
            y=0,
            line=dict(color="gray", width=0.75, dash="dot"),
            row=1,
            col=col_idx,
        )

        for system in SYSTEMS:
            sub = grouped_all[
                (grouped_all["condition"] == condition)
                & (grouped_all["system"] == system)
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
                        "FPA: %{y:.1f}°<br>"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=col_idx,
            )

        fig.update_yaxes(range=y_range, row=1, col=col_idx)

    fig.update_yaxes(
        title_text="<b>Foot progression angle (°)</b>",
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
# RMSE
# -------------------------------------------------------------------

def calculate_rmse(reference_values: np.ndarray, test_values: np.ndarray) -> float:
    return float(np.sqrt(np.mean((test_values - reference_values) ** 2)))


def calculate_fpa_rmse(fpas: pd.DataFrame) -> tuple[float, float]:
    wide = fpas.pivot_table(
        index=["cycle", "percent_gait_cycle", "condition"],
        columns="system",
        values="fpa",
    ).reset_index()

    per_stride = (
        wide.groupby(["condition", "cycle"])
        .apply(
            lambda x: calculate_rmse(
                np.array(x["qualisys"]),
                np.array(x["rtmpose_dlc"]),
            )
        )
        .reset_index(name="rmse")
    )

    mean_rmse = per_stride.groupby("condition")["rmse"].mean()
    return float(mean_rmse.mean()), float(mean_rmse.std())


# -------------------------------------------------------------------
# RUN
# -------------------------------------------------------------------

if __name__ == "__main__":
    fpas = load_fpa_data()
    fig = make_fpa_condition_overlay_figure(fpas)

    fig.show()

    pio.kaleido.scope.mathjax = None
    fig.write_image(OUT_PDF)
    fig.write_image(OUT_PNG, scale=3)

    mean_rmse, std_rmse = calculate_fpa_rmse(fpas)
    print(f"FPA RMSE vs Qualisys across all conditions: {mean_rmse:.2f}° ± {std_rmse:.2f}°")
    print("\nSaved:")
    print(f"  {OUT_PDF}")
    print(f"  {OUT_PNG}")
