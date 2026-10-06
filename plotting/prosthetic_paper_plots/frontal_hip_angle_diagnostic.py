from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =============================================================================
# CONFIG
# =============================================================================

OUTPUT_DIR = Path(r"C:\Users\aaron\Documents\prosthetics_paper\frontal_hip_angle_diagnostic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEMS = ["qualisys", "rtmpose_dlc"]

SYSTEM_LABELS = {
    "qualisys": "Qualisys",
    "rtmpose_dlc": "FMC-Hybrid",
}

SYSTEM_STYLES = {
    "qualisys": {"color": "#4d4d4d", "dash": "solid"},
    "rtmpose_dlc": {"color": "#1f77b4", "dash": "solid"},
}

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
    },
}


# =============================================================================
# VECTOR MATH
# =============================================================================

def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, eps)


def wrap_180(angle_deg: np.ndarray) -> np.ndarray:
    return (angle_deg + 180.0) % 360.0 - 180.0


def calculate_frontal_geometry(
    left_hip: np.ndarray,
    right_hip: np.ndarray,
    right_knee: np.ndarray,
) -> pd.DataFrame:
    """
    Compute three geometrically related frontal-plane quantities from the SAME
    three 3-D points.

    Definitions
    -----------
    pelvis_obliquity:
        Signed tilt of the left->right hip vector relative to the horizontal
        plane. This matches the manuscript definition:
            atan2(v_z, ||v_xy||)
        Positive = right hip higher.

    global_thigh_angle:
        Signed right-thigh angle relative to global DOWN in the vertical plane
        defined by:
            - global vertical
            - the HORIZONTAL projection of left->right pelvis direction
        Positive = knee directed toward the subject's right/lateral direction.

    pelvis_relative_angle:
        The same thigh vector expressed relative to pelvis-local DOWN, where
        pelvis-local vertical is global vertical projected perpendicular to the
        full 3-D left->right hip axis. This is the "new" simplified frontal hip
        angle used in the previous script.

    Geometry identity
    -----------------
    Because these three quantities are defined in the same vertical/frontal
    plane, they should satisfy:

        pelvis_relative_angle
            = global_thigh_angle - pelvis_obliquity

    apart from numerical precision / angle wrapping.

    This identity is the key diagnostic. If it fails, the new angle
    implementation is wrong. If it holds but differs from the repository's
    Cardan hip abd/add angle, the difference is due to the coordinate-system
    definition rather than a coding bug in this simplified measure.
    """
    up = np.zeros_like(left_hip)
    up[:, 2] = 1.0
    down = -up

    pelvis = right_hip - left_hip
    pelvis_hat = normalize(pelvis)

    pelvis_horizontal = pelvis.copy()
    pelvis_horizontal[:, 2] = 0.0
    pelvis_horizontal_hat = normalize(pelvis_horizontal)

    # Manuscript pelvic obliquity:
    pelvis_obliquity = np.degrees(
        np.arctan2(
            pelvis[:, 2],
            np.linalg.norm(pelvis[:, :2], axis=1),
        )
    )

    # The plane spanned by horizontal subject-right and global vertical.
    # This automatically follows subject yaw and avoids assuming global +X
    # must point to the subject's right.
    frontal_normal_global = normalize(np.cross(up, pelvis_horizontal_hat))

    thigh = right_knee - right_hip
    thigh_global_frontal = (
        thigh
        - np.sum(thigh * frontal_normal_global, axis=1, keepdims=True)
        * frontal_normal_global
    )
    thigh_global_frontal = normalize(thigh_global_frontal)

    global_lateral = np.sum(thigh_global_frontal * pelvis_horizontal_hat, axis=1)
    global_down = np.sum(thigh_global_frontal * down, axis=1)

    global_thigh_angle = np.degrees(
        np.arctan2(global_lateral, global_down)
    )

    # Previous simplified pelvis-relative angle.
    pelvis_up = up - np.sum(up * pelvis_hat, axis=1, keepdims=True) * pelvis_hat
    pelvis_up = normalize(pelvis_up)
    pelvis_down = -pelvis_up

    pelvis_frontal_normal = normalize(np.cross(pelvis_up, pelvis_hat))
    thigh_pelvis_frontal = (
        thigh
        - np.sum(thigh * pelvis_frontal_normal, axis=1, keepdims=True)
        * pelvis_frontal_normal
    )
    thigh_pelvis_frontal = normalize(thigh_pelvis_frontal)

    pelvis_lateral = np.sum(thigh_pelvis_frontal * pelvis_hat, axis=1)
    pelvis_down_component = np.sum(thigh_pelvis_frontal * pelvis_down, axis=1)

    pelvis_relative_angle = np.degrees(
        np.arctan2(pelvis_lateral, pelvis_down_component)
    )

    expected_relative = wrap_180(global_thigh_angle - pelvis_obliquity)
    geometry_residual = wrap_180(pelvis_relative_angle - expected_relative)

    return pd.DataFrame({
        "pelvis_obliquity": pelvis_obliquity,
        "global_thigh_angle": global_thigh_angle,
        "pelvis_relative_angle": pelvis_relative_angle,
        "global_minus_pelvis": expected_relative,
        "geometry_residual": geometry_residual,
    })


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


def joint_angle_path(recording: Path, system: str) -> Path:
    return (
        recording
        / "validation"
        / system
        / "joint_angles"
        / "joint_angles_per_stride.csv"
    )


def load_geometry(
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

    for marker in ["left_hip", "right_hip", "right_knee"]:
        if marker not in points.columns.get_level_values(1):
            raise ValueError(f"{path} missing marker: {marker}")

    def xyz(marker: str) -> np.ndarray:
        return np.column_stack([
            points[("x", marker)].to_numpy(float),
            points[("y", marker)].to_numpy(float),
            points[("z", marker)].to_numpy(float),
        ])

    geom = calculate_frontal_geometry(
        left_hip=xyz("left_hip"),
        right_hip=xyz("right_hip"),
        right_knee=xyz("right_knee"),
    )

    geom.insert(0, "percent_gait_cycle", points["percent_gait_cycle"].to_numpy())
    geom.insert(0, "cycle", points["cycle"].to_numpy())
    geom.insert(0, "condition", condition)
    geom.insert(0, "system", system)
    geom.insert(0, "experiment", experiment)
    return geom


def load_existing_cardan(
    experiment: str,
    condition: str,
    recording: Path,
    system: str,
) -> pd.DataFrame:
    path = joint_angle_path(recording, system)
    if not path.exists():
        raise FileNotFoundError(f"Missing joint-angle file: {path}")

    df = pd.read_csv(path)

    required = {"joint", "side", "component", "angle", "cycle", "percent_gait_cycle"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    hip = df[
        (df["joint"] == "hip")
        & (df["side"] == "right")
        & (df["component"] == "abd_add")
    ][["cycle", "percent_gait_cycle", "angle"]].copy()

    if hip.empty:
        raise ValueError(
            f"No right hip / abd_add rows in {path}. "
            f"Available hip components: "
            f"{df.loc[df['joint'] == 'hip', ['side', 'component']].drop_duplicates().to_dict('records')}"
        )

    hip = hip.rename(columns={"angle": "existing_cardan_abd_add"})
    hip.insert(0, "condition", condition)
    hip.insert(0, "system", system)
    hip.insert(0, "experiment", experiment)
    return hip


def load_experiment(experiment: str) -> pd.DataFrame:
    cfg = EXPERIMENTS[experiment]
    rows = []

    for condition, recording in cfg["recordings"].items():
        for system in SYSTEMS:
            geom = load_geometry(experiment, condition, recording, system)
            cardan = load_existing_cardan(experiment, condition, recording, system)

            merged = geom.merge(
                cardan[
                    [
                        "experiment",
                        "system",
                        "condition",
                        "cycle",
                        "percent_gait_cycle",
                        "existing_cardan_abd_add",
                    ]
                ],
                on=[
                    "experiment",
                    "system",
                    "condition",
                    "cycle",
                    "percent_gait_cycle",
                ],
                how="left",
                validate="one_to_one",
            )

            missing_cardan = merged["existing_cardan_abd_add"].isna().mean()
            if missing_cardan > 0:
                print(
                    f"WARNING: {experiment}/{condition}/{system}: "
                    f"{missing_cardan:.1%} of trajectory samples did not match "
                    f"the existing Cardan stride file."
                )

            rows.append(merged)

    return pd.concat(rows, ignore_index=True)


# =============================================================================
# SUMMARIES
# =============================================================================

def waveform_summary(data: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "pelvis_obliquity",
        "global_thigh_angle",
        "pelvis_relative_angle",
        "global_minus_pelvis",
        "existing_cardan_abd_add",
    ]

    long = data.melt(
        id_vars=[
            "experiment",
            "system",
            "condition",
            "cycle",
            "percent_gait_cycle",
        ],
        value_vars=metrics,
        var_name="metric",
        value_name="angle",
    )

    return (
        long.groupby(
            ["experiment", "system", "condition", "metric", "percent_gait_cycle"],
            as_index=False,
        )["angle"]
        .agg(mean="mean", std="std")
    )


def diagnostic_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (experiment, system, condition), group in data.groupby(
        ["experiment", "system", "condition"]
    ):
        residual = group["geometry_residual"].dropna().to_numpy(float)

        paired = group[
            ["pelvis_relative_angle", "existing_cardan_abd_add"]
        ].dropna()

        if len(paired) >= 2:
            new = paired["pelvis_relative_angle"].to_numpy(float)
            old = paired["existing_cardan_abd_add"].to_numpy(float)
            difference = new - old

            correlation = (
                float(np.corrcoef(new, old)[0, 1])
                if np.std(new) > 0 and np.std(old) > 0
                else np.nan
            )
            rmse = float(np.sqrt(np.mean(difference ** 2)))
            mean_difference = float(np.mean(difference))
        else:
            correlation = np.nan
            rmse = np.nan
            mean_difference = np.nan

        rows.append({
            "experiment": experiment,
            "system": system,
            "condition": condition,
            "max_abs_geometry_identity_residual_deg": (
                float(np.max(np.abs(residual))) if len(residual) else np.nan
            ),
            "mean_abs_geometry_identity_residual_deg": (
                float(np.mean(np.abs(residual))) if len(residual) else np.nan
            ),
            "new_vs_existing_cardan_r": correlation,
            "new_vs_existing_cardan_rmse_deg": rmse,
            "new_minus_existing_cardan_mean_deg": mean_difference,
        })

    return pd.DataFrame(rows)


# =============================================================================
# PLOTTING
# =============================================================================

def rgba(hex_color: str, alpha: float) -> str:
    color = hex_color.lstrip("#")
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def add_mean_band(
    fig: go.Figure,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    color: str,
    name: str,
    legendgroup: str,
    showlegend: bool,
    row: int,
    col: int,
    dash: str = "solid",
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
            fillcolor=rgba(color, 0.12),
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
            y=mean,
            mode="lines",
            line=dict(color=color, width=2.0, dash=dash),
            name=name,
            legendgroup=legendgroup,
            showlegend=showlegend,
            hovertemplate="Gait cycle: %{x:.1f}%<br>Angle: %{y:.2f}°<extra></extra>",
        ),
        row=row,
        col=col,
    )


def make_geometry_components_figure(
    data: pd.DataFrame,
    experiment: str,
) -> go.Figure:
    """
    3 rows x 5 conditions:
      row 1 = pelvic obliquity
      row 2 = global frontal thigh angle
      row 3 = new pelvis-relative angle

    Qualisys and FMC-Hybrid are overlaid so we can see where disagreement enters.
    """
    cfg = EXPERIMENTS[experiment]
    summary = waveform_summary(data)

    row_metrics = [
        ("pelvis_obliquity", "Pelvic obliquity (°)"),
        ("global_thigh_angle", "Global frontal thigh angle (°)"),
        ("pelvis_relative_angle", "Pelvis-relative thigh angle (°)"),
    ]

    fig = make_subplots(
        rows=3,
        cols=5,
        shared_xaxes=True,
        shared_yaxes="rows",
        subplot_titles=[cfg["labels"][c] for c in cfg["order"]],
        horizontal_spacing=0.025,
        vertical_spacing=0.07,
    )

    for row, (metric, y_label) in enumerate(row_metrics, start=1):
        metric_data = summary[
            (summary["experiment"] == experiment)
            & (summary["metric"] == metric)
        ]

        ymin = (metric_data["mean"] - metric_data["std"].fillna(0)).min()
        ymax = (metric_data["mean"] + metric_data["std"].fillna(0)).max()
        pad = 0.08 * (ymax - ymin + 1e-9)
        y_range = [float(ymin - pad), float(ymax + pad)]

        for col, condition in enumerate(cfg["order"], start=1):
            for system in SYSTEMS:
                sub = metric_data[
                    (metric_data["condition"] == condition)
                    & (metric_data["system"] == system)
                ].sort_values("percent_gait_cycle")

                if sub.empty:
                    continue

                style = SYSTEM_STYLES[system]
                add_mean_band(
                    fig=fig,
                    x=sub["percent_gait_cycle"].to_numpy(),
                    mean=sub["mean"].to_numpy(),
                    std=sub["std"].fillna(0).to_numpy(),
                    color=style["color"],
                    name=SYSTEM_LABELS[system],
                    legendgroup=system,
                    showlegend=(row == 1 and col == 1),
                    row=row,
                    col=col,
                    dash=style["dash"],
                )

            fig.update_yaxes(range=y_range, row=row, col=col)

        fig.update_yaxes(
            title_text=f"<b>{y_label}</b>",
            title_font=dict(size=15),
            row=row,
            col=1,
        )

    for col in range(1, 6):
        fig.update_xaxes(
            title_text="<b>Gait cycle (%)</b>",
            title_font=dict(size=15),
            row=3,
            col=col,
        )

    fig.update_layout(
        template="simple_white",
        width=1800,
        height=900,
        font=dict(family="Arial", size=14, color="black"),
        legend=dict(
            orientation="h",
            x=0.5,
            y=-0.10,
            xanchor="center",
            yanchor="top",
            font=dict(size=14),
        ),
        margin=dict(l=85, r=15, t=45, b=70),
    )

    fig.update_xaxes(
        range=[0, 100],
        tickvals=[0, 50, 100],
        showline=True,
        mirror=True,
        linecolor="black",
        ticks="outside",
    )
    fig.update_yaxes(
        showline=True,
        mirror=True,
        linecolor="black",
        ticks="outside",
    )

    for ann in fig.layout.annotations:
        ann.font.size = 14
        ann.font.weight = "bold"

    return fig


def make_identity_check_figure(
    data: pd.DataFrame,
    experiment: str,
) -> go.Figure:
    """
    The new vector calculation should sit exactly on top of:
        global thigh angle - pelvic obliquity

    Plot one row for each system, with the five conditions in columns.
    """
    cfg = EXPERIMENTS[experiment]

    fig = make_subplots(
        rows=2,
        cols=5,
        shared_xaxes=True,
        shared_yaxes="rows",
        subplot_titles=[cfg["labels"][c] for c in cfg["order"]],
        horizontal_spacing=0.025,
        vertical_spacing=0.10,
    )

    colors = {
        "pelvis_relative_angle": "#1f77b4",
        "global_minus_pelvis": "#d62728",
    }
    labels = {
        "pelvis_relative_angle": "Vector calculation",
        "global_minus_pelvis": "Global thigh − pelvic obliquity",
    }

    for row, system in enumerate(SYSTEMS, start=1):
        sys = data[
            (data["experiment"] == experiment)
            & (data["system"] == system)
        ]

        y_values = []
        for metric in ["pelvis_relative_angle", "global_minus_pelvis"]:
            y_values.extend(sys[metric].dropna().tolist())

        ymin, ymax = min(y_values), max(y_values)
        pad = 0.08 * (ymax - ymin + 1e-9)
        y_range = [ymin - pad, ymax + pad]

        for col, condition in enumerate(cfg["order"], start=1):
            cond = sys[sys["condition"] == condition]

            for metric in ["pelvis_relative_angle", "global_minus_pelvis"]:
                summary = (
                    cond.groupby("percent_gait_cycle")[metric]
                    .agg(["mean", "std"])
                    .reset_index()
                )

                fig.add_trace(
                    go.Scatter(
                        x=summary["percent_gait_cycle"],
                        y=summary["mean"],
                        mode="lines",
                        line=dict(
                            color=colors[metric],
                            width=2.2 if metric == "pelvis_relative_angle" else 1.5,
                            dash="solid" if metric == "pelvis_relative_angle" else "dash",
                        ),
                        name=labels[metric],
                        legendgroup=metric,
                        showlegend=(row == 1 and col == 1),
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )

            fig.update_yaxes(range=y_range, row=row, col=col)

        fig.update_yaxes(
            title_text=f"<b>{SYSTEM_LABELS[system]} (°)</b>",
            title_font=dict(size=15),
            row=row,
            col=1,
        )

    for col in range(1, 6):
        fig.update_xaxes(
            title_text="<b>Gait cycle (%)</b>",
            title_font=dict(size=15),
            row=2,
            col=col,
        )

    fig.update_layout(
        template="simple_white",
        width=1800,
        height=650,
        font=dict(family="Arial", size=14, color="black"),
        legend=dict(
            orientation="h",
            x=0.5,
            y=-0.14,
            xanchor="center",
            yanchor="top",
            font=dict(size=14),
        ),
        margin=dict(l=90, r=15, t=45, b=75),
    )

    fig.update_xaxes(
        range=[0, 100],
        tickvals=[0, 50, 100],
        showline=True,
        mirror=True,
        linecolor="black",
        ticks="outside",
    )
    fig.update_yaxes(
        showline=True,
        mirror=True,
        linecolor="black",
        ticks="outside",
    )

    for ann in fig.layout.annotations:
        ann.font.size = 14
        ann.font.weight = "bold"

    return fig


def make_new_vs_existing_figure(
    data: pd.DataFrame,
    experiment: str,
) -> go.Figure:
    """
    Compare the NEW simplified pelvis-relative angle against the repository's
    EXISTING 3-D hip Cardan Y component (abd_add).

    They are not expected to be identical, because the existing Cardan hip
    proximal frame uses:
        z = hips_center -> neck_center
        x = left_hip -> right_hip

    whereas the new measure deliberately uses global vertical rather than the
    trunk/neck vector.
    """
    cfg = EXPERIMENTS[experiment]

    fig = make_subplots(
        rows=2,
        cols=5,
        shared_xaxes=True,
        shared_yaxes="rows",
        subplot_titles=[cfg["labels"][c] for c in cfg["order"]],
        horizontal_spacing=0.025,
        vertical_spacing=0.10,
    )

    metrics = [
        ("pelvis_relative_angle", "New pelvis-relative", "#1f77b4", "solid"),
        ("existing_cardan_abd_add", "Existing Cardan abd/add", "#9467bd", "dash"),
    ]

    for row, system in enumerate(SYSTEMS, start=1):
        sys = data[
            (data["experiment"] == experiment)
            & (data["system"] == system)
        ]

        values = pd.concat(
            [
                sys["pelvis_relative_angle"],
                sys["existing_cardan_abd_add"],
            ]
        ).dropna()

        ymin, ymax = values.min(), values.max()
        pad = 0.08 * (ymax - ymin + 1e-9)
        y_range = [float(ymin - pad), float(ymax + pad)]

        for col, condition in enumerate(cfg["order"], start=1):
            cond = sys[sys["condition"] == condition]

            for metric, label, color, dash in metrics:
                summary = (
                    cond.groupby("percent_gait_cycle")[metric]
                    .agg(["mean", "std"])
                    .reset_index()
                )

                fig.add_trace(
                    go.Scatter(
                        x=summary["percent_gait_cycle"],
                        y=summary["mean"],
                        mode="lines",
                        line=dict(color=color, width=2.0, dash=dash),
                        name=label,
                        legendgroup=metric,
                        showlegend=(row == 1 and col == 1),
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )

            fig.update_yaxes(range=y_range, row=row, col=col)

        fig.update_yaxes(
            title_text=f"<b>{SYSTEM_LABELS[system]} (°)</b>",
            title_font=dict(size=15),
            row=row,
            col=1,
        )

    for col in range(1, 6):
        fig.update_xaxes(
            title_text="<b>Gait cycle (%)</b>",
            title_font=dict(size=15),
            row=2,
            col=col,
        )

    fig.update_layout(
        template="simple_white",
        width=1800,
        height=650,
        font=dict(family="Arial", size=14, color="black"),
        legend=dict(
            orientation="h",
            x=0.5,
            y=-0.14,
            xanchor="center",
            yanchor="top",
            font=dict(size=14),
        ),
        margin=dict(l=90, r=15, t=45, b=75),
    )

    fig.update_xaxes(
        range=[0, 100],
        tickvals=[0, 50, 100],
        showline=True,
        mirror=True,
        linecolor="black",
        ticks="outside",
    )
    fig.update_yaxes(
        showline=True,
        mirror=True,
        linecolor="black",
        ticks="outside",
    )

    for ann in fig.layout.annotations:
        ann.font.size = 14
        ann.font.weight = "bold"

    return fig


def save_figure(fig: go.Figure, stem: Path) -> None:
    fig.write_html(stem.with_suffix(".html"))
    fig.write_image(stem.with_suffix(".png"), scale=2)
    fig.write_image(stem.with_suffix(".pdf"))


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    all_data = []

    for experiment in EXPERIMENTS:
        print(f"\n{'=' * 80}")
        print(f"Loading {experiment}")
        print(f"{'=' * 80}")

        data = load_experiment(experiment)
        all_data.append(data)

        diag = diagnostic_summary(data)

        print("\nGeometry identity check")
        print(
            diag[
                [
                    "system",
                    "condition",
                    "max_abs_geometry_identity_residual_deg",
                ]
            ]
            .round(8)
            .to_string(index=False)
        )

        print("\nNew simplified angle versus existing Cardan abd/add")
        print(
            diag[
                [
                    "system",
                    "condition",
                    "new_vs_existing_cardan_r",
                    "new_vs_existing_cardan_rmse_deg",
                    "new_minus_existing_cardan_mean_deg",
                ]
            ]
            .round(3)
            .to_string(index=False)
        )

        diag.to_csv(
            OUTPUT_DIR / f"{experiment}_diagnostic_summary.csv",
            index=False,
        )

        waveform_summary(data).to_csv(
            OUTPUT_DIR / f"{experiment}_waveform_summary.csv",
            index=False,
        )

        save_figure(
            make_geometry_components_figure(data, experiment),
            OUTPUT_DIR / f"{experiment}_geometry_components",
        )

        save_figure(
            make_identity_check_figure(data, experiment),
            OUTPUT_DIR / f"{experiment}_geometry_identity_check",
        )

        save_figure(
            make_new_vs_existing_figure(data, experiment),
            OUTPUT_DIR / f"{experiment}_new_vs_existing_cardan",
        )

    all_data = pd.concat(all_data, ignore_index=True)
    all_data.to_csv(
        OUTPUT_DIR / "all_adjustments_frontal_hip_diagnostic_per_stride.csv",
        index=False,
    )

    all_diagnostics = diagnostic_summary(all_data)
    all_diagnostics.to_csv(
        OUTPUT_DIR / "all_adjustments_diagnostic_summary.csv",
        index=False,
    )

    print(f"\nSaved all diagnostic outputs to:\n{OUTPUT_DIR}")
