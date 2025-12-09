from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ----------------------- Inputs -----------------------

CONDITIONS = {
    "neg_6_0": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_20_59_TF01_toe_angle_neg_6_trial_1",
    "neg_3_0": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_25_38_TF01_toe_angle_neg_3_trial_1",
    "neutral": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_28_46_TF01_toe_angle_neutral_trial_1",
    "pos_3_0": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_31_49_TF01_toe_angle_pos_3_trial_1",
    "pos_6_0": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_34_37_TF01_toe_angle_pos_6_trial_1",
}
TRACKER = "mediapipe_dlc"

# reference direction (line of progression) and ground normal
REF_DIRECTION = np.array([0, 1, 0], dtype=float)  # forward
GROUND_NORMAL = np.array([0, 0, 1], dtype=float)  # up

# Plotting config
COND_ORDER = ['neg_6_0', 'neg_3_0', 'neutral', 'pos_3_0', 'pos_6_0']
COND_COLORS = {
    'neg_6_0': "#94342b",
    'neg_3_0': "#d39182",
    'neutral': "#524F4F",
    'pos_3_0': "#7bb6c6",
    'pos_6_0': "#447c8e",
}
COND_LABELS = {
    'neutral': 'Neutral',
    'neg_6_0': 'Toe-in (-6°)',
    'pos_6_0': 'Toe-out (+6°)',
    'neg_3_0': 'Toe-in (-3°)',
    'pos_3_0': 'Toe-out (+3°)'
}

ERRORBAR_STEP = 5   # every 5th point along the curve
MAX_JITTER = 1.0    # % gait cycle


# -------------------- Helpers --------------------


def calculate_foot_progression_angle(
    foot_vector: np.ndarray,
    reference_vector: np.ndarray,
    axis_of_rotation: np.ndarray,
) -> float:
    """
    FPA = signed angle between reference (progression) and foot long axis,
    both projected to plane ⟂ axis_of_rotation. Degrees.
    """
    n = axis_of_rotation.astype(float)
    n /= (np.linalg.norm(n) + 1e-12)

    a = reference_vector.astype(float)
    b = foot_vector.astype(float)

    # project to plane
    a_proj = a - np.dot(a, n) * n
    b_proj = b - np.dot(b, n) * n

    # unit vectors
    a_hat = a_proj / (np.linalg.norm(a_proj) + 1e-12)
    b_hat = b_proj / (np.linalg.norm(b_proj) + 1e-12)

    # signed angle about n
    sin_theta = np.dot(np.cross(a_hat, b_hat), n)
    cos_theta = np.dot(a_hat, b_hat)
    return np.degrees(np.arctan2(sin_theta, cos_theta))


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def build_heel_to_toe_summary(
    conditions: dict[str, str | Path],
    tracker: str = TRACKER,
) -> pd.DataFrame:
    """
    For each condition:
      - load joint_trajectory_by_stride
      - build toe-heel vectors
      - compute per-sample FPA
    Returns one concatenated DataFrame (heel_to_toe_summary).
    """
    all_rows: list[pd.DataFrame] = []

    for condition, path in conditions.items():
        path = Path(path)
        traj_path = path / "validation" / tracker / f"{tracker}_joint_trajectory_by_stride.csv"
        df = pd.read_csv(traj_path)

        toe_heel = df[df["marker"].isin(["heel", "toe"])]

        toe_heel_means = (
            toe_heel
            .groupby(['stride', 'percent_gait_cycle', 'system', 'marker'], as_index=False)[['x', 'y', 'z']]
            .mean()
        )

        pivoted = toe_heel_means.pivot_table(
            index=["system", "stride", "percent_gait_cycle"],
            columns="marker",
            values=["x", "y", "z"],
        )

        # toe-heel vector per stride, %GC
        vectors = {}
        for coord in ["x", "y", "z"]:
            vectors[coord] = pivoted[(coord, "toe")] - pivoted[(coord, "heel")]

        heel_to_toe = pd.DataFrame(vectors).reset_index()
        heel_to_toe["condition"] = condition

        # drop rows where either heel or toe was missing
        heel_to_toe = heel_to_toe.dropna(subset=["x", "y", "z"])

        # FPA per stride, per %GC
        heel_to_toe["fpa"] = heel_to_toe.apply(
            lambda row: calculate_foot_progression_angle(
                np.array([row["x"], row["y"], row["z"]]),
                reference_vector=REF_DIRECTION,
                axis_of_rotation=GROUND_NORMAL,
            ),
            axis=1,
        )

        all_rows.append(heel_to_toe)

    return pd.concat(all_rows, ignore_index=True)


def export_stance_summary(heel_to_toe_summary: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    """
    Restrict to stance (<=60% GC), average FPA per stride, then
    compute mean/SD/count per system/condition. Saves to CSV and
    returns the summary DataFrame.
    """
    stance = heel_to_toe_summary.query("percent_gait_cycle <= 60").copy()

    per_stride = (
        stance
        .groupby(['system', 'condition', 'stride'])['fpa']
        .mean()
        .reset_index()
    )

    summary = (
        per_stride
        .groupby(['system', 'condition'])['fpa']
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )

    summary = summary.round({'mean': 3, 'std': 3})
    summary.to_csv(csv_path, index=False)
    return summary


def make_fpa_system_comparison_figure(
    heel_to_toe_summary: pd.DataFrame,
    tracker: str = TRACKER,
    errorbar_step: int = ERRORBAR_STEP,
    max_jitter: float = MAX_JITTER,
) -> go.Figure:
    """
    Two-row figure:
      row 1: FreeMoCap (tracker)
      row 2: Qualisys
    Mean FPA curves for each condition + jittered SD error bars.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("FreeMoCap", "Qualisys"),
        shared_xaxes=True,
        vertical_spacing=0.12,
        row_heights=[0.5, 0.5],
    )

    systems_order = [tracker, "qualisys"]

    # precompute per-condition jitter offsets (shared across systems)
    n_cond = len(COND_ORDER)
    if n_cond > 1:
        offsets = np.linspace(-max_jitter, max_jitter, n_cond)
    else:
        offsets = np.array([0.0])
    cond_offset = dict(zip(COND_ORDER, offsets))

    for row_idx, system in enumerate(systems_order, 1):
        for condition in COND_ORDER:
            data = heel_to_toe_summary[
                (heel_to_toe_summary["condition"] == condition)
                & (heel_to_toe_summary["system"] == system)
            ]
            if data.empty:
                continue

            grouped = (
                data.groupby("percent_gait_cycle")["fpa"]
                .agg(["mean", "std"])
                .reset_index()
            )

            # mean line
            fig.add_trace(
                go.Scatter(
                    x=grouped["percent_gait_cycle"],
                    y=grouped["mean"],
                    mode="lines",
                    name=COND_LABELS.get(condition, condition) if row_idx == 1 else None,
                    line=dict(color=COND_COLORS[condition], width=3),
                    legendgroup=condition,
                    showlegend=(row_idx == 1),
                    hovertemplate=(
                        f"<b>{system.capitalize()} - "
                        f"{COND_LABELS.get(condition, condition)}</b><br>"
                        "Gait Cycle: %{x}%<br>"
                        "FPA: %{y:.1f}°<br>"
                        "<extra></extra>"
                    ),
                ),
                row=row_idx,
                col=1,
            )

            # jittered SD error bars
            x_all = grouped["percent_gait_cycle"].to_numpy()
            m_all = grouped["mean"].to_numpy()
            sd_all = grouped["std"].to_numpy()

            if len(x_all) == 0:
                continue

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
                        size=6,
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
                        f"<b>{system.capitalize()} - "
                        f"{COND_LABELS.get(condition, condition)}</b><br>"
                        "Gait Cycle: %{x:.1f}%<br>"
                        "Mean FPA: %{y:.1f}°<br>"
                        "SD: %{error_y.array:.1f}°<br>"
                        "<extra></extra>"
                    ),
                ),
                row=row_idx,
                col=1,
            )

        # zero-line
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray",
            opacity=0.5,
            row=row_idx,
            col=1,
        )

    # axes & layout
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_xaxes(title_text="Gait Cycle (%)", row=2, col=1)
    fig.update_yaxes(title_text="FPA (degrees)", row=1, col=1)
    fig.update_yaxes(title_text="FPA (degrees)", row=2, col=1)

    fig.update_layout(
        title="Foot Progression Angle Throughout Gait Cycle: System Comparison",
        hovermode="x unified",
        height=800,
        template="plotly_white",
        font=dict(size=20),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            borderwidth=1,
            traceorder="normal",
        ),
        margin=dict(b=120),
    )

    return fig


# -------------------- Run script --------------------


if __name__ == "__main__":
    heel_to_toe_summary = build_heel_to_toe_summary(CONDITIONS, tracker=TRACKER)

    # stance table for paper / PPT
    export_stance_summary(heel_to_toe_summary, Path("stance_fpa_summary.csv"))
    summary_stats = (
    heel_to_toe_summary
        .groupby(["condition", "system"])["fpa"]
        .agg(["mean", "std"])
        .reset_index()
   )
    
    pivot = summary_stats.pivot(
    index="condition",
    columns="system",
    values="mean"
    ).reset_index()

    pivot["delta_mean"] = pivot["mediapipe_dlc"] - pivot["qualisys"]
    overall_delta_mean = pivot["delta_mean"].mean()
    overall_delta_std  = pivot["delta_mean"].std(ddof=1)

    # figure
    fig1 = make_fpa_system_comparison_figure(heel_to_toe_summary, tracker=TRACKER)
    fig1.write_html("foot_progression_angle_system_comparison.html")
    # or fig1.show()
