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
    'neg_6_0': '-6°',
    'pos_6_0': '+6°',
    'neg_3_0': '-3°',
    'pos_3_0': '+3°',
}

SYSTEM_LABELS = {
    "mediapipe_dlc": "FreeMoCap",
    "qualisys": "Qualisys",
}

ERRORBAR_STEP = 10
MAX_JITTER = 1.0
STANCE_SWING_BOUNDARY = 60


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
    Two-column figure: FreeMoCap | Qualisys side-by-side.
    Mean FPA curves for each condition + jittered SD error bars.
    """
    FIG_W_IN = 3.45
    FIG_H_IN = 2.4
    DPI = 300

    W = int(FIG_W_IN * DPI)
    H = int(FIG_H_IN * DPI)

    BASE = 16
    TICK = 14
    LEG = 14
    TITLE = 14

    systems_order = [tracker, "qualisys"]
    subplot_titles = [SYSTEM_LABELS.get(s, s) for s in systems_order]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.05,
    )

    # precompute per-condition jitter offsets
    n_cond = len(COND_ORDER)
    if n_cond > 1:
        offsets = np.linspace(-max_jitter, max_jitter, n_cond)
    else:
        offsets = np.array([0.0])
    cond_offset = dict(zip(COND_ORDER, offsets))

    # compute shared y-range across all data
    all_means = []
    all_stds = []
    for system in systems_order:
        for condition in COND_ORDER:
            data = heel_to_toe_summary[
                (heel_to_toe_summary["condition"] == condition)
                & (heel_to_toe_summary["system"] == system)
            ]
            if data.empty:
                continue
            grouped = data.groupby("percent_gait_cycle")["fpa"].agg(["mean", "std"])
            all_means.extend(grouped["mean"].tolist())
            all_stds.extend(grouped["std"].fillna(0).tolist())

    ymin = min(m - s for m, s in zip(all_means, all_stds))
    ymax = max(m + s for m, s in zip(all_means, all_stds))
    pad = 0.08 * (ymax - ymin + 1e-9)
    ylo, yhi = ymin - pad, ymax + pad

    for col_idx, system in enumerate(systems_order, 1):
        # stance/swing boundary
        fig.add_vline(
            x=STANCE_SWING_BOUNDARY,
            line=dict(color="gray", width=1, dash="dash"),
            row=1,
            col=col_idx,
        )

        # zero reference line
        fig.add_hline(
            y=0,
            line=dict(color="gray", width=0.75, dash="dot"),
            row=1,
            col=col_idx,
        )

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

            display_label = COND_LABELS.get(condition, condition)
            system_label = SYSTEM_LABELS.get(system, system)

            # mean line
            fig.add_trace(
                go.Scatter(
                    x=grouped["percent_gait_cycle"],
                    y=grouped["mean"],
                    mode="lines",
                    name=display_label if col_idx == 1 else None,
                    line=dict(color=COND_COLORS[condition], width=1.75),
                    legendgroup=condition,
                    showlegend=(col_idx == 1),
                    hovertemplate=(
                        f"<b>{system_label} – {display_label}</b><br>"
                        "Gait cycle: %{x:.1f}%<br>"
                        "FPA: %{y:.1f}°<br>"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=col_idx,
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
                        "Mean FPA: %{y:.1f}°<br>"
                        "SD: %{error_y.array:.1f}°<br>"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=col_idx,
            )

        fig.update_yaxes(range=[ylo, yhi], row=1, col=col_idx)

    # y-axis title on left column only
    fig.update_yaxes(
        title_text="<b>Foot progression angle (°)</b>",
        title_font=dict(size=BASE),
        row=1,
        col=1,
    )

    # x-axis label on both columns
    for col_idx in range(1, 3):
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
            y=-0.08,
            xanchor="center",
            yanchor="top",
            font=dict(size=LEG),
            tracegroupgap=4,
        ),
        margin=dict(l=58, r=8, t=28, b=62),
    )

    # Bold subplot titles
    for annotation in fig.layout.annotations:
        annotation.font.size = TITLE
        annotation.font.weight = "bold"

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
    heel_to_toe_summary = build_heel_to_toe_summary(CONDITIONS, tracker=TRACKER)
    import plotly.io as pio
    pio.kaleido.scope.mathjax = None

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
    overall_delta_std = pivot["delta_mean"].std(ddof=1)

    # figure
    fig1 = make_fpa_system_comparison_figure(heel_to_toe_summary, tracker=TRACKER)
    fig1.write_html("foot_progression_angle_system_comparison.html")
    fig1.write_image("fpa_plot.pdf")

    # or fig1.show()