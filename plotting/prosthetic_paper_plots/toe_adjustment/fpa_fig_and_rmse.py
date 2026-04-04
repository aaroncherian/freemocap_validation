from pathlib import Path
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


conditions = {
    "neg_6_0": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_20_59_TF01_toe_angle_neg_6_trial_1",
    "neg_3_0": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_25_38_TF01_toe_angle_neg_3_trial_1",
    "neutral": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_28_46_TF01_toe_angle_neutral_trial_1",
    "pos_3_0": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_31_49_TF01_toe_angle_pos_3_trial_1",
    "pos_6_0": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_34_37_TF01_toe_angle_pos_6_trial_1",
}

trackers = ["rtmpose_dlc", "qualisys"]

# reference direction (line of progression) and ground normal
a = np.array([0, 1, 0], dtype=float)
n = np.array([0, 0, 1], dtype=float)

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
    "mediapipe_dlc": "FreeMoCap-DLC",
    "qualisys": "Qualisys",
    "rtmpose_dlc": "FreeMoCap-DLC",
}

ERRORBAR_STEP = 10
MAX_JITTER = 1.0
STANCE_SWING_BOUNDARY = 60


def calculate_foot_progression_angle(
    foot_vector: np.ndarray,
    reference_vector: np.ndarray,
    axis_of_rotation: np.ndarray
) -> float:
    """
    FPA = signed angle between reference (progression) and foot long axis,
    both projected to plane ⟂ axis_of_rotation. Degrees. Positive ≈ toe-out
    with right-handed frame and +Z up.
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
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def make_fpa_system_comparison_figure(
    fpas: pd.DataFrame,
    tracker: str = "rtmpose_dlc",
    errorbar_step: int = ERRORBAR_STEP,
    max_jitter: float = MAX_JITTER,
) -> go.Figure:
    """
    Two-column figure: tracker | Qualisys side-by-side.
    Mean FPA curves for each condition + jittered SD error bars.
    """
    FIG_W_IN = 2
    FIG_H_IN = 1.0
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
    offsets = np.linspace(-max_jitter, max_jitter, n_cond) if n_cond > 1 else np.array([0.0])
    cond_offset = dict(zip(COND_ORDER, offsets))

    # compute shared y-range across all data
    all_means: list[float] = []
    all_stds: list[float] = []

    for system in systems_order:
        for condition in COND_ORDER:
            data = fpas[(fpas["condition"] == condition) & (fpas["system"] == system)]
            if data.empty:
                continue
            grouped = data.groupby("percent_gait_cycle")["fpa"].agg(["mean", "std"])
            all_means.extend(grouped["mean"].tolist())
            all_stds.extend(grouped["std"].fillna(0).tolist())

    if len(all_means) == 0:
        raise ValueError("No data found to plot. Check your fpas dataframe and system/condition labels.")

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
            data = fpas[(fpas["condition"] == condition) & (fpas["system"] == system)]
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
            y=-0.28,
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

def calculate_rmse(reference_values: np.ndarray, test_values:np.ndarray):
    
    error = test_values - reference_values
    squared_error = error**2
    return np.sqrt(np.mean(squared_error))


def calculate_fpa_rmse(
    fpas: pd.DataFrame):

    wide = fpas.pivot_table(
        index = ["cycle", "percent_gait_cycle", "condition"],
        columns = "system",
        values = "fpa"
    ).reset_index()


    fpa_rmse = pd.DataFrame()
    fpa_rmse['rmse']= wide.groupby(["condition", "cycle"]).apply(lambda x: calculate_rmse(np.array(x['qualisys']), np.array(x['rtmpose_dlc'])))
    
    mean_rmse = fpa_rmse.groupby("condition")["rmse"].mean().reset_index()
    std_rmse = fpa_rmse.groupby("condition")["rmse"].std().reset_index()

    total_rmse = mean_rmse["rmse"].mean()
    total_std = mean_rmse["rmse"].std()
    return total_rmse, total_std 

fpa_list = []


for condition, root_path in conditions.items():
    for tracker in trackers:
        vector_df = pd.DataFrame()

        path_to_csv = Path(root_path)/"validation"/tracker/"trajectories"/"trajectories_per_stride.csv"

        traj_df = pd.read_csv(path_to_csv)
        foot_heel_df = traj_df.query("marker in ['right_foot_index','right_heel']").pivot_table(
            index = ["cycle", "percent_gait_cycle"],
            columns = "marker",
            values = ["x", "y", "z"]
        ).reset_index()

        # vector_df = foot_heel_df[['cycle','percent_gait_cycle']].copy().reset_index(drop=True)
        for value in ["x", "y", "z"]:
            vector_df[value] = foot_heel_df[value]["right_foot_index"] - foot_heel_df[value]["right_heel"]
        vector_df["cycle"] = foot_heel_df["cycle"]
        vector_df["percent_gait_cycle"] = foot_heel_df["percent_gait_cycle"]
        fpa_df = vector_df[["cycle", "percent_gait_cycle"]].copy()
        fpa_df["fpa"] = vector_df.apply(lambda x: calculate_foot_progression_angle(
            np.array([x['x'], x['y'], x['z']]), 
            reference_vector=a, 
            axis_of_rotation=n), 
            axis = 1)
        fpa_df["system"] = tracker
        fpa_df["condition"] = condition
        
        fpa_list.append(fpa_df)

import plotly.io as pio
pio.kaleido.scope.mathjax = None
fpas = pd.concat(fpa_list, ignore_index=True)



calculate_fpa_rmse(fpas)

fig1 = make_fpa_system_comparison_figure(fpas, tracker="rtmpose_dlc")
# fig1.show()
path_to_save = Path(r"C:\Users\aaron\Documents\prosthetics_paper")
fig1.write_image(path_to_save / "fpa_plot.pdf")
fig1.write_image(path_to_save / "fpa_plot.png", scale=3)

mean_rmse, std_rmse = calculate_fpa_rmse(fpas)
print("FPA RMSE vs Qualisys across all conditions (mean ± std):", f"{mean_rmse:.2f}° ± {std_rmse:.2f}°")
