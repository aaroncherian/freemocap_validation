from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# ----------------------- Inputs -----------------------
conditions = {
    "neg_6_0": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_20_59_TF01_toe_angle_neg_6_trial_1",
    "neg_3_0": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_25_38_TF01_toe_angle_neg_3_trial_1",
    "neutral": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_28_46_TF01_toe_angle_neutral_trial_1",
    "pos_3_0": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_31_49_TF01_toe_angle_pos_3_trial_1",
    "pos_6_0": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_34_37_TF01_toe_angle_pos_6_trial_1",
}
tracker = "mediapipe_dlc"

# reference direction (line of progression) and ground normal
a = np.array([0, 1, 0], dtype=float)
n = np.array([0, 0, 1], dtype=float)

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

# ------------------------------------------------------
# Build per-stride toe-heel vectors and compute FPA
# ------------------------------------------------------
heel_to_toe_summary = pd.DataFrame()

for condition, path in conditions.items():
    trajectories_by_stride_path = Path(path) / "validation" / tracker / f"{tracker}_joint_trajectory_by_stride.csv"
    trajectories_by_stride_df = pd.read_csv(trajectories_by_stride_path)

    toe_heel = trajectories_by_stride_df[trajectories_by_stride_df["marker"].isin(["heel", "toe"])]

    toe_heel_means = (
        toe_heel
        .groupby(['stride', 'percent_gait_cycle', 'system', 'marker'], as_index=False)[['x', 'y', 'z']]
        .mean()
    )

    pivoted = toe_heel_means.pivot_table(
        index=["system", "stride", "percent_gait_cycle"],
        columns="marker",
        values=["x", "y", "z"]
    )

    # toe-heel vector per stride, %GC
    vectors = {}
    for coord in ["x", "y", "z"]:
        vectors[coord] = pivoted[(coord, "toe")] - pivoted[(coord, "heel")]

    heel_to_toe = pd.DataFrame(vectors).reset_index()
    heel_to_toe['condition'] = condition

    # drop rows where either heel or toe was missing
    heel_to_toe = heel_to_toe.dropna(subset=["x", "y", "z"])

    # FPA per stride, per %GC
    heel_to_toe['fpa'] = heel_to_toe.apply(
        lambda row: calculate_foot_progression_angle(
            np.array([row['x'], row['y'], row['z']]),
            reference_vector=a,
            axis_of_rotation=n
        ),
        axis=1
    )

    heel_to_toe_summary = pd.concat([heel_to_toe_summary, heel_to_toe], ignore_index=True)

heel_to_toe_stance = heel_to_toe_summary.query("percent_gait_cycle <= 60").copy()

per_stride = heel_to_toe_stance.groupby(['system', 'condition', 'stride'])['fpa'].mean().reset_index()
summary = per_stride.groupby(['system','condition'])['fpa'].agg(['mean','std','count']).reset_index()
summary = summary.round({'mean': 3, 'std': 3})
# Export to CSV (easy to pull into PowerPoint/Keynote/Excel)
summary.to_csv("stance_fpa_summary.csv", index=False)
# ------------------ Plotting config -------------------
# Use the colors you requested, mapped to the 6°/3° condition names
COND_ORDER = ['neg_6_0', 'neg_3_0', 'neutral', 'pos_3_0', 'pos_6_0']
COND_COLORS = {
    'neg_6_0': "#94342b",  # (was neg_5_6)
    'neg_3_0': "#d39182",  # (was neg_2_8)
    'neutral': "#524F4F",
    'pos_3_0': "#7bb6c6",  # (was pos_2_8)
    'pos_6_0': "#447c8e",  # (was pos_5_6)
}
condition_labels = {
    'neutral': 'Neutral',
    'neg_6_0': 'Toe-in (-6°)',
    'pos_6_0': 'Toe-out (+6°)',
    'neg_3_0': 'Toe-in (-3°)',
    'pos_3_0': 'Toe-out (+3°)'
}

# jitter + sampling settings for SD bars
errorbar_step = 5   # every 5th point along the curve
max_jitter = 1.0    # % gait cycle


def hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


# ====================
# Figure 1: FPA over gait (two rows: FMC vs QTM)
# ====================
fig1 = make_subplots(
    rows=2, cols=1,
    subplot_titles=("FreeMoCap","Qualisys"),
    shared_xaxes=True,
    vertical_spacing=0.12,
    row_heights=[0.5, 0.5]
)

systems_order = [tracker, 'qualisys']

# precompute per-condition jitter offsets (shared across systems)
n_cond = len(COND_ORDER)
if n_cond > 1:
    offsets = np.linspace(-max_jitter, max_jitter, n_cond)
else:
    offsets = np.array([0.0])
COND_OFFSET = {cond: off for cond, off in zip(COND_ORDER, offsets)}

for sys_idx, system in enumerate(systems_order, 1):
    for condition in COND_ORDER:
        data = heel_to_toe_summary[
            (heel_to_toe_summary['condition'] == condition) &
            (heel_to_toe_summary['system'] == system)
        ]
        if data.empty:
            continue

        grouped = (
            data.groupby('percent_gait_cycle')['fpa']
            .agg(['mean', 'std'])
            .reset_index()
        )

        # ---------- mean line (same as before) ----------
        fig1.add_trace(
            go.Scatter(
                x=grouped['percent_gait_cycle'],
                y=grouped['mean'],
                mode='lines',
                name=condition_labels.get(condition, condition) if sys_idx == 1 else None,
                line=dict(color=COND_COLORS[condition], width=3),
                legendgroup=condition,
                showlegend=(sys_idx == 1),
                hovertemplate=(
                    f"<b>{system.capitalize()} - "
                    f"{condition_labels.get(condition, condition)}</b><br>"
                    "Gait Cycle: %{x}%<br>"
                    "FPA: %{y:.1f}°<br>"
                    "<extra></extra>"
                )
            ),
            row=sys_idx, col=1
        )

        # ---------- jittered SD bars instead of shaded band ----------
        x_all = grouped['percent_gait_cycle'].to_numpy()
        m_all = grouped['mean'].to_numpy()
        sd_all = grouped['std'].to_numpy()

        if len(x_all) == 0:
            continue

        # subsample along the curve
        idx = np.arange(0, len(x_all), errorbar_step)
        x_base = x_all[idx]
        m_err  = m_all[idx]
        sd_err = sd_all[idx]

        # apply condition-specific jitter, but keep first/last anchored
        x_err = x_base + COND_OFFSET[condition]
        if x_err.size > 0:
            x_err[0]  = x_base[0]
            x_err[-1] = x_base[-1]

        r, g, b = hex_to_rgb(COND_COLORS[condition])

        fig1.add_trace(
            go.Scatter(
                x=x_err,
                y=m_err,
                mode='markers',
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
                    f"{condition_labels.get(condition, condition)}</b><br>"
                    "Gait Cycle: %{x:.1f}%<br>"
                    "Mean FPA: %{y:.1f}°<br>"
                    "SD: %{error_y.array:.1f}°<br>"
                    "<extra></extra>"
                )
            ),
            row=sys_idx, col=1
        )

    # zero-line
    fig1.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
        row=sys_idx,
        col=1
    )

# axes & layout (legend along bottom)
fig1.update_xaxes(title_text="", row=1, col=1)
fig1.update_xaxes(title_text="Gait Cycle (%)", row=2, col=1)
fig1.update_yaxes(title_text="FPA (degrees)", row=1, col=1)
fig1.update_yaxes(title_text="FPA (degrees)", row=2, col=1)

fig1.update_layout(
    title="Foot Progression Angle Throughout Gait Cycle: System Comparison",
    hovermode='x unified',
    height=800,
    template='plotly_white',
    font=dict(size=20),
    legend=dict(
        orientation="h",
        yanchor="top", y=-0.12,
        xanchor="center", x=0.5,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="gray",
        borderwidth=1,
        traceorder="normal"
    ),
    margin=dict(b=120),
)

fig1.write_html("foot_progression_angle_system_comparison.html")

# ====================
# Figure 2: One panel per condition; lines = systems
# ====================
fig2 = make_subplots(
    rows=1, cols=len(COND_ORDER),
    subplot_titles=[condition_labels[c] for c in COND_ORDER],
    shared_yaxes=True,
    horizontal_spacing=0.04
)

systems = list(heel_to_toe_summary['system'].dropna().unique())
system_colors = px.colors.qualitative.Set2[:len(systems)]

for i, condition in enumerate(COND_ORDER, start=1):
    data = heel_to_toe_summary[heel_to_toe_summary['condition'] == condition]
    for j, system in enumerate(systems):
        system_data = data[data['system'] == system]
        if system_data.empty:
            continue
        grouped = system_data.groupby('percent_gait_cycle')['fpa'].mean().reset_index()

        fig2.add_trace(
            go.Scatter(
                x=grouped['percent_gait_cycle'],
                y=grouped['fpa'],
                mode='lines',
                name=system if i == 1 else "",
                line=dict(color=system_colors[j], width=2),
                showlegend=(i == 1),
                legendgroup=system
            ),
            row=1, col=i
        )

# axes & layout
middle = (len(COND_ORDER) + 1) // 2
fig2.update_xaxes(title_text="Gait Cycle (%)", row=1, col=middle)
fig2.update_yaxes(title_text="FPA (degrees)", row=1, col=1)
fig2.update_layout(
    title="Foot Progression Angle by Tracking System",
    height=500,
    template='plotly_white',
    hovermode='x unified',
    legend=dict(
        orientation="h",
        yanchor="top", y=-0.18,
        xanchor="center", x=0.5
    ),
    margin=dict(b=120)
)

# ------------------ Show figs -------------------
fig1.show()
# fig2.show()
