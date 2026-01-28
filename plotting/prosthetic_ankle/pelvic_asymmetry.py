import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

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

COND_ORDER = ["neg_5", "neg_25", "neutral", "pos_25", "pos_5"]

COND_COLORS = {
    "neg_5": "#94342b",
    "neg_25": "#d39182",
    "neutral": "#524F4F",
    "pos_25": "#7bb6c6",
    "pos_5": "#447c8e",
}

COND_LABELS = {
    "neg_5": "-5 cm",
    "neg_25": "-2.5 cm",
    "neutral": "Neutral",
    "pos_25": "+2.5 cm",
    "pos_5": "+5 cm",
}

SYSTEM_LABELS = {
    "qualisys": "Qualisys",
    "mediapipe_dlc": "FreeMoCap",
}

SYSTEMS_ORDER = ["mediapipe_dlc", "qualisys"]  # columns: FreeMoCap | Qualisys
SIDES_ORDER = ["left", "right"]  # rows: Left hip | Right hip

SIDE_LABELS = {
    "left": "Left hip height (mm)",
    "right": "Right hip height (mm)",
}

ERRORBAR_STEP = 10
MAX_JITTER = 1.0
STANCE_SWING_BOUNDARY = 60

# Figure settings
FIG_W_IN = 3.45
FIG_H_IN = 4.2
DPI = 300

W = int(FIG_W_IN * DPI)
H = int(FIG_H_IN * DPI)

BASE = 14
TICK = 12
LEG = 11
TITLE = 14


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def get_hip_curve(traj_df: pd.DataFrame, marker: str, axis: str = "z"):
    """Extract mean and std hip trajectory curves from long-format trajectories dataframe."""
    mean_sub = (
        traj_df[
            (traj_df["marker"] == marker)
            & (traj_df["axis"] == axis)
            & (traj_df["stat"] == "mean")
        ]
        .sort_values("percent_gait_cycle")
    )
    std_sub = (
        traj_df[
            (traj_df["marker"] == marker)
            & (traj_df["axis"] == axis)
            & (traj_df["stat"] == "std")
        ]
        .sort_values("percent_gait_cycle")
    )
    if mean_sub.empty:
        raise ValueError(f"No rows found for marker={marker}, axis={axis}, stat=mean")

    percent = mean_sub["percent_gait_cycle"].to_numpy()
    mean_values = mean_sub["value"].to_numpy().astype(float)
    std_values = std_sub["value"].to_numpy().astype(float) if not std_sub.empty else np.zeros_like(mean_values)
    
    return percent, mean_values, std_values


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


# -------------------------------------------------------------------
# PLOTTING
# -------------------------------------------------------------------

def plot_hip_trajectories_grid(recordings: dict, out_path: Path | str | None = None) -> go.Figure:
    subplot_titles = [SYSTEM_LABELS.get(s, s) for s in SYSTEMS_ORDER]

    fig = make_subplots(
        rows=len(SIDES_ORDER),
        cols=len(SYSTEMS_ORDER),
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        shared_yaxes="rows",
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
    )

    # First pass: collect all data and compute y-ranges per row
    all_curves = {}
    row_yranges = {side: {"min": np.inf, "max": -np.inf} for side in SIDES_ORDER}

    for cond in COND_ORDER:
        if cond not in recordings:
            continue
        base = recordings[cond]

        for system in SYSTEMS_ORDER:
            traj_csv = base / "validation" / system / "trajectories" / "trajectories_per_stride_summary_stats.csv"
            if not traj_csv.exists():
                print(f"⚠️ Missing: {traj_csv}")
                continue

            traj_df = pd.read_csv(traj_csv)

            for side in SIDES_ORDER:
                marker = f"{side}_hip"
                try:
                    percent, mean_vals, std_vals = get_hip_curve(traj_df, marker=marker, axis="z")
                    all_curves[(cond, system, side)] = (percent, mean_vals, std_vals)
                    row_yranges[side]["min"] = min(row_yranges[side]["min"], (mean_vals - std_vals).min())
                    row_yranges[side]["max"] = max(row_yranges[side]["max"], (mean_vals + std_vals).max())
                except Exception as e:
                    print(f"Error: {cond}, {system}, {side}: {e}")

    # Compute jitter offsets per condition
    n_cond = len(COND_ORDER)
    if n_cond > 1:
        offsets = np.linspace(-MAX_JITTER, MAX_JITTER, n_cond)
    else:
        offsets = np.array([0.0])
    cond_offset = dict(zip(COND_ORDER, offsets))

    # Second pass: plot
    for row_idx, side in enumerate(SIDES_ORDER, start=1):
        # Compute y-range with padding
        ymin = row_yranges[side]["min"]
        ymax = row_yranges[side]["max"]
        pad = 0.08 * (ymax - ymin + 1e-9)
        ylo, yhi = ymin - pad, ymax + pad

        for col_idx, system in enumerate(SYSTEMS_ORDER, start=1):
            # Stance/swing boundary
            fig.add_vline(
                x=STANCE_SWING_BOUNDARY,
                line=dict(color="gray", width=1, dash="dash"),
                row=row_idx,
                col=col_idx,
            )

            for cond in COND_ORDER:
                key = (cond, system, side)
                if key not in all_curves:
                    continue

                percent, mean_vals, std_vals = all_curves[key]
                color = COND_COLORS.get(cond, "#555")
                display_label = COND_LABELS.get(cond, cond)
                system_label = SYSTEM_LABELS.get(system, system)

                # Mean line
                fig.add_trace(
                    go.Scatter(
                        x=percent,
                        y=mean_vals,
                        mode="lines",
                        name=display_label if (row_idx == 1 and col_idx == 1) else None,
                        legendgroup=cond,
                        line=dict(color=color, width=1.75),
                        showlegend=(row_idx == 1 and col_idx == 1),
                        hovertemplate=(
                            f"<b>{system_label} – {side.capitalize()} hip – {display_label}</b><br>"
                            "Gait cycle: %{x:.1f}%<br>"
                            "Height: %{y:.1f} mm<br>"
                            "<extra></extra>"
                        ),
                    ),
                    row=row_idx,
                    col=col_idx,
                )

                # Jittered SD error bars
                idx = np.arange(0, len(percent), ERRORBAR_STEP)
                x_base = percent[idx]
                m_err = mean_vals[idx]
                sd_err = std_vals[idx]

                x_err = x_base + cond_offset[cond]
                if x_err.size > 0:
                    x_err[0] = x_base[0]
                    x_err[-1] = x_base[-1]

                r, g, b = hex_to_rgb(color)

                fig.add_trace(
                    go.Scatter(
                        x=x_err,
                        y=m_err,
                        mode="markers",
                        legendgroup=cond,
                        showlegend=False,
                        marker=dict(
                            color=color,
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
                            f"<b>{system_label} – {side.capitalize()} hip – {display_label}</b><br>"
                            "Gait cycle: %{x:.1f}%<br>"
                            "Mean: %{y:.1f} mm<br>"
                            "SD: %{error_y.array:.1f} mm<br>"
                            "<extra></extra>"
                        ),
                    ),
                    row=row_idx,
                    col=col_idx,
                )

            fig.update_yaxes(range=[ylo, yhi], row=row_idx, col=col_idx)

        # Y-axis title (left column only)
        fig.update_yaxes(
            title_text=f"<b>{SIDE_LABELS[side]}</b>",
            title_font=dict(size=BASE),
            row=row_idx,
            col=1,
        )

    # X-axis labels (bottom row only)
    for col_idx in range(1, len(SYSTEMS_ORDER) + 1):
        fig.update_xaxes(
            title_text="<b>Gait cycle (%)</b>",
            title_font=dict(size=BASE),
            row=len(SIDES_ORDER),
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
            y=-0.05,
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

    if out_path:
        out_path = Path(out_path)
        fig.write_html(out_path.with_suffix(".html"))
        print(f"Saved: {out_path.with_suffix('.html')}")

    return fig


# -------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------

if __name__ == "__main__":
    fig = plot_hip_trajectories_grid(recordings, out_path="hip_trajectories")
    fig.show()