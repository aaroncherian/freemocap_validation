from __future__ import annotations

from pathlib import Path
import re

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

CONDITION_STYLE: dict[str, dict[str, str]] = {
    "neg_5_6": {"line": "#94342b"},
    "neg_2_8": {"line": "#d39182"},
    "neutral": {"line": "#524F4F"},
    "pos_2_8": {"line": "#7bb6c6"},
    "pos_5_6": {"line": "#447c8e"},
}

SYSTEM_LABELS = {
    "mediapipe_dlc": "FreeMoCap",
    "qualisys": "Qualisys",
}

STANCE_SWING_BOUNDARY = 60  # percent gait cycle


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    s = hex_color.lstrip("#")
    if not re.fullmatch(r"[0-9A-Fa-f]{6}", s):
        raise ValueError(f"Invalid hex color: {hex_color}")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def load_angle_summary_for_tracker(
    conditions: dict[str, Path | str],
    tracker_dir: str,
    *,
    joint: str,
    side: str = "right",
    component: str = "dorsi_plantar",
) -> pd.DataFrame:
    """
    Read:
        <root>/validation/<tracker_dir>/joint_angles/joint_angles_per_stride_summary_stats.csv
    and return:
        ['system', 'condition', 'joint', 'percent_gait_cycle', 'mean', 'std']
    """
    all_summaries: list[pd.DataFrame] = []

    for cond, root in conditions.items():
        root = Path(root)
        csv_path = (
            root
            / "validation"
            / tracker_dir
            / "joint_angles"
            / "joint_angles_per_stride_summary_stats.csv"
        )
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing summary CSV for condition '{cond}' and tracker '{tracker_dir}': {csv_path}"
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

        wide = (
            df.pivot(index=["percent_gait_cycle"], columns="stat", values="value")
            .reset_index()
        )

        if not {"mean", "std"}.issubset(wide.columns):
            raise ValueError(
                f"Expected 'mean' and 'std' in stat column of {csv_path}, got {wide.columns}"
            )

        wide["system"] = tracker_dir
        wide["condition"] = cond
        wide["joint"] = joint

        wide = wide[["system", "condition", "joint", "percent_gait_cycle", "mean", "std"]]
        all_summaries.append(wide)

    out = pd.concat(all_summaries, ignore_index=True)
    print(f"[{tracker_dir}] loaded rows ({joint=}):", len(out))
    return out


# -------------------------------------------------------------------
# PLOTTING
# -------------------------------------------------------------------

def make_knee_and_ankle_figure(
    summary: pd.DataFrame,
    out_path: Path,
    *,
    joints_in_rows: list[str] = ("knee", "ankle"),
    systems_in_cols: list[str] = ("mediapipe_dlc", "qualisys"),
    errorbar_step: int = 10,
    max_jitter: float = 1.0,
    flip_sign_for: set[str] | None = None,
) -> Path:
    """
    `summary` must have:
        ['system','condition','joint','percent_gait_cycle','mean','std']
    """
    flip_sign_for = flip_sign_for or set()

    FIG_W_IN = 2
    FIG_H_IN = 1.6
    DPI = 300

    W = int(FIG_W_IN * DPI)
    H = int(FIG_H_IN * DPI)

    BASE = 16
    TICK = 14
    LEG = 14
    TITLE = 14

    systems = [s for s in systems_in_cols if s in summary["system"].unique()]
    joints = [j for j in joints_in_rows if j in summary["joint"].unique()]

    if not systems:
        raise ValueError("No known systems found in summary['system'].")
    if not joints:
        raise ValueError("No known joints found in summary['joint'].")

    # Use display labels for subplot titles
    subplot_titles = [SYSTEM_LABELS.get(s, s) for s in systems]

    fig = make_subplots(
        rows=len(joints),
        cols=len(systems),
        shared_xaxes=True,
        shared_yaxes="rows",  # share y-axis within each row
        subplot_titles=subplot_titles,
        vertical_spacing=0.10,
        horizontal_spacing=0.05,
    )

    conditions = [c for c in CONDITION_ORDER if c in set(summary["condition"].unique())]
    conditions += [c for c in sorted(summary["condition"].unique()) if c not in conditions]

    if len(conditions) > 1:
        offsets = np.linspace(-max_jitter, max_jitter, len(conditions))
    else:
        offsets = np.array([0.0])
    cond_offset = dict(zip(conditions, offsets))

    def yrange_for(sub: pd.DataFrame) -> tuple[float, float]:
        ymin = (sub["mean"] - sub["std"]).min()
        ymax = (sub["mean"] + sub["std"]).max()
        pad = 0.08 * (ymax - ymin + 1e-9)
        return float(ymin - pad), float(ymax + pad)

    for row_idx, joint in enumerate(joints, start=1):
        sub_joint = summary[summary["joint"] == joint].copy()
        if joint in flip_sign_for:
            sub_joint["mean"] *= -1

        ylo, yhi = yrange_for(sub_joint)

        for col_idx, system in enumerate(systems, start=1):
            # Add stance/swing boundary line
            # fig.add_vline(
            #     x=STANCE_SWING_BOUNDARY,
            #     line=dict(color="gray", width=1, dash="dash"),
            #     row=row_idx,
            #     col=col_idx,
            # )

            # Add zero reference line for ankle
            if joint == "ankle":
                fig.add_hline(
                    y=0,
                    line=dict(color="gray", width=0.75, dash="dot"),
                    row=row_idx,
                    col=col_idx,
                )

            for cond in conditions:
                sub = sub_joint[
                    (sub_joint["system"] == system) & (sub_joint["condition"] == cond)
                ].sort_values("percent_gait_cycle")
                if sub.empty:
                    continue

                line_color = CONDITION_STYLE.get(cond, {"line": "#555"})["line"]
                display_label = CONDITION_LABELS.get(cond, cond)
                system_label = SYSTEM_LABELS.get(system, system)

                x = sub["percent_gait_cycle"].to_numpy()
                m = sub["mean"].to_numpy()
                sd = sub["std"].to_numpy()

                # mean line
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=m,
                        mode="lines",
                        name=display_label if (row_idx == 1 and col_idx == 1) else None,
                        legendgroup=cond,
                        line=dict(color=line_color, width=1.75),
                        showlegend=(row_idx == 1 and col_idx == 1),
                        hovertemplate=(
                            f"<b>{system_label} – {joint.capitalize()} – {display_label}</b><br>"
                            "Gait cycle: %{x:.1f}%<br>"
                            "Angle: %{y:.1f}°<br>"
                            "<extra></extra>"
                        ),
                    ),
                    row=row_idx,
                    col=col_idx,
                )

                # jittered SD bars
                idx = np.arange(0, len(x), errorbar_step)
                x_base = x[idx]
                m_err = m[idx]
                sd_err = sd[idx]

                x_err = x_base + cond_offset[cond]
                if x_err.size > 0:
                    x_err[0] = x_base[0]
                    x_err[-1] = x_base[-1]

                r, g, b = hex_to_rgb(line_color)

                fig.add_trace(
                    go.Scatter(
                        x=x_err,
                        y=m_err,
                        mode="markers",
                        legendgroup=cond,
                        showlegend=False,
                        marker=dict(
                            color=line_color,
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
                            f"<b>{system_label} – {joint.capitalize()} – {display_label}</b><br>"
                            "Gait cycle: %{x:.1f}%<br>"
                            "Mean angle: %{y:.1f}°<br>"
                            "SD: %{error_y.array:.1f}°<br>"
                            "<extra></extra>"
                        ),
                    ),
                    row=row_idx,
                    col=col_idx,
                )

            fig.update_yaxes(range=[ylo, yhi], row=row_idx, col=col_idx)

        # y-axis title on left column only
        joint_label = "Knee flexion" if joint == "knee" else "Ankle dorsiflexion"
        fig.update_yaxes(
            title_text=f"<b>{joint_label} (°)</b>",
            row=row_idx,
            col=1,
            title_font=dict(size=BASE),
        )

    # x-axis labels on bottom row
    for col_idx in range(1, len(systems) + 1):
        fig.update_xaxes(
            title_text="<b>Gait cycle (%)</b>",
            row=len(joints),
            col=col_idx,
            title_font=dict(size=BASE),
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
            y=-0.15,
            xanchor="center",
            yanchor="top",
            font=dict(size=LEG),
            tracegroupgap=4,
        ),
        margin=dict(l=58, r=8, t=28, b=62),
    )

    # Update subplot titles font (bold)
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

    fig.write_html(out_path)
    
    import plotly.io as pio
    pio.kaleido.scope.mathjax = None
    path_to_save = Path(r"C:\Users\aaron\Documents\prosthetics_paper")
    fig.write_image(path_to_save / "ankle_adjustment_plot.pdf")
    return out_path


# -------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------

def run_knee_and_ankle_summary(
    conditions: dict[str, str | Path],
    out_dir: str | Path = "angle_summary_plots",
) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    joints_to_load = [
        dict(joint="knee", side="right", component="flex_ext"),
        dict(joint="ankle", side="right", component="dorsi_plantar"),
    ]

    all_rows = []
    for jt in joints_to_load:
        for tracker in ("mediapipe_dlc", "qualisys"):
            all_rows.append(
                load_angle_summary_for_tracker(
                    {k: Path(v) for k, v in conditions.items()},
                    tracker_dir=tracker,
                    joint=jt["joint"],
                    side=jt["side"],
                    component=jt["component"],
                )
            )

    summary_all = pd.concat(all_rows, ignore_index=True)

    out_html = out_dir / "knee_and_ankle_system_comparison.html"

    make_knee_and_ankle_figure(
        summary_all,
        out_html,
        joints_in_rows=["knee", "ankle"],
        systems_in_cols=["mediapipe_dlc", "qualisys"],
        flip_sign_for={"ankle", "knee"},
    )

    return [out_html]


if __name__ == "__main__":
    conditions = {
        "neutral": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1",
        "neg_2_8": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_03_15_TF01_flexion_neg_2_8_trial_1",
        "neg_5_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_11_55_05_TF01_flexion_neg_5_6_trial_1",
        "pos_2_8": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_09_05_TF01_flexion_pos_2_8_trial_1",
        "pos_5_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_12_36_TF01_flexion_pos_5_6_trial_1",
    }

    outputs = run_knee_and_ankle_summary(conditions, out_dir="ankle_summary_plots")
    for p in outputs:
        print(p)