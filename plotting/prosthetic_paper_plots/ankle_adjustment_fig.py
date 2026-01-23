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

CONDITION_STYLE: dict[str, dict[str, str]] = {
    "neg_5_6": {"line": "#94342b"},
    "neg_2_8": {"line": "#d39182"},
    "neutral": {"line": "#524F4F"},
    "pos_2_8": {"line": "#7bb6c6"},
    "pos_5_6": {"line": "#447c8e"},
}


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



def global_yrange(summ: pd.DataFrame) -> tuple[float, float]:
    ymin = (summ["mean"] - summ["std"]).min()
    ymax = (summ["mean"] + summ["std"]).max()
    pad = 0.05 * (ymax - ymin + 1e-9)
    return float(ymin - pad), float(ymax + pad)


# -------------------------------------------------------------------
# PLOTTING
# -------------------------------------------------------------------

def make_knee_and_ankle_figure(
    summary: pd.DataFrame,
    out_path: Path,
    *,
    joints_in_rows: list[str] = ("knee", "ankle"),
    systems_in_cols: list[str] = ("mediapipe_dlc", "qualisys"),
    errorbar_step: int = 5,
    max_jitter: float = 1.0,
    flip_sign_for: set[str] | None = None,  # e.g. {"ankle"} if you only want ankle flipped
) -> Path:
    """
    `summary` must have:
        ['system','condition','joint','percent_gait_cycle','mean','std']
    """
    flip_sign_for = flip_sign_for or set()

    # filter to what we actually have
    systems = [s for s in systems_in_cols if s in summary["system"].unique()]
    joints = [j for j in joints_in_rows if j in summary["joint"].unique()]

    if not systems:
        raise ValueError("No known systems found in summary['system'].")
    if not joints:
        raise ValueError("No known joints found in summary['joint'].")

    fig = make_subplots(
        rows=len(joints),
        cols=len(systems),
        shared_xaxes=True,
        shared_yaxes=False,  # knee/ankle ranges often differ; keep separate per row
        subplot_titles=[s for s in systems],  # top row titles
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )

    # condition order & jitter offsets (same logic as your current figure)
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
        pad = 0.05 * (ymax - ymin + 1e-9)
        return float(ymin - pad), float(ymax + pad)

    for row_idx, joint in enumerate(joints, start=1):
        # compute y-range per joint (across both systems)
        sub_joint = summary[summary["joint"] == joint].copy()
        if joint in flip_sign_for:
            sub_joint["mean"] *= -1

        ylo, yhi = yrange_for(sub_joint)

        for col_idx, system in enumerate(systems, start=1):
            for cond in conditions:
                sub = sub_joint[
                    (sub_joint["system"] == system) & (sub_joint["condition"] == cond)
                ].sort_values("percent_gait_cycle")
                if sub.empty:
                    continue

                line_color = CONDITION_STYLE.get(cond, {"line": "#555"})["line"]

                x = sub["percent_gait_cycle"].to_numpy()
                m = sub["mean"].to_numpy()
                sd = sub["std"].to_numpy()

                # mean line
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=m,
                        mode="lines",
                        name=cond if (row_idx == 1 and col_idx == 1) else None,
                        legendgroup=cond,
                        line=dict(color=line_color, width=3),
                        showlegend=(row_idx == 1 and col_idx == 1),
                        hovertemplate=(
                            f"<b>{system} – {joint} – {cond}</b><br>"
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
                            f"<b>{system} – {joint} – {cond}</b><br>"
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

        # row-specific y-axis title (left column only)
        fig.update_yaxes(
            title_text=f"<b>{joint.capitalize()} angle (°)</b>",
            row=row_idx,
            col=1,
            title_font=dict(size=24),
        )

    # x-axis labels on bottom row
    for col_idx in range(1, len(systems) + 1):
        fig.update_xaxes(
            title_text="<b>Gait cycle (%)</b>",
            row=len(joints),
            col=col_idx,
            title_font=dict(size=24),
        )

    fig.update_layout(
        title="<b>Knee and ankle angles per system: conditions overlaid (mean ± SD bars)</b>",
        template="plotly_white",
        height=900,
        width=1200 * len(systems),
        font=dict(size=24),
    )

    fig.add_shape(
        type="line",
        xref="paper",
        yref="paper",
        x0=0.0,
        x1=1.0,
        y0=0.5,
        y1=0.5,
        line=dict(
            color="rgba(0,0,0,0.25)",
            width=2,
        ),
        layer="above",
    )

    fig.write_html(out_path)
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

    # If you want to keep your sign flip, I’d recommend making it explicit:
    #   flip_sign_for={"ankle"}  OR  set()  OR {"knee","ankle"}
    make_knee_and_ankle_figure(
        summary_all,
        out_html,
        joints_in_rows=["knee", "ankle"],
        systems_in_cols=["mediapipe_dlc", "qualisys"],
        flip_sign_for={"ankle", "knee"},  # change/remove as you prefer
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
