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


def load_ankle_angle_summary_for_tracker(
    conditions: dict[str, Path | str],
    tracker_dir: str,
    *,
    joint: str = "ankle",
    side: str = "right",
    component: str = "dorsi_plantar",
) -> pd.DataFrame:
    """
    Read:
        <root>/validation/<tracker_dir>/joint_angles/joint_angles_per_stride_summary_stats.csv
    and return:
        ['system', 'condition', 'percent_gait_cycle', 'mean', 'std']
    with system hard-coded to `tracker_dir`.
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

        # Filter if columns exist (Qualisys/mediapipe may or may not store these)
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
            df.pivot(
                index=["percent_gait_cycle"],
                columns="stat",
                values="value",
            )
            .reset_index()
        )

        if not {"mean", "std"}.issubset(wide.columns):
            raise ValueError(
                f"Expected 'mean' and 'std' in stat column of {csv_path}, got {wide.columns}"
            )

        # IMPORTANT: overwrite any existing tracker/system labels
        wide["system"] = tracker_dir
        wide["condition"] = cond

        wide = wide[["system", "condition", "percent_gait_cycle", "mean", "std"]]
        all_summaries.append(wide)

    out = pd.concat(all_summaries, ignore_index=True)
    # Just so you can see counts per tracker
    print(f"[{tracker_dir}] loaded rows:", len(out))
    return out


def global_yrange(summ: pd.DataFrame) -> tuple[float, float]:
    ymin = (summ["mean"] - summ["std"]).min()
    ymax = (summ["mean"] + summ["std"]).max()
    pad = 0.05 * (ymax - ymin + 1e-9)
    return float(ymin - pad), float(ymax + pad)


# -------------------------------------------------------------------
# PLOTTING
# -------------------------------------------------------------------

def make_ankle_angle_figure(
    summary: pd.DataFrame,
    out_path: Path,
    errorbar_step: int = 5,
    max_jitter: float = 1.0,
) -> Path:
    """
    `summary` must have:
        ['system', 'condition', 'percent_gait_cycle', 'mean', 'std']
    """
    # DEBUG: what systems do we actually have?
    print("Systems in summary:", summary["system"].value_counts())

    systems = ["mediapipe_dlc", "qualisys"]
    systems = [s for s in systems if s in summary["system"].unique()]

    if not systems:
        raise ValueError("No known systems (mediapipe_dlc/qualisys) found in summary['system'].")

    summary["mean"] *= -1
    fig = make_subplots(
        rows=1,
        cols=len(systems),
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=systems,
        horizontal_spacing=0.05,
    )

    ylo, yhi = global_yrange(summary)

    # Condition-wise jitter offsets
    conditions = [c for c in CONDITION_ORDER if c in set(summary["condition"].unique())]
    conditions += [c for c in sorted(summary["condition"].unique()) if c not in conditions]

    if len(conditions) > 1:
        offsets = np.linspace(-max_jitter, max_jitter, len(conditions))
    else:
        offsets = np.array([0.0])
    cond_offset = dict(zip(conditions, offsets))

    for col_idx, system in enumerate(systems, start=1):
        for cond in conditions:
            sub = (
                summary[(summary["system"] == system) & (summary["condition"] == cond)]
                .sort_values("percent_gait_cycle")
            )
            if sub.empty:
                continue

            style = CONDITION_STYLE.get(cond, {"line": "#555"})
            line_color = style["line"]

            x = sub["percent_gait_cycle"].to_numpy()
            m = sub["mean"].to_numpy()
            sd = sub["std"].to_numpy()

            # --- mean line ---
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=m,
                    mode="lines",
                    name=cond if col_idx == 1 else None,
                    legendgroup=cond,
                    line=dict(color=line_color, width=3),
                    showlegend=(col_idx == 1),
                    hovertemplate=(
                        f"<b>{system} – {cond}</b><br>"
                        "Gait cycle: %{x:.1f}%<br>"
                        "Angle: %{y:.1f}°<br>"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=col_idx,
            )

            # --- jittered SD bars ---
            if len(x) == 0:
                continue

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
                    name=None,
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
                        f"<b>{system} – {cond}</b><br>"
                        "Gait cycle: %{x:.1f}%<br>"
                        "Mean angle: %{y:.1f}°<br>"
                        "SD: %{error_y.array:.1f}°<br>"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=col_idx,
            )

        fig.update_yaxes(range=[ylo, yhi], row=1, col=col_idx)

    for col_idx in range(1, len(systems) + 1):
        fig.update_xaxes(
            title_text="<b>Gait cycle (%)</b>",
            row=1,
            col=col_idx,
            title_font=dict(size=24),
        )

    fig.update_yaxes(
        title_text="<b>Angle (°)</b>",
        row=1,
        col=1,
        title_font=dict(size=24),
    )

    fig.update_layout(
        title="<b>Ankle flexion/extension per system: conditions overlaid (mean ± SD bars)</b>",
        template="plotly_white",
        height=840,
        width=1200 * len(systems),
        font=dict(size=24),
    )

    fig.write_html(out_path)
    return out_path


# -------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------

def run_ankle_angle_summary(
    conditions: dict[str, str | Path],
    out_dir: str | Path = "ankle_summary_plots",
) -> list[Path]:
    """
    Loads per-condition summary CSVs for mediapipe_dlc and qualisys:

        <root>/validation/mediapipe_dlc/joint_angles/joint_angles_per_stride_summary_stats.csv
        <root>/validation/qualisys/joint_angles/joint_angles_per_stride_summary_stats.csv
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fmc_summ = load_ankle_angle_summary_for_tracker(
        {k: Path(v) for k, v in conditions.items()},
        tracker_dir="mediapipe_dlc",
    )

    qual_summ = load_ankle_angle_summary_for_tracker(
        {k: Path(v) for k, v in conditions.items()},
        tracker_dir="qualisys",
    )

    summary_all = pd.concat([fmc_summ, qual_summ], ignore_index=True)

    # DEBUG: show counts by system
    print("Combined summary rows per system:")
    print(summary_all["system"].value_counts())

    out_html = out_dir / "ankle_angle_system_comparison.html"
    make_ankle_angle_figure(summary_all, out_html)

    return [out_html]


if __name__ == "__main__":
    conditions = {
        "neutral": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1",
        "neg_2_8": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_03_15_TF01_flexion_neg_2_8_trial_1",
        "neg_5_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_11_55_05_TF01_flexion_neg_5_6_trial_1",
        "pos_2_8": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_09_05_TF01_flexion_pos_2_8_trial_1",
        "pos_5_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_12_36_TF01_flexion_pos_5_6_trial_1",
    }

    outputs = run_ankle_angle_summary(conditions, out_dir="ankle_summary_plots")
    for p in outputs:
        print(p)
