from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# -------------------------------------------------------------------
# CONFIG (match your condition names)
# -------------------------------------------------------------------

CONDITION_ORDER = ["neg_5", "neg_25", "neutral", "pos_25", "pos_5"]

# Reuse your palette idea (reds -> neutral -> blues)
CONDITION_STYLE: dict[str, dict[str, str]] = {
    "neg_5":   {"line": "#94342b"},
    "neg_25":  {"line": "#d39182"},
    "neutral": {"line": "#524F4F"},
    "pos_25":  {"line": "#7bb6c6"},
    "pos_5":   {"line": "#447c8e"},
}


# -------------------------------------------------------------------
# HELPERS (same as your ankle script)
# -------------------------------------------------------------------

def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    s = hex_color.lstrip("#")
    if not re.fullmatch(r"[0-9A-Fa-f]{6}", s):
        raise ValueError(f"Invalid hex color: {hex_color}")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def global_yrange(summ: pd.DataFrame) -> tuple[float, float]:
    ymin = (summ["mean"] - summ["std"]).min()
    ymax = (summ["mean"] + summ["std"]).max()
    pad = 0.05 * (ymax - ymin + 1e-9)
    return float(ymin - pad), float(ymax + pad)


def load_angle_summary_for_tracker(
    conditions: dict[str, Path | str],
    tracker_dir: str,
    *,
    joint: str,
    component: str,
    side_preference: tuple[str, ...] = ("mid", "right", "left"),
) -> pd.DataFrame:
    """
    Reads:
        <root>/validation/<tracker_dir>/joint_angles/joint_angles_per_stride_summary_stats.csv

    Returns wide:
        ['system', 'condition', 'percent_gait_cycle', 'mean', 'std']
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

        # Filter by joint/component if columns exist
        if "joint" in df.columns:
            df = df[df["joint"] == joint]
        if "component" in df.columns:
            df = df[df["component"] == component]

        # Side handling (pelvis may be stored as mid OR right depending on your choice)
        if "side" in df.columns and not df.empty:
            sides_present = [s for s in df["side"].dropna().unique().tolist()]
            chosen_side = None
            for s in side_preference:
                if s in sides_present:
                    chosen_side = s
                    break
            if chosen_side is not None:
                df = df[df["side"] == chosen_side]

        if df.empty:
            raise ValueError(
                f"No rows in {csv_path} for joint={joint}, component={component}. "
                f"If pelvis side is stored differently, adjust side_preference."
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
        wide = wide[["system", "condition", "percent_gait_cycle", "mean", "std"]]
        all_summaries.append(wide)

    out = pd.concat(all_summaries, ignore_index=True)
    print(f"[{tracker_dir}] loaded rows:", len(out))
    return out


# -------------------------------------------------------------------
# PLOTTING (same structure & style as ankle_ankles_fig.py)
# -------------------------------------------------------------------

def make_pelvis_obliquity_figure(
    summary: pd.DataFrame,
    out_path: Path,
    *,
    systems: list[str],
    errorbar_step: int = 5,
    max_jitter: float = 1.0,
    title: str = "<b>Pelvic obliquity per system: conditions overlaid (mean ± SD bars)</b>",
) -> Path:
    """
    `summary` must have:
        ['system', 'condition', 'percent_gait_cycle', 'mean', 'std']
    """
    systems = [s for s in systems if s in summary["system"].unique()]
    if not systems:
        raise ValueError("No requested systems found in summary['system'].")

    fig = make_subplots(
        rows=1,
        cols=len(systems),
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=systems,
        horizontal_spacing=0.05,
    )

    ylo, yhi = global_yrange(summary)

    # Condition ordering + jitter offsets
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
                        "Obliquity: %{y:.2f}°<br>"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=col_idx,
            )

            # --- jittered SD bars (markers) ---
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
                        "Mean: %{y:.2f}°<br>"
                        "SD: %{error_y.array:.2f}°<br>"
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
        title_text="<b>Obliquity (°)</b>",
        row=1,
        col=1,
        title_font=dict(size=24),
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=840,
        width=1200 * len(systems),
        font=dict(size=24),
    )
    fig.show()
    fig.write_html(out_path)
    return out_path


# -------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------

def run_pelvis_obliquity_summary(
    conditions: dict[str, str | Path],
    trackers: list[str],
    out_dir: str | Path = "pelvis_summary_plots",
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for tracker in trackers:
        summ = load_angle_summary_for_tracker(
            {k: Path(v) for k, v in conditions.items()},
            tracker_dir=tracker,
            joint="pelvis",
            component="obliquity",
            side_preference=("mid", "right", "left"),
        )
        all_summaries.append(summ)

    summary_all = pd.concat(all_summaries, ignore_index=True)

    out_html = out_dir / "pelvis_obliquity_system_comparison.html"
    make_pelvis_obliquity_figure(
        summary_all,
        out_html,
        systems=trackers,          # column order
        errorbar_step=5,
        max_jitter=1.0,
        title="<b>Pelvic obliquity per system: leg-length conditions overlaid (mean ± SD bars)</b>",
    )
    return out_html


if __name__ == "__main__":
    recordings = {
        "neg_5": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_38_16_TF01_leg_length_neg_5_trial_1",
        "neg_25": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_43_15_TF01_leg_length_neg_25_trial_1",
        "neutral": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_46_54_TF01_leg_length_neutral_trial_1",
        "pos_25": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_50_56_TF01_leg_length_pos_25_trial_1",
        "pos_5": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_55_21_TF01_leg_length_pos_5_trial_1",
    }

    trackers = ["qualisys", "rtmpose", "mediapipe_dlc"]

    out = run_pelvis_obliquity_summary(recordings, trackers, out_dir="pelvis_summary_plots")
    print(out)
