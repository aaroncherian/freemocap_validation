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

# Fixed, high-contrast palette for the conditions
CONDITION_STYLE: dict[str, dict[str, str]] = {
    "neg_5_6": {"line": "#94342b"},   # red-brown
    "neg_2_8": {"line": "#d39182"},   # light clay
    "neutral": {"line": "#524F4F"},   # medium grey
    "pos_2_8": {"line": "#7bb6c6"},   # soft teal
    "pos_5_6": {"line": "#447c8e"},   # deep teal
}


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """
    Convert a hex color string like '#94342b' into an (r, g, b) tuple.
    Assumes a valid 6-char hex string.
    """
    s = hex_color.lstrip("#")
    if not re.fullmatch(r"[0-9A-Fa-f]{6}", s):
        raise ValueError(f"Invalid hex color: {hex_color}")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def load_ankle_angle_data(
    conditions: dict[str, Path | str],
    tracker: str = "mediapipe_dlc",
    angle_name: str = "ankle_dorsi_plantar_r",
) -> pd.DataFrame:
    """
    Load ankle-angle-by-stride CSVs for each condition and concatenate them.

    Expects files:
        <root>/validation/<tracker>/<tracker>_joint_angle_by_stride.csv

    Returns a DataFrame with at least:
        ['system', 'stride', 'percent_gait_cycle', 'ankle_angle', 'condition']
    """
    dfs: list[pd.DataFrame] = []

    for cond, root in conditions.items():
        root = Path(root)
        csv_path = root / "validation" / tracker / f"{tracker}_joint_angle_by_stride.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV for condition '{cond}': {csv_path}")

        df = pd.read_csv(csv_path)

        # keep only the ankle angle we care about
        df = df[df["angle"] == angle_name].copy()
        df = df.rename(columns={"value": "ankle_angle"})
        df = df.drop(columns=["angle"])
        df["condition"] = cond

        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)

    required_cols = {"system", "stride", "percent_gait_cycle", "ankle_angle", "condition"}
    missing = required_cols - set(all_df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    return all_df


def summarize_ankle_angle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize ankle angle as mean ± SD across strides for each
    (system, condition, percent_gait_cycle).
    """
    summ = (
        df.groupby(["system", "condition", "percent_gait_cycle"])["ankle_angle"]
          .agg(["mean", "std"])
          .reset_index()
    )
    summ["std"] = summ["std"].fillna(0.0)
    return summ


def global_yrange(summ: pd.DataFrame) -> tuple[float, float]:
    """
    Return a padded global y-range based on mean ± SD.
    """
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
    Create an ankle flexion/extension figure with mean curves and
    jittered SD error bars for each condition and system.

    `summary` must have columns:
        ['system', 'condition', 'percent_gait_cycle', 'mean', 'std']
    """
    systems = sorted(summary["system"].unique())

    # Respect canonical condition order, then append any extras
    conditions = [c for c in CONDITION_ORDER if c in set(summary["condition"].unique())]
    conditions += [c for c in sorted(summary["condition"].unique()) if c not in conditions]

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

            # -------- mean line --------
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

            # -------- jittered SD bars (subsampled) --------
            if len(x) == 0:
                continue

            idx = np.arange(0, len(x), errorbar_step)
            x_base = x[idx]
            m_err = m[idx]
            sd_err = sd[idx]

            x_err = x_base + cond_offset[cond]
            # keep first/last bars aligned with the curve
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

    # Axes & layout
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


def save_highres_png(fig, path: Path, width_in: float = 8.0, height_in: float = 4.5, dpi: int = 300):
    """
    Save a Plotly figure as a high-res PNG using kaleido.
    """
    pio.write_image(
        fig,
        str(path),
        format="png",
        width=int(width_in * dpi),
        height=int(height_in * dpi),
        scale=1,
        engine="kaleido",
    )


# -------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------

def run_ankle_angle_summary(
    conditions: dict[str, str | Path],
    tracker: str = "mediapipe_dlc",
    out_dir: str | Path = "ankle_summary_plots",
) -> list[Path]:
    """
    High-level helper:
    - loads per-condition CSVs
    - summarizes ankle angle
    - builds the HTML figure

    Returns a list of output paths (currently only the HTML figure).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_ankle_angle_data({k: Path(v) for k, v in conditions.items()}, tracker=tracker)
    summ = summarize_ankle_angle(df)

    out_html = out_dir / "ankle_angle_system_comparison.html"
    make_ankle_angle_figure(summ, out_html)

    return [out_html]


if __name__ == "__main__":
    # Example usage (edit these paths for your machine):
    conditions = {
        "neutral": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1",
        "neg_2_8": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_03_15_TF01_flexion_neg_2_8_trial_1",
        "neg_5_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_11_55_05_TF01_flexion_neg_5_6_trial_1",
        "pos_2_8": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_09_05_TF01_flexion_pos_2_8_trial_1",
        "pos_5_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_12_36_TF01_flexion_pos_5_6_trial_1",
    }

    outputs = run_ankle_angle_summary(conditions, tracker="mediapipe_dlc", out_dir="ankle_summary_plots")
    for p in outputs:
        print(p)
