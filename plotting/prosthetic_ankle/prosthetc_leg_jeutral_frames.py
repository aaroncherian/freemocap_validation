from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import yaml  # pip install pyyaml

from skellymodels.managers.human import Human


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
condition_order = ["neg_5", "neg_25", "neutral", "pos_25", "pos_5"]

# YAML folder + mapping
yaml_folder = Path(r"C:\Users\aaron\Documents\GitHub\freemocap_validation\config_yamls\prosthetic_data")
yaml_files = {
    "neg_5": yaml_folder / "leg_length_neg5.yaml",
    "neg_25": yaml_folder / "leg_length_neg25.yaml",
    "neutral": yaml_folder / "leg_length_neutral.yaml",
    "pos_25": yaml_folder / "leg_length_pos25.yaml",
    "pos_5": yaml_folder / "leg_length_pos5.yaml",
}

# Expected offsets
inch_offsets = {"neg_5": -0.5, "neg_25": -0.25, "neutral": 0.0, "pos_25": 0.25, "pos_5": 0.5}
INCH_TO_MM = 25.4
mm_offsets = {k: v * INCH_TO_MM for k, v in inch_offsets.items()}

tick_label_map = {
    cond: f"{mm_offsets[cond]:.2f} mm" if cond != "neutral" else "neutral"
    for cond in condition_order
}

# Plot styling
COLOR_FMC = "#1f77b4"
COLOR_Q   = "#d62728"
COLOR_EXP = "#383838"
MARKER_SIZE = 10


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def median_and_mad(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan, np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(med), float(mad)


def leg_length_series_mm(human: Human) -> np.ndarray:
    """
    Right knee->ankle Euclidean distance per frame [mm].
    """
    return np.linalg.norm(
        human.body.xyz.as_dict["right_knee"] - human.body.xyz.as_dict["right_ankle"],
        axis=1,
    )


def load_neutral_frames_from_yaml(yaml_path: Path) -> tuple[int, int]:
    """
    Reads:
    JointAnglesStep:
      neutral_frames: [start, end]
    Returns (start, end) as ints.
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    try:
        frames = cfg["JointAnglesStep"]["neutral_frames"]
        start, end = int(frames[0]), int(frames[1])
    except Exception as e:
        raise ValueError(
            f"Could not parse JointAnglesStep.neutral_frames from {yaml_path}. "
            f"Expected like: JointAnglesStep: {{neutral_frames: [90, 190]}}"
        ) from e

    if end < start:
        raise ValueError(f"neutral_frames end < start in {yaml_path}: {start}, {end}")

    return start, end


def slice_inclusive(x: np.ndarray, start: int, end: int) -> np.ndarray:
    """
    Inclusive slice: [start, end] inclusive.
    Clamps to valid range.
    """
    n = len(x)
    s = max(0, start)
    e = min(n - 1, end)
    if e < s:
        return np.array([], dtype=float)
    return x[s : e + 1]


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

def main():
    rows_samples: list[dict] = []
    rows_summary: list[dict] = []
    rows_comp: list[dict] = []

    # --- Load all conditions ---
    per_cond = {}  # condition -> dict(system -> dict(data, med, mad, n, frames_used))

    for cond in condition_order:
        rec = recordings[cond]
        yaml_path = yaml_files[cond]
        start_f, end_f = load_neutral_frames_from_yaml(yaml_path)

        f_folder = rec / "validation" / "mediapipe_dlc"
        q_folder = rec / "validation" / "qualisys"

        f_h = Human.from_data(f_folder)
        q_h = Human.from_data(q_folder)

        f_all = leg_length_series_mm(f_h)
        q_all = leg_length_series_mm(q_h)

        # Make sure we don't index beyond either array
        max_len = min(len(f_all), len(q_all))
        f_all = f_all[:max_len]
        q_all = q_all[:max_len]

        f_win = slice_inclusive(f_all, start_f, end_f)
        q_win = slice_inclusive(q_all, start_f, end_f)

        f_med, f_mad = median_and_mad(f_win)
        q_med, q_mad = median_and_mad(q_win)

        per_cond[cond] = {
            "neutral_frames": (start_f, end_f),
            "freemocap": {"data": f_win, "median": f_med, "mad": f_mad, "n": len(f_win)},
            "qualisys": {"data": q_win, "median": q_med, "mad": q_mad, "n": len(q_win)},
        }

        # samples DF
        for system in ["freemocap", "qualisys"]:
            data = per_cond[cond][system]["data"]
            for i, val in enumerate(data):
                rows_samples.append(
                    dict(
                        condition=cond,
                        system=system,
                        neutral_frame_start=start_f,
                        neutral_frame_end=end_f,
                        sample_index=i,  # index within the window
                        leg_length_mm=float(val),
                    )
                )

        # summary DF
        rows_summary.append(
            dict(
                condition=cond,
                system="freemocap",
                neutral_frame_start=start_f,
                neutral_frame_end=end_f,
                median_leg_length_mm=f_med,
                mad_leg_length_mm=f_mad,
                n_frames=len(f_win),
            )
        )
        rows_summary.append(
            dict(
                condition=cond,
                system="qualisys",
                neutral_frame_start=start_f,
                neutral_frame_end=end_f,
                median_leg_length_mm=q_med,
                mad_leg_length_mm=q_mad,
                n_frames=len(q_win),
            )
        )

    df_neutral_samples = pd.DataFrame(rows_samples)
    df_neutral_summary = pd.DataFrame(rows_summary)

    # --- comparison DF: Δ from neutral condition (per system) ---
    f_base = per_cond["neutral"]["freemocap"]["median"]
    q_base = per_cond["neutral"]["qualisys"]["median"]

    for cond in condition_order:
        f = per_cond[cond]["freemocap"]
        q = per_cond[cond]["qualisys"]

        rows_comp.append(
            dict(
                condition=cond,
                expected_delta_mm=mm_offsets[cond],

                fmc_median_mm=f["median"],
                fmc_delta_mm=f["median"] - f_base,
                fmc_variability_mm=f["mad"],
                fmc_n_frames=f["n"],

                q_median_mm=q["median"],
                q_delta_mm=q["median"] - q_base,
                q_variability_mm=q["mad"],
                q_n_frames=q["n"],
            )
        )

    df_neutral_comparison = pd.DataFrame(rows_comp)

    # --- plot ---
    df_plot = df_neutral_comparison.set_index("condition").loc[condition_order].reset_index()
    x = np.arange(len(condition_order))
    offset = 0.12
    x_fmc = x - offset
    x_q = x + offset
    tick_text = [tick_label_map[c] for c in condition_order]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_fmc,
            y=df_plot["fmc_delta_mm"],
            mode="markers",
            name="FreeMoCap Δ",
            marker=dict(color=COLOR_FMC, size=MARKER_SIZE, line=dict(width=0.5, color="black")),
            error_y=dict(type="data", array=df_plot["fmc_variability_mm"], visible=True, thickness=1.2, width=4),
            customdata=df_plot["condition"],
            hovertemplate="Condition: %{customdata}<br>FreeMoCap Δ: %{y:.2f} mm<br>MAD: %{error_y.array:.2f} mm<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_q,
            y=df_plot["q_delta_mm"],
            mode="markers",
            name="Qualisys Δ",
            marker=dict(color=COLOR_Q, size=MARKER_SIZE, symbol="square", line=dict(width=0.5, color="black")),
            error_y=dict(type="data", array=df_plot["q_variability_mm"], visible=True, thickness=1.2, width=4),
            customdata=df_plot["condition"],
            hovertemplate="Condition: %{customdata}<br>Qualisys Δ: %{y:.2f} mm<br>MAD: %{error_y.array:.2f} mm<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=df_plot["expected_delta_mm"],
            mode="lines+markers",
            name="Expected Δ (mm)",
            line=dict(dash="dash", width=2, color=COLOR_EXP),
            marker=dict(size=MARKER_SIZE, line=dict(width=1.5, color=COLOR_EXP)),
            hovertemplate="Expected Δ: %{y:.2f} mm<extra></extra>",
        )
    )

    fig.update_layout(
        template="simple_white",
        width=900,
        height=650,
        font=dict(size=16),
        title=dict(
            text="Neutral A-pose window: change in median leg length relative to neutral condition",
            x=0.5,
            xanchor="center",
            font=dict(size=20),
        ),
        xaxis=dict(
            title="Prosthetic alignment condition",
            tickmode="array",
            tickvals=x,
            ticktext=tick_text,
        ),
        yaxis=dict(
            title="Δ Median leg length from neutral [mm]",
            zeroline=True,
            zerolinecolor="gray",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
        margin=dict(l=80, r=40, t=80, b=110),
    )

    fig.show()

    print("\nNeutral window summary head:")
    print(df_neutral_summary.head())

    print("\nNeutral window comparison head:")
    print(df_neutral_comparison.head())

    return df_neutral_samples, df_neutral_summary, df_neutral_comparison


if __name__ == "__main__":
    df_neutral_samples, df_neutral_summary, df_neutral_comparison = main()
