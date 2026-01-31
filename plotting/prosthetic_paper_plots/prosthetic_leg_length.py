from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

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

# Prosthetic alignment offsets, originally in inches
inch_offsets = {
    "neg_5": -0.5,
    "neg_25": -0.25,
    "neutral": 0.0,
    "pos_25": 0.25,
    "pos_5": 0.5,
}

INCH_TO_MM = 25.4

# Convert expected offsets to mm
mm_offsets = {cond: val * INCH_TO_MM for cond, val in inch_offsets.items()}

tick_label_map = {
    cond: f"{mm_offsets[cond]:.2f} mm" if cond != "neutral" else "Neutral"
    for cond in condition_order
}

# -------------------------------------------------------------------
# DATA STRUCTURES
# -------------------------------------------------------------------

@dataclass
class LegResults:
    data: np.ndarray   # per-frame leg lengths [mm]
    mean: float        # median leg length [mm]
    std: float         # MAD [mm]


def leg_length_from_human(human: "Human") -> LegResults:
    """
    Compute right-leg length as distance between knee and ankle in mm.
    Returns per-frame data, median, and MAD.
    """
    leg_lengths_mm = np.linalg.norm(
        human.body.xyz.as_dict["right_knee"] - human.body.xyz.as_dict["right_ankle"],
        axis=1,
    )

    leg_length_median = np.median(leg_lengths_mm)
    leg_length_mad = np.median(np.abs(leg_lengths_mm - leg_length_median))

    return LegResults(
        data=leg_lengths_mm,
        mean=leg_length_median,
        std=leg_length_mad,
    )


from skellymodels.managers.human import Human  # keep import here to avoid F401 above

freemocap_results: dict[str, LegResults] = {}
qualisys_results: dict[str, LegResults] = {}

# -------------------------------------------------------------------
# LOAD HUMANS AND COMPUTE LEG LENGTHS
# -------------------------------------------------------------------

for name, recording_path in recordings.items():
    path_to_freemocap_folder = recording_path / "validation" / "mediapipe_dlc"
    path_to_qualisys_folder = recording_path / "validation" / "qualisys"

    f_human: Human = Human.from_data(path_to_freemocap_folder)
    q_human: Human = Human.from_data(path_to_qualisys_folder)

    freemocap_results[name] = leg_length_from_human(f_human)
    qualisys_results[name] = leg_length_from_human(q_human)

# -------------------------------------------------------------------
# DATAFRAME 1: PER-FRAME LEG LENGTH SAMPLES
# -------------------------------------------------------------------

rows_samples: list[dict] = []

for condition in condition_order:
    f_res = freemocap_results[condition]
    q_res = qualisys_results[condition]

    # Freemocap samples
    for value in f_res.data:
        rows_samples.append(
            {
                "condition": condition,
                "system": "freemocap",
                "leg_length_mm": value,
            }
        )

    # Qualisys samples
    for value in q_res.data:
        rows_samples.append(
            {
                "condition": condition,
                "system": "qualisys",
                "leg_length_mm": value,
            }
        )

df_leglength_samples = pd.DataFrame(rows_samples)

# -------------------------------------------------------------------
# DATAFRAME 2: SUMMARY (MEDIAN + MAD)
# -------------------------------------------------------------------

rows_summary: list[dict] = []

for condition in condition_order:
    f_res = freemocap_results[condition]
    q_res = qualisys_results[condition]

    rows_summary.append(
        {
            "condition": condition,
            "system": "freemocap",
            "median_leg_length_mm": f_res.mean,
            "mad_leg_length_mm": f_res.std,
            "n_frames": len(f_res.data),
        }
    )

    rows_summary.append(
        {
            "condition": condition,
            "system": "qualisys",
            "median_leg_length_mm": q_res.mean,
            "mad_leg_length_mm": q_res.std,
            "n_frames": len(q_res.data),
        }
    )

df_leglength_summary = pd.DataFrame(rows_summary)

# -------------------------------------------------------------------
# DATAFRAME 3: COMPARISON (Δ FROM NEUTRAL + EXPECTED OFFSETS)
# -------------------------------------------------------------------

# Baseline (neutral medians) in mm
fmc_neutral = freemocap_results["neutral"].mean
q_neutral = qualisys_results["neutral"].mean

rows_comp: list[dict] = []

for cond in condition_order:
    f_res = freemocap_results[cond]
    q_res = qualisys_results[cond]

    rows_comp.append(
        {
            "condition": cond,

            # Expected mechanical change [mm]
            "expected_delta_mm": mm_offsets[cond],

            # FreeMoCap values
            "fmc_median_mm": f_res.mean,
            "fmc_delta_mm": f_res.mean - fmc_neutral,
            "fmc_variability_mm": f_res.std,  # MAD
            "fmc_n_frames": len(f_res.data),

            # Qualisys values
            "q_median_mm": q_res.mean,
            "q_delta_mm": q_res.mean - q_neutral,
            "q_variability_mm": q_res.std,   # MAD
            "q_n_frames": len(q_res.data),
        }
    )

df_leg_comparison = pd.DataFrame(rows_comp)

# -------------------------------------------------------------------
# OPTIONAL: SAVE DATAFRAMES
# -------------------------------------------------------------------
# df_leglength_samples.to_csv("leg_length_samples_mm.csv", index=False)
# df_leglength_summary.to_csv("leg_length_summary_mm.csv", index=False)
# df_leg_comparison.to_csv("leg_length_comparison_mm.csv", index=False)

# -------------------------------------------------------------------
# QUICK QC PLOT: FREEMOCAP DISTRIBUTIONS (MATPLOTLIB)
# -------------------------------------------------------------------

fig, axes = plt.subplots(1, len(condition_order), figsize=(15, 3), sharey=True)
fig.suptitle("FreeMoCap Leg Length Distributions by Condition")

for ax, cond in zip(axes, condition_order):
    result = freemocap_results[cond]
    ax.hist(result.data, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(np.mean(result.data), color="red", linestyle="--", label="mean")
    ax.axvline(np.median(result.data), color="blue", linestyle="-", label="median")
    ax.set_title(cond)
    ax.set_xlabel("Leg length [mm]")

axes[0].set_ylabel("Frame count")
axes[-1].legend()
plt.tight_layout()
# plt.show()

# -------------------------------------------------------------------
# PLOTLY: Δ MEDIAN LEG LENGTH VS EXPECTED OFFSETS (mm)
# -------------------------------------------------------------------

FIG_W_IN = 3.5   # single-column target width; use ~7.0 for double column
FIG_H_IN = 2.6   # adjust to taste (paper figures are usually not tall)

DPI = 300
fig_width_px  = int(FIG_W_IN * DPI)
fig_height_px = int(FIG_H_IN * DPI)

BASE_FONT = 15
TICK_FONT = 14
LEGEND_FONT = 14

# Cleaner tick labels (units in axis title instead)
tick_text = [f"{mm_offsets[c]:.2f}" if c != "neutral" else "0" for c in condition_order]

# Ensure plot dataframe follows condition_order
df_plot = (
    df_leg_comparison
    .set_index("condition")
    .loc[condition_order]
    .reset_index()
)

# Numeric x for jitter
x_base = np.arange(len(condition_order))

# Small horizontal offsets for “jitter” between systems
offset = 0.1
x_fmc = x_base - offset
x_q   = x_base + offset
x_exp = x_base  # expected line centered

tick_text = [tick_label_map[c] for c in condition_order]

# Colors
COLOR_FMC = "#1f77b4"   # FreeMoCap blue
COLOR_Q   = "#d62728"   # Qualisys red
COLOR_EXP = "#383838"   # expected line

MARKER_SIZE = 10

fig = go.Figure()

# FreeMoCap Δ with error bars
fig.add_trace(
    go.Scatter(
        x=x_fmc,
        y=df_plot["fmc_delta_mm"],
        mode="markers",
        name="FreeMoCap Δ",
        marker=dict(
            color=COLOR_FMC,
            size=MARKER_SIZE,
            line=dict(width=0.5, color="black"),
        ),
        error_y=dict(
            type="data",
            array=df_plot["fmc_variability_mm"],  # MAD [mm]
            visible=True,
            thickness=1.2,
            width=4,
        ),
        hovertemplate=(
            "Condition: %{customdata}<br>"
            "FreeMoCap Δ: %{y:.2f} mm<br>"
            "MAD: %{error_y.array:.2f} mm<extra></extra>"
        ),
        customdata=df_plot["condition"],
    )
)

# Qualisys Δ with error bars
fig.add_trace(
    go.Scatter(
        x=x_q,
        y=df_plot["q_delta_mm"],
        mode="markers",
        name="Qualisys Δ",
        marker=dict(
            color=COLOR_Q,
            size=MARKER_SIZE,
            symbol="square",
            line=dict(width=0.5, color="black"),
        ),
        error_y=dict(
            type="data",
            array=df_plot["q_variability_mm"],  # MAD [mm]
            visible=True,
            thickness=1.2,
            width=4,
        ),
        hovertemplate=(
            "Condition: %{customdata}<br>"
            "Qualisys Δ: %{y:.2f} mm<br>"
            "MAD: %{error_y.array:.2f} mm<extra></extra>"
        ),
        customdata=df_plot["condition"],
    )
)

# Expected mechanical offsets (mm), dashed line
fig.add_trace(
    go.Scatter(
        x=x_exp,
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
    width=fig_width_px,
    height=fig_height_px,

    font=dict(
        family="Arial",
        size=BASE_FONT,
        color="black",
    ),

    # tighter margins: more plot, less whitespace
    margin=dict(l=55, r=15, t=10, b=45),

    xaxis=dict(
        title=" <b>Prosthetic adjustment (mm) </b>",
        tickmode="array",
        tickvals=x_base,
        ticktext=tick_text,
        title_font=dict(size=BASE_FONT),
        tickfont=dict(size=TICK_FONT),
        showline=True,
        linecolor="black",
        mirror=True,
        ticks="outside",
        ticklen=4,
    ),
    yaxis=dict(
        title="<b>Δ median leg length from neutral (mm)</b>",
        title_font=dict(size=BASE_FONT),
        tickfont=dict(size=TICK_FONT),
        showline=True,
        linecolor="black",
        mirror=True,
        ticks="outside",
        ticklen=4,

        # make the zero line subtle (or set to False if you prefer)
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor="rgba(0,0,0,0.35)",
    ),

    # Put legend inside the plotting area to save space
    legend=dict(
        orientation="h",
        x=0.02,
        y=0.98,
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.7)",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1,
        font=dict(size=LEGEND_FONT),
    ),
)

fig.update_traces(marker=dict(size=7))
fig.show()
