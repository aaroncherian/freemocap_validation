from pathlib import Path
import pandas as pd

conditions = {
    "neg_5_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_11_55_05_TF01_flexion_neg_5_6_trial_1",
    "neg_2_8": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_03_15_TF01_flexion_neg_2_8_trial_1",
    "neutral": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1",
    "pos_2_8": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_09_05_TF01_flexion_pos_2_8_trial_1",
    "pos_5_6": r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_12_36_TF01_flexion_pos_5_6_trial_1",
}
tracker = "mediapipe_dlc"

summary_df = pd.DataFrame()
for condition, path in conditions.items():
    path = Path(path)
    path_to_gait_events = path/'validation'/tracker/f'{tracker}_joint_trajectory_by_stride.csv'

    df = pd.read_csv(path_to_gait_events)
    df = df.query("marker == 'toe'")
    df['condition'] = condition

    summary_df = pd.concat([summary_df, df], ignore_index = True)

toe = summary_df.loc[:, ["system","condition","stride","percent_gait_cycle","z"]].copy()

# keep only the analysis window (you chose >= 80%)
toe80 = toe.query("80 <= percent_gait_cycle <= 95").copy()

# find the index (within toe80) of the minimum z per (system, condition, stride)
idx_min = (
    toe80
    .groupby(["system","condition","stride"])["z"]
    .idxmin()
)

# extract those rows (one per stride), keep value and %GC
per_stride_min = (
    toe80.loc[idx_min, ["system","condition","stride","percent_gait_cycle","z"]]
         .rename(columns={"percent_gait_cycle":"pct_at_min",
                          "z":"toe_clear_min"})
         .reset_index(drop=True)
)

per_stride_min["toe_clear_min"].describe()

# ensure numeric (in case CSV dtypes are weird)
per_stride_min["pct_at_min"] = pd.to_numeric(per_stride_min["pct_at_min"], errors="coerce")
per_stride_min["toe_clear_min"] = pd.to_numeric(per_stride_min["toe_clear_min"], errors="coerce")

# --- summarize across strides: mean ± SD and n ---
min_summary = (
    per_stride_min
    .groupby(["system","condition"], as_index=False)
    .agg(
        mean_min_mm      = ("toe_clear_min", "mean"),
        sd_min_mm        = ("toe_clear_min", "std"),
        mean_pct_at_min  = ("pct_at_min", "mean"),
        sd_pct_at_min    = ("pct_at_min", "std"),
        n_strides        = ("toe_clear_min", "size"),
    )
    .sort_values(["system","condition"])
)

print("Per-stride minima (one row per stride):")
print(per_stride_min.head())

print("\nSummary (mean ± SD across strides):")
print(min_summary)


import plotly.graph_objects as go

COND_ORDER = ["neg_5_6","neg_2_8","neutral","pos_2_8","pos_5_6"]
COLOR_MAP = {
    "mediapipe_dlc": "rgb(31,119,180)",  # FMC blue
    "qualisys": "rgb(214,39,40)",        # QTM red
}

fig = go.Figure()
shown = set()

for cond in COND_ORDER:
    for sys in ["mediapipe_dlc", "qualisys"]:
        df_sub = per_stride_min.query("condition == @cond and system == @sys")
        if df_sub.empty:
            continue
        side = "positive" if sys == "qualisys" else "negative"

        fig.add_trace(go.Violin(
            y=df_sub["toe_clear_min"],
            x=[cond]*len(df_sub),
            name=sys,
            legendgroup=sys,
            showlegend=(sys not in shown),   # one legend entry per system
            side=side,
            line_color=COLOR_MAP[sys],
            fillcolor=COLOR_MAP[sys],
            opacity=0.6,
            box=dict(visible=True, width=0.25),      # inner boxplot
            meanline=dict(visible=True, color="black", width=1),  # mean line
            points="all",
            jitter=0.3,
            scalemode="width"
        ))
        shown.add(sys)

fig.update_layout(
    title="Split Violin: Minimum Toe Clearance by Condition",
    violinmode="overlay",  # required for split violins
    xaxis_title="Condition",
    yaxis_title="Minimum Toe Clearance (mm)",
    legend=dict(orientation="h", y=1.05),
    margin=dict(t=80, r=20, l=70, b=60)
)

# fig.show()


import numpy as np
import pandas as pd
import plotly.graph_objects as go

COND_ORDER = ["neg_5_6","neg_2_8","neutral","pos_2_8","pos_5_6"]
SYSTEM_ORDER = ["qualisys", "mediapipe_dlc"]  # controls legend/plot order
DISPLAY_NAME = {"qualisys": "Qualisys", "mediapipe_dlc": "FreeMoCap (MediaPipe+DLC)"}
COLOR_MAP = {"mediapipe_dlc": "rgb(31,119,180)", "qualisys": "rgb(214,39,40)"}

# --- build a summary table from per_stride_min (recommended; always consistent with your extraction) ---
summary = (
    per_stride_min
    .groupby(["system","condition"], as_index=False)
    .agg(
        mean_mm=("toe_clear_min", "mean"),
        sd_mm=("toe_clear_min", "std"),
        n=("toe_clear_min", "size"),
    )
)

# optional: use SEM instead of SD for error bars
summary["sem_mm"] = summary["sd_mm"] / np.sqrt(summary["n"])

# choose what you want error bars to represent
ERR_COL = "sd_mm"   # "sd_mm" or "sem_mm"

# ensure condition ordering
summary["condition"] = pd.Categorical(summary["condition"], categories=COND_ORDER, ordered=True)
summary = summary.sort_values(["condition", "system"])

# --- plot: two points per condition + connecting line ---
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Figure dimensions (match leg length plot) ---
FIG_W_IN = 1.8   # single-column target width; use ~7.0 for double column
FIG_H_IN = 1.3
DPI = 300
fig_width_px = int(FIG_W_IN * DPI)
fig_height_px = int(FIG_H_IN * DPI)

# --- Font sizes ---
BASE_FONT = 15
TICK_FONT = 14
LEGEND_FONT = 14

# --- Colors (consistent with leg length plot) ---
COLOR_FMC = "#1f77b4"
COLOR_Q = "#d62728"

MARKER_SIZE = 7

# --- Condition setup ---
COND_ORDER = ["neg_5_6", "neg_2_8", "neutral", "pos_2_8", "pos_5_6"]

# Map conditions to display labels (adjust units as needed)
tick_label_map = {
    "neg_5_6": "−5.6°",
    "neg_2_8": "−2.8°",
    "neutral": "Neutral",
    "pos_2_8": "+2.8°",
    "pos_5_6": "+5.6°",
}
tick_text = [tick_label_map[c] for c in COND_ORDER]

# --- Build summary from per_stride_min (assumes this df exists) ---
summary = (
    per_stride_min
    .groupby(["system", "condition"], as_index=False)
    .agg(
        mean_mm=("toe_clear_min", "mean"),
        sd_mm=("toe_clear_min", "std"),
        n=("toe_clear_min", "size"),
    )
)

summary["condition"] = pd.Categorical(
    summary["condition"], categories=COND_ORDER, ordered=True
)
summary = summary.sort_values(["condition", "system"])

# --- Numeric x with jitter ---
x_base = np.arange(len(COND_ORDER))
offset = 0.1
x_fmc = x_base - offset
x_q = x_base + offset

# --- Build plot ---
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Figure dimensions (match leg length plot) ---
FIG_W_IN = 1.8   # single-column target width; use ~7.0 for double column
FIG_H_IN = 1.3
DPI = 300
fig_width_px = int(FIG_W_IN * DPI)
fig_height_px = int(FIG_H_IN * DPI)

# --- Font sizes ---
BASE_FONT = 15
TICK_FONT = 14
LEGEND_FONT = 14

# --- Colors (consistent with leg length plot) ---
COLOR_FMC = "#1f77b4"
COLOR_Q = "#d62728"

MARKER_SIZE = 7

# --- Condition setup ---
COND_ORDER = ["neg_5_6", "neg_2_8", "neutral", "pos_2_8", "pos_5_6"]

# Map conditions to display labels (adjust units as needed)
tick_label_map = {
    "neg_5_6": "−5.6°",
    "neg_2_8": "−2.8°",
    "neutral": "Neutral",
    "pos_2_8": "+2.8°",
    "pos_5_6": "+5.6°",
}
tick_text = [tick_label_map[c] for c in COND_ORDER]

# --- Build summary from per_stride_min (assumes this df exists) ---
summary = (
    per_stride_min
    .groupby(["system", "condition"], as_index=False)
    .agg(
        mean_mm=("toe_clear_min", "mean"),
        sd_mm=("toe_clear_min", "std"),
        n=("toe_clear_min", "size"),
    )
)

summary["condition"] = pd.Categorical(
    summary["condition"], categories=COND_ORDER, ordered=True
)
summary = summary.sort_values(["condition", "system"])

# --- Numeric x with jitter ---
x_base = np.arange(len(COND_ORDER))
offset = 0.1
x_fmc = x_base - offset
x_q = x_base + offset

# --- Build plot ---
fig = go.Figure()

# FreeMoCap trace
df_fmc = summary.query("system == 'mediapipe_dlc'").reset_index(drop=True)
fig.add_trace(
    go.Scatter(
        x=x_fmc,
        y=df_fmc["mean_mm"],
        mode="markers+lines",
        name="FreeMoCap",
        marker=dict(
            color=COLOR_FMC,
            size=MARKER_SIZE,
            symbol="circle",
            line=dict(width=0.5, color="black"),
        ),
        line=dict(width=1.5, color=COLOR_FMC),
        error_y=dict(
            type="data",
            array=df_fmc["sd_mm"],
            visible=True,
            thickness=1.2,
            width=4,
        ),
        hovertemplate=(
            "Condition: %{customdata}<br>"
            "Mean: %{y:.1f} mm<br>"
            "SD: %{error_y.array:.1f} mm<extra></extra>"
        ),
        customdata=df_fmc["condition"],
    )
)

# Qualisys trace
df_q = summary.query("system == 'qualisys'").reset_index(drop=True)
fig.add_trace(
    go.Scatter(
        x=x_q,
        y=df_q["mean_mm"],
        mode="markers+lines",
        name="Qualisys",
        marker=dict(
            color=COLOR_Q,
            size=MARKER_SIZE,
            symbol="square",
            line=dict(width=0.5, color="black"),
        ),
        line=dict(width=1.5, color=COLOR_Q),
        error_y=dict(
            type="data",
            array=df_q["sd_mm"],
            visible=True,
            thickness=1.2,
            width=4,
        ),
        hovertemplate=(
            "Condition: %{customdata}<br>"
            "Mean: %{y:.1f} mm<br>"
            "SD: %{error_y.array:.1f} mm<extra></extra>"
        ),
        customdata=df_q["condition"],
    )
)

# --- Layout (matching leg length plot) ---
fig.update_layout(
    template="simple_white",
    width=fig_width_px,
    height=fig_height_px,
    font=dict(family="Arial", size=BASE_FONT, color="black"),
    margin=dict(l=55, r=15, t=10, b=45),
    xaxis=dict(
        title="<b>Prosthetic flexion adjustment (°)</b>",
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
        title="<b>Minimum toe clearance (mm)</b>",
        title_font=dict(size=BASE_FONT),
        tickfont=dict(size=TICK_FONT),
        showline=True,
        linecolor="black",
        mirror=True,
        ticks="outside",
        ticklen=4,
        zeroline=False,
    ),
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

# fig.show()

import plotly.io as pio
pio.kaleido.scope.mathjax = None
path_to_save = Path(r"C:\Users\aaron\Documents\prosthetics_paper")
fig.write_image(path_to_save / "toe_clearance.pdf")