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
fig = go.Figure()

# 1) add the points (one trace per system) so legend is clean
for sys in SYSTEM_ORDER:
    df_sys = summary.query("system == @sys")
    fig.add_trace(go.Scatter(
        x=df_sys["condition"],
        y=df_sys["mean_mm"],
        mode="markers+lines",   # lines connect conditions for each system (optional)
        name=DISPLAY_NAME.get(sys, sys),
        marker=dict(size=10),
        line=dict(width=2),
        error_y=dict(type="data", array=df_sys[ERR_COL], visible=True),
    ))

# 2) add thin connector lines between Qualisys and FreeMoCap within each condition
# (these do NOT show in legend)
for cond in COND_ORDER:
    df_c = summary.query("condition == @cond").set_index("system")
    if not all(s in df_c.index for s in ["qualisys", "mediapipe_dlc"]):
        continue
    y_q = float(df_c.loc["qualisys", "mean_mm"])
    y_f = float(df_c.loc["mediapipe_dlc", "mean_mm"])

    fig.add_trace(go.Scatter(
        x=[cond, cond],
        y=[y_q, y_f],
        mode="lines",
        line=dict(width=1, dash="dot", color="rgba(0,0,0,0.5)"),
        showlegend=False,
        hoverinfo="skip",
    ))

fig.update_traces(marker_color=None)  # leave colors to default unless you set them below

# apply your preferred colors explicitly (optional)
for tr in fig.data:
    # only system traces have names matching DISPLAY_NAME values
    if tr.name == DISPLAY_NAME["qualisys"]:
        tr.marker.color = COLOR_MAP["qualisys"]
        tr.line.color = COLOR_MAP["qualisys"]
    elif tr.name == DISPLAY_NAME["mediapipe_dlc"]:
        tr.marker.color = COLOR_MAP["mediapipe_dlc"]
        tr.line.color = COLOR_MAP["mediapipe_dlc"]

fig.update_layout(
    title=f"Toe Clearance (mean ± {'SD' if ERR_COL=='sd_mm' else 'SEM'})",
    xaxis_title="Condition",
    yaxis_title="Toe Clearance (mm)",
    legend=dict(orientation="h", y=1.08),
    margin=dict(t=90, r=20, l=70, b=60),
)

fig.show()

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- 1) Pair FMC and Qualisys per (condition, stride) ---
pair = (
    per_stride_min
    .pivot_table(
        index=["condition", "stride"],
        columns="system",
        values="toe_clear_min",
        aggfunc="mean",
    )
    .reset_index()
)

# keep only rows where both systems exist
pair = pair.dropna(subset=["qualisys", "mediapipe_dlc"]).copy()

pair["mean_mm"] = (pair["qualisys"] + pair["mediapipe_dlc"]) / 2.0
pair["diff_mm"] = (pair["mediapipe_dlc"] - pair["qualisys"])  # FMC - QTM

# --- 2) Compute BA stats ---
bias = pair["diff_mm"].mean()
sd = pair["diff_mm"].std(ddof=1)
loa_low = bias - 1.96 * sd
loa_high = bias + 1.96 * sd

print(f"n paired strides: {len(pair)}")
print(f"Bias (FMC - QTM): {bias:.3f} mm")
print(f"SD of diff:       {sd:.3f} mm")
print(f"LoA:              [{loa_low:.3f}, {loa_high:.3f}] mm")

# --- 3) Plot ---
COND_ORDER = ["neg_5_6","neg_2_8","neutral","pos_2_8","pos_5_6"]
pair["condition"] = pd.Categorical(pair["condition"], categories=COND_ORDER, ordered=True)

fig = px.scatter(
    pair,
    x="mean_mm",
    y="diff_mm",
    color="condition",
    category_orders={"condition": COND_ORDER},
    opacity=0.65,
    title="Bland–Altman: Late-swing minimum toe height (FreeMoCap − Qualisys)",
    labels={
        "mean_mm": "Mean toe height of methods (mm)",
        "diff_mm": "Difference (FreeMoCap − Qualisys) (mm)",
        "condition": "Condition",
    },
)

# BA reference lines
fig.add_hline(y=bias, line_width=2, line_dash="solid")
fig.add_hline(y=loa_low, line_width=2, line_dash="dash")
fig.add_hline(y=loa_high, line_width=2, line_dash="dash")

# annotate bias/LoA
fig.add_annotation(
    x=pair["mean_mm"].max(),
    y=bias,
    xanchor="right",
    yanchor="bottom",
    text=f"bias = {bias:.1f} mm",
    showarrow=False,
)
fig.add_annotation(
    x=pair["mean_mm"].max(),
    y=loa_high,
    xanchor="right",
    yanchor="bottom",
    text=f"+1.96 SD = {loa_high:.1f} mm",
    showarrow=False,
)
fig.add_annotation(
    x=pair["mean_mm"].max(),
    y=loa_low,
    xanchor="right",
    yanchor="top",
    text=f"−1.96 SD = {loa_low:.1f} mm",
    showarrow=False,
)

fig.update_layout(
    margin=dict(t=80, r=20, l=70, b=60),
    legend=dict(orientation="h", y=1.05),
)

fig.show()
