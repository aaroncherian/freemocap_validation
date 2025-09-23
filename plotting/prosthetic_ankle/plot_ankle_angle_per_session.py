from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

path_to_recording = Path(r'D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1')
tracker = 'mediapipe_dlc'

COLOR_MAP = {
    "mediapipe_dlc": "rgb(31,119,180)",  # FMC blue
    "qualisys":      "rgb(214,39,40)",   # QTM red
}


path_to_angle_strides = path_to_recording/'validation'/tracker/f'{tracker}_joint_angle_by_stride.csv'
df = pd.read_csv(path_to_angle_strides)
df = df[df["angle"] == "ankle_dorsi_plantar_r"]      # keep only that angle
df = df.rename(columns={"value": "ankle_angle"})     # match old naming
df = df.drop(columns=["angle"])                      # optional: drop the angle column
# --- All cycles overlayed ---
fig = px.line(
    df,
    x="percent_gait_cycle",
    y="ankle_angle",
    color="system",         # freemocap vs qualisys
    line_group="stride",    # plot each stride separately
    hover_data=["stride"],  # show stride number on hover
)
fig.show()

palette = px.colors.qualitative.Set2
systems = df["system"].unique()

df_summary = (
    df.groupby(["system", "percent_gait_cycle"])
      .agg(mean_angle=("ankle_angle", "mean"),
           std_angle=("ankle_angle", "std"))
      .reset_index()
)

# --- Mean and std band ---
fig = go.Figure()

for i, system in enumerate(systems):
    group = df_summary[df_summary["system"] == system]
    line_color = COLOR_MAP[system]

    # Mean line
    fig.add_trace(go.Scatter(
        x=group["percent_gait_cycle"],
        y=group["mean_angle"],
        mode="lines",
        line=dict(color=line_color, width=2),
        name=f"{system} mean"
    ))

    # Shaded std band (same color, lighter alpha)
    fig.add_trace(go.Scatter(
        x=pd.concat([group["percent_gait_cycle"], group["percent_gait_cycle"][::-1]]),
        y=pd.concat([group["mean_angle"] - group["std_angle"],
                     (group["mean_angle"] + group["std_angle"])[::-1]]),
        fill="toself",
        fillcolor=line_color.replace("rgb", "rgba").replace(")", ",0.2)"),
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=False
    ))

fig.update_layout(
    xaxis_title="<b>% Gait Cycle</b>",
    yaxis_title="<b>Ankle Angle (deg)</b>",
    title="<b>Neutral Ankle Angle over Gait Cycle (Mean ± SD)</b>",
    yaxis = dict(
        tickfont = dict(size=24),
    ),
    xaxis = dict(
        tickfont = dict(size=24),
    ),
)

fig.update_xaxes(title_font=dict(size=30))
fig.update_yaxes(title_font=dict(size=30))
fig.show()