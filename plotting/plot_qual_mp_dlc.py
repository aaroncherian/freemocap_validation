from pathlib import Path
import pandas as pd
import plotly.express as px



def build_combined_df(system_to_csv, frames=(1000,1500),
                      markers=("right_knee","right_heel","right_ankle","right_foot_index")) -> pd.DataFrame:

    start, stop = frames
    dfs = []
    for tracker, csv_path in system_to_csv.items():
        tmp = pd.read_csv(csv_path)
        needed = {"frame","keypoint","x","y","z"}
        missing = needed - set(tmp.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {missing}")

        tmp = tmp[(tmp["frame"] >= start) & (tmp["frame"] <= stop) & (tmp["keypoint"].isin(markers))].copy()
        tmp.insert(0, "tracker", tracker)
        dfs.append(tmp)

    if not dfs:
        return pd.DataFrame(columns=["tracker","frame","keypoint","x","y","z"])

    combined = pd.concat(dfs, ignore_index=True)
    cols = ["tracker","frame","keypoint","x","y","z"]
    other = [c for c in combined.columns if c not in cols]
    return combined[cols + other]


import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def simple_marker_grid(
    df: pd.DataFrame,
    title="3D Marker Trajectories (Flexion Neutral Trial)",
    markers=("right_knee","right_ankle","right_heel","right_foot_index"),
    axes=("x","y","z"),
    height=1200, width=1800,
    rolling=None,
):
    # normalize tracker names
    tracker_norm = {
        "qualisys":"Qualisys","Qualisys":"Qualisys",
        "mediapipe":"MediaPipe","medipipe":"MediaPipe","MediaPipe":"MediaPipe",
        "mediapipe_dlc":"MediaPipe+DLC","mediapipe+dlc":"MediaPipe+DLC",
        "mp_dlc":"MediaPipe+DLC","mp+dlc":"MediaPipe+DLC","MediaPipe+DLC":"MediaPipe+DLC",
    }
    df = df.copy()
    df["tracker"] = df["tracker"].map(lambda s: tracker_norm.get(str(s), str(s)))
    df = df[df["keypoint"].isin(markers)]

    # long form
    long = df.melt(id_vars=["tracker","frame","keypoint"], value_vars=list(axes),
                   var_name="axis", value_name="pos")

    # de-dup + sort
    long = (long.groupby(["tracker","keypoint","axis","frame"], as_index=False)["pos"]
                 .mean()
                 .sort_values(["keypoint","axis","tracker","frame"]))
    if rolling and rolling > 1:
        long["pos"] = (long.groupby(["tracker","keypoint","axis"])["pos"]
                           .transform(lambda s: s.rolling(rolling, center=True, min_periods=1).mean()))

    label_map = {"right_knee":"KNEE","right_ankle":"ANKLE",
                 "right_heel":"HEEL","right_foot_index":"TOE"}
    row_labels = [label_map[m] for m in markers]
    col_labels = [a.upper() for a in axes]

    fig = make_subplots(
        rows=len(markers), cols=len(axes),
        shared_xaxes=False, shared_yaxes=False,
        horizontal_spacing=0.06, vertical_spacing=0.06,
        subplot_titles=col_labels + [""]*((len(markers)-1)*len(axes))
    )

    # >>> emphasis hierarchy & draw order
    draw_order = ["Qualisys", "MediaPipe+DLC", "MediaPipe", ]  # DLC added last (on top)
    colors = {"Qualisys":"black", "MediaPipe":"#d62728", "MediaPipe+DLC":"#1f77b4"}  # red, blue
    dashes = {"Qualisys":"solid", "MediaPipe":"solid", "MediaPipe+DLC":"solid"}
    widths = {"Qualisys":3.0, "MediaPipe":2.2, "MediaPipe+DLC":3.2}
    opac   = {"Qualisys":.80, "MediaPipe":0.45, "MediaPipe+DLC":.90}

    for r, marker in enumerate(markers, start=1):
        for c, axis in enumerate(axes, start=1):
            sub = long[(long["keypoint"]==marker) & (long["axis"]==axis)]
            for tr in draw_order:
                s = sub[sub["tracker"]==tr]
                if s.empty: 
                    continue
                fig.add_trace(
                    go.Scattergl(
                        x=s["frame"], y=s["pos"],
                        mode="lines",
                        line=dict(color=colors[tr], width=widths[tr], dash=dashes[tr]),
                        opacity=opac[tr],
                        name=tr, legendgroup=tr, legendrank=draw_order.index(tr),
                        showlegend=(r==1 and c==1),
                        hovertemplate=f"{row_labels[r-1]} | {axis.upper()} | {tr}<br>"
                                      "frame=%{x}<br>pos=%{y:.2f} mm<extra></extra>",
                    ),
                    row=r, col=c
                )
            if c==1:
                fig.update_yaxes(title_text=row_labels[r-1], row=r, col=c)
            if r==len(markers):
                fig.update_xaxes(title_text="Frame", row=r, col=c)

    fig.update_layout(
        title=title,
        height=height, width=width,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1.0),
        hovermode="x unified",
        font=dict(size=12),
        plot_bgcolor="white",
    )
    # subtle grids for readability
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    return fig

# --- Usage ---
# df = build_combined_df(path_dict, frames=(1000,1500))
# fig = simple_marker_grid(df, rolling=None)   # set rolling=5 if you want a light visual smooth
# fig.show()
# fig.write_html("simple_marker_grid.html", include_plotlyjs="cdn")


path_dict = {
    "qualisys": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1\validation\qualisys\freemocap_data_by_frame.csv"),
    "mediapipe": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1\validation\mediapipe\freemocap_data_by_frame.csv"),
    "mediapipe_dlc": Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1\validation\mediapipe_dlc\freemocap_data_by_frame.csv")}

df = build_combined_df(path_dict, frames=(1000,1500))

fig = simple_marker_grid(df, title="3D Marker Trajectories (Flexion Neutral Trial)")
fig.show()

f = 2
