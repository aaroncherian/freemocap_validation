import numpy as np
import pandas as pd
from validation.utils.actor_utils import make_freemocap_actor_from_landmarks
from dataclasses import dataclass

@dataclass
class TRCResult:
    dataframe: pd.DataFrame
    landmark_names: list[str]

def flatten_data(skeleton_3d_data):
    num_frames = skeleton_3d_data.shape[0]
    num_markers = skeleton_3d_data.shape[1]

    skeleton_data_flat = skeleton_3d_data.reshape(num_frames,num_markers*3)

    return skeleton_data_flat

def create_freemocap_trc(tracker_name:str, tracked_points_data:np.ndarray):


    human = make_freemocap_actor_from_landmarks(freemocap_tracker=tracker_name, 
                                                landmarks=tracked_points_data)

    body_trajectory = human.body.trajectories['3d_xyz']
    z_up_data = body_trajectory.as_numpy
    y_up_data = z_up_data.copy()
    y_up_data = z_up_data[..., [1, 2, 0]]   # -> [X_forward, Y_up, Z_right]

    skel_3d_flat = flatten_data(y_up_data)
    skel_3d_flat_dataframe = pd.DataFrame(skel_3d_flat)

    return TRCResult(
        dataframe=skel_3d_flat_dataframe,
        landmark_names=body_trajectory.landmark_names
    )
import re

def create_qualisys_trc(df: pd.DataFrame, frame_rate: float = 30):
    # --- 0) pull off frame & time BEFORE dropping them ---
    # (qualisys CSV often has 'Frame' and 'Time' columns)
    if 'Frame' in df.columns:
        frames = df['Frame'].astype(int).to_numpy()
    else:
        frames = np.arange(len(df), dtype=int)

    if 'Time' in df.columns:
        times = df['Time'].astype(float).to_numpy()
    else:
        times = frames / frame_rate

    # now drop all non‐marker columns
    df = df.loc[:, ~df.columns.str.match(r"^(Frame|Time|unix_timestamps)$")]

    # --- 1) regex‐based marker extraction (unchanged) ---
    marker_axes = []
    for col in df.columns:
        m = re.match(r"(.+?)[\s\._]([XYZ])$", col)
        if m:
            marker_axes.append((col, *m.groups()))

    axis_sets = {}
    for _, name, ax in marker_axes:
        axis_sets.setdefault(name, set()).add(ax)

    marker_names = []
    seen = set()
    for col, name, ax in marker_axes:
        if name not in seen and axis_sets.get(name) == {"X","Y","Z"}:
            marker_names.append(name)
            seen.add(name)

    # --- 2) build lab coords and remap to OpenSim (your working mapping) ---
    coord_order = ["X","Y","Z"]
    stack = [ df[f"{name} {ax}"].to_numpy(float)
              for name in marker_names
              for ax   in coord_order ]
    data_lab = np.stack(stack, axis=1).reshape(len(df), len(marker_names), 3)

    data_os = np.empty_like(data_lab)
    data_os[...,0] =  data_lab[...,0]    # X_os ← X_q
    data_os[...,1] =  data_lab[...,2]    # Y_os ← Z_q
    data_os[...,2] = -data_lab[...,1]    # Z_os ← -Y_q

    # --- 3) flatten & prepend Frame#/Time columns ---
    flat = flatten_data(data_os)                             # shape (n_frames, n_markers*3)
    full = np.column_stack([frames, times, flat])           # now (n_frames, 2 + n_markers*3)
    columns = ["Frame#", "Time"] + [f for name in marker_names for f in (f"{name}X",f"{name}Y",f"{name}Z")]

    flat_df = pd.DataFrame(full, columns=columns)

    return TRCResult(dataframe=flat_df, landmark_names=marker_names)




#     create_trajectory_trc(skeleton_data_frame=skel_3d_flat_dataframe, 
#                                     keypoints_names=body_trajectory.landmark_names, 
#                                     frame_rate=30, 
#                                     data_array_folder_path=path_to_recording_folder/'validation'/f'{tracker_name}',
#                                     tracker_name=tracker_name)
# f = 2

# create_trc.create_trajectory_trc(skel_3d_flat_dataframe,mediapipe_indices, 30, data_array_folder_path)