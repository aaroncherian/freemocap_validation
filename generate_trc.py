from pathlib import Path
import numpy as np
import pandas as pd

from trc_utils.find_good_frame import find_good_frame
from trc_utils.skeleton_y_up_alignment import align_skeleton_with_origin
from trc_utils import create_trc
from skellymodels.model_info.mediapipe_model_info import MediapipeModelInfo


path_to_recording_folder = Path(r'D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_09_05_TF01_flexion_pos_2_8_trial_1')

path_to_data = path_to_recording_folder/'output_data'/'mediapipe_body_3d_xyz.npy'

# data_array_folder_path = freemocap_data_folder_path / sessionID / data_array_folder
skel3d_data = np.load(path_to_data)


good_frame = find_good_frame(skel3d_data,MediapipeModelInfo.landmark_names, .3,debug = False)
y_up_skel_data = align_skeleton_with_origin(skeleton_data=skel3d_data, good_frame=good_frame, skeleton_indices=MediapipeModelInfo.landmark_names, debug=True)


skel_3d_flat = create_trc.flatten_mediapipe_data(y_up_skel_data)
skel_3d_flat_dataframe = pd.DataFrame(skel_3d_flat)

create_trc.create_trajectory_trc(skeleton_data_frame=skel_3d_flat_dataframe, keypoints_names=MediapipeModelInfo.landmark_names, frame_rate=30, data_array_folder_path=path_to_recording_folder/'output_data')
f = 2

# create_trc.create_trajectory_trc(skel_3d_flat_dataframe,mediapipe_indices, 30, data_array_folder_path)