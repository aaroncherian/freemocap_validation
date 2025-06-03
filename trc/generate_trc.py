from pathlib import Path
import numpy as np
import pandas as pd

from trc_utils.find_good_frame import find_good_frame
from trc_utils.skeleton_y_up_alignment import align_skeleton_with_origin
from trc_utils import create_trc
from skellymodels.model_info.mediapipe_model_info import MediapipeModelInfo
from validation.utils.actor_utils import make_freemocap_actor_from_landmarks

tracker_name = 'rtmpose'

path_to_recording_folder = Path(r'D:\2025-04-23_atc_testing\freemocap\2025-04-23_19-11-05-612Z_atc_test_walk_trial_2')

path_to_data = path_to_recording_folder/'validation'/f'{tracker_name}'/f'{tracker_name}_body_3d_xyz.npy'

skel3d_data = np.load(path_to_data)

human = make_freemocap_actor_from_landmarks(freemocap_tracker=tracker_name, landmarks=skel3d_data)

# good_frame = find_good_frame(skel3d_data,MediapipeModelInfo.landmark_names, .3,debug = False)
# y_up_skel_data = align_skeleton_with_origin(skeleton_data=skel3d_data, good_frame=good_frame, skeleton_indices=MediapipeModelInfo.landmark_names, debug=True)
body_trajectory = human.body.trajectories['3d_xyz']
z_up_data = body_trajectory.as_numpy
y_up_data = z_up_data.copy()
y_up_data = z_up_data[..., [1, 2, 0]]   # -> [X_forward, Y_up, Z_right]

skel_3d_flat = create_trc.flatten_data(y_up_data)
skel_3d_flat_dataframe = pd.DataFrame(skel_3d_flat)

create_trc.create_trajectory_trc(skeleton_data_frame=skel_3d_flat_dataframe, 
                                 keypoints_names=body_trajectory.landmark_names, 
                                 frame_rate=30, 
                                 data_array_folder_path=path_to_recording_folder/'validation'/f'{tracker_name}',
                                 tracker_name=tracker_name)
f = 2

# create_trc.create_trajectory_trc(skel_3d_flat_dataframe,mediapipe_indices, 30, data_array_folder_path)