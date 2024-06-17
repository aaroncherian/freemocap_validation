from skellymodels.create_model_skeleton import create_mediapipe_skeleton_model, create_rigid_mediapipe_skeleton_model
from freemocap_functions.calculate_rigid_bones import enforce_rigid_bones_from_skeleton
from freemocap_functions.calculate_center_of_mass import calculate_center_of_mass_from_skeleton
from pathlib import Path
import numpy as np

path_to_recording_folder = Path(r'D:\2024-04-25_P01\1.0_recordings\P01_WalkRun_Trial1_four_cameras')


freemocap_3d_data = np.load(path_to_recording_folder/'output_data'/'mediapipe_body_3d_xyz.npy')


mediapipe_skeleton_model = create_mediapipe_skeleton_model()
mediapipe_skeleton_model.integrate_freemocap_3d_data(freemocap_3d_data)

rigid_marker_data = enforce_rigid_bones_from_skeleton(mediapipe_skeleton_model)

rigid_skeleton_model = create_rigid_mediapipe_skeleton_model()
rigid_skeleton_model.integrate_freemocap_3d_data(rigid_marker_data)

rigid_skeleton_segment_com, rigid_body_com = calculate_center_of_mass_from_skeleton(rigid_skeleton_model)

np.save(path_to_recording_folder/'output_data'/'mediapipe_body_3d_xyz_rigid.npy', rigid_skeleton_model.marker_data_as_numpy)
np.save(path_to_recording_folder/'output_data'/'center_of_mass'/'rigid_total_body_center_of_mass_xyz.npy', rigid_body_com)

f = 2 

