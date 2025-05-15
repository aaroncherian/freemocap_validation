from skellyalign.models.alignment_config import SpatialAlignmentConfig
from skellyalign.run_alignment import run_ransac_spatial_alignment
from skellyalign.plots.scatter_3d import plot_3d_scatter
from markers_for_alignment import markers_for_alignment
from pathlib import Path
import numpy as np
from skellymodels.create_model_skeleton import create_mediapipe_skeleton_model, create_qualisys_tf01_skeleton_model

path_to_freemocap_recording_folder=Path(r"D:\2023-06-07_TF01\1.0_recordings\treadmill_calib\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1")
output_data_folder = 'mediapipe_output_data'

path_to_aligned_data_folder = path_to_freemocap_recording_folder/output_data_folder/'aligned_data'
path_to_aligned_data_folder.mkdir(parents=True, exist_ok=True)

path_to_aligned_center_of_mass_folder = path_to_aligned_data_folder/'center_of_mass'
path_to_aligned_center_of_mass_folder.mkdir(parents=True, exist_ok=True)


tf01_flex_neutral_config = SpatialAlignmentConfig(
    path_to_freemocap_recording_folder=path_to_freemocap_recording_folder,
    path_to_freemocap_output_data = path_to_freemocap_recording_folder/output_data_folder/'mediapipe_body_3d_xyz.npy',
    freemocap_skeleton_function = create_mediapipe_skeleton_model,
    path_to_qualisys_output_data = path_to_freemocap_recording_folder/'qualisys_data'/ 'qualisys_joint_centers_3d_xyz.npy',
    qualisys_skeleton_function= create_qualisys_tf01_skeleton_model,
    markers_for_alignment=markers_for_alignment,
    frames_to_sample=20,
    max_iterations=50,
    inlier_threshold=40
)


# p01_nih_alignment_normal_mp_config = SpatialAlignmentConfig(
#     path_to_freemocap_recording_folder=path_to_freemocap_recording_folder,
#     path_to_freemocap_output_data = path_to_freemocap_recording_folder/output_data_folder/'mediapipe_body_3d_xyz.npy',
#     freemocap_skeleton_function = create_mediapipe_skeleton_model,
#     path_to_qualisys_output_data = path_to_freemocap_recording_folder/'qualisys_data'/ 'qualisys_joint_centers_3d_xyz.npy',
#     qualisys_skeleton_function= create_qualisys_tf01_skeleton_model,
#     markers_for_alignment=markers_for_alignment,
#     frames_to_sample=20,
#     max_iterations=50,
#     inlier_threshold=40
# )



if __name__ == '__main__':
    from skellyalign.run_alignment import run_ransac_spatial_alignment
    from freemocap_functions.calculate_center_of_mass import calculate_center_of_mass_from_skeleton
    from freemocap_functions.calculate_rigid_bones import enforce_rigid_bones_from_skeleton

    aligned_freemocap_skeleton_model, transformation_matrix = run_ransac_spatial_alignment(tf01_flex_neutral_config)

    # merged_segment_com_data, total_body_com = calculate_center_of_mass_from_skeleton(aligned_freemocap_skeleton_model)
    plot_3d_scatter(freemocap_data=aligned_freemocap_skeleton_model.marker_data_as_numpy, qualisys_data= np.load(tf01_flex_neutral_config.path_to_qualisys_output_data))
    np.save(path_to_aligned_data_folder/'mediapipe_body_3d_xyz.npy', aligned_freemocap_skeleton_model.original_marker_data_as_numpy)
    # np.save(path_to_aligned_center_of_mass_folder/'segmentCOM_frame_joint_xyz_rigid.npy', merged_segment_com_data)
    # np.save(path_to_aligned_center_of_mass_folder/'total_body_center_of_mass_xyz_rigid.npy', total_body_com)


    
