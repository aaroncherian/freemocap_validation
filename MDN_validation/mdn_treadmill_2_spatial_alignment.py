from skellyalign.models.alignment_config import SpatialAlignmentConfig
from skellyalign.run_alignment import run_ransac_spatial_alignment
from skellyalign.plots.scatter_3d import plot_3d_scatter
from MDN_validation.mdn_marker_set import markers_for_alignment
from pathlib import Path
import numpy as np
from skellymodels.create_model_skeleton import create_mediapipe_skeleton_model, create_qualisys_skeleton_model

# path_to_freemocap_recording_folder=Path(r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_13_48_44_MDN_treadmill_2")
path_to_freemocap_recording_folder = Path(r"D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\mdn_treadmill_2_corrected")

path_to_aligned_data_folder = path_to_freemocap_recording_folder/'aligned_data'
path_to_aligned_data_folder.mkdir(parents=True, exist_ok=True)

path_to_aligned_center_of_mass_folder = path_to_aligned_data_folder/'center_of_mass'
path_to_aligned_center_of_mass_folder.mkdir(parents=True, exist_ok=True)


mdn_treadmill_2_alignment = SpatialAlignmentConfig(
    path_to_freemocap_recording_folder=path_to_freemocap_recording_folder,
    path_to_freemocap_output_data = path_to_freemocap_recording_folder/'output_data'/'mediapipe_body_3d_xyz.npy',
    freemocap_skeleton_function = create_mediapipe_skeleton_model,
    path_to_qualisys_output_data = path_to_freemocap_recording_folder/'qualisys_data'/ 'qualisys_joint_centers_3d_xyz.npy',
    qualisys_skeleton_function= create_qualisys_skeleton_model,
    markers_for_alignment=markers_for_alignment,
    frames_to_sample=20,
    max_iterations=100,
    inlier_threshold=40
)


if __name__ == '__main__':
    from skellyalign.run_alignment import run_ransac_spatial_alignment
    from freemocap_functions.calculate_center_of_mass import calculate_center_of_mass_from_skeleton
    from freemocap_functions.calculate_rigid_bones import enforce_rigid_bones_from_skeleton

    # plot_3d_scatter(freemocap_data=np.load(mdn_treadmill_2_alignment.path_to_freemocap_output_data), qualisys_data= np.load(mdn_treadmill_2_alignment.path_to_qualisys_output_data))

    aligned_freemocap_skeleton_model, transformation_matrix = run_ransac_spatial_alignment(mdn_treadmill_2_alignment)

    merged_segment_com_data, total_body_com = calculate_center_of_mass_from_skeleton(aligned_freemocap_skeleton_model)
    plot_3d_scatter(freemocap_data=aligned_freemocap_skeleton_model.marker_data_as_numpy, qualisys_data= np.load(mdn_treadmill_2_alignment.path_to_qualisys_output_data))
    np.save(path_to_aligned_data_folder/'mediapipe_body_3d_xyz.npy', aligned_freemocap_skeleton_model.original_marker_data_as_numpy)
    np.save(path_to_aligned_center_of_mass_folder/'segmentCOM_frame_joint_xyz.npy', merged_segment_com_data)
    np.save(path_to_aligned_center_of_mass_folder/'total_body_center_of_mass_xyz.npy', total_body_com)


    
