from skellyalign.models.alignment_config import SpatialAlignmentConfig
from skellyalign.run_alignment import run_ransac_spatial_alignment
from skellyalign.plots.scatter_3d import plot_3d_scatter
from p01_marker_set import markers_for_alignment
from pathlib import Path
import numpy as np
from skellymodels.create_model_skeleton import create_mediapipe_skeleton_model, create_qualisys_skeleton_model

path_to_freemocap_recording_folder=Path(r"D:\2024-04-25_P01\1.0_recordings\sesh_2024-04-25_14_45_59_P01_NIH_Trial1")

path_to_aligned_data_folder = path_to_freemocap_recording_folder/'aligned_data'
path_to_aligned_data_folder.mkdir(parents=True, exist_ok=True)

path_to_aligned_center_of_mass_folder = path_to_aligned_data_folder/'center_of_mass'
path_to_aligned_center_of_mass_folder.mkdir(parents=True, exist_ok=True)


p01_nih_alignment_config = SpatialAlignmentConfig(
    path_to_freemocap_recording_folder=path_to_freemocap_recording_folder,
    path_to_freemocap_output_data = path_to_freemocap_recording_folder/'output_data'/'mediapipe_body_3d_xyz.npy',
    freemocap_skeleton_function = create_mediapipe_skeleton_model,
    path_to_qualisys_output_data = path_to_freemocap_recording_folder/'qualisys_data'/ 'qualisys_joint_centers_3d_xyz.npy',
    qualisys_skeleton_function= create_qualisys_skeleton_model,
    markers_for_alignment=markers_for_alignment,
    frames_to_sample=20,
    max_iterations=50,
    inlier_threshold=40
)


if __name__ == '__main__':
    from skellyalign.run_alignment import run_ransac_spatial_alignment
    from freemocap_functions.calculate_center_of_mass import calculate_center_of_mass_from_skeleton

    aligned_freemocap_skeleton_model, transformation_matrix = run_ransac_spatial_alignment(p01_nih_alignment_config)

    merged_segment_com_data, total_body_com = calculate_center_of_mass_from_skeleton(aligned_freemocap_skeleton_model)
    # plot_3d_scatter(freemocap_data=aligned_freemocap_data, qualisys_data= np.load(p01_nih_alignment_config.path_to_qualisys_output_data))
    np.save(path_to_aligned_data_folder/'mediapipe_body_3d_xyz.npy', aligned_freemocap_skeleton_model.original_marker_data_as_numpy)
    np.save(path_to_aligned_center_of_mass_folder/'segmentCOM_frame_joint_xyz.npy', merged_segment_com_data)
    np.save(path_to_aligned_center_of_mass_folder/'total_body_center_of_mass_xyz.npy', total_body_com)



    f = 2
    

# path_to_recording = Path(r"D:\2024-04-25_P01\1.0_recordings\sesh_2024-04-25_14_45_59_P01_NIH_Trial1")
# path_to_freemocap_output_data = path_to_recording/'output_data'/'mediapipe_body_3d_xyz.npy'
# path_to_qualisys_output_data = path_to_recording/'qualisys_data'/ 'qualisys_joint_centers_3d_xyz.npy'
# path_to_save_aligned_data = path_to_recording/'output_data'/'aligned_data'/'mediapipe_body_3d_xyz.npy'
# path_to_save_aligned_data.mkdir(parents=True, exist_ok=True)


# freemocap_data = np.load(path_to_freemocap_output_data)
# freemocap_model = create_mediapipe_skeleton_model()
# freemocap_model.integrate_freemocap_3d_data(freemocap_data)


# qualisys_data = np.load(path_to_qualisys_output_data)
# qualisys_model = create_qualisys_skeleton_model()
# qualisys_model.integrate_freemocap_3d_data(qualisys_data)

# alignment_config = SpatialAlignmentConfig(
#     path_to_freemocap_output_data = path_to_freemocap_output_data,
#     freemocap_skeleton_function = create_mediapipe_skeleton_model,
#     path_to_qualisys_output_data = path_to_qualisys_output_data,
#     qualisys_skeleton_function= create_qualisys_skeleton_model,
#     markers_for_alignment=markers_for_alignment,
#     frames_to_sample=20,
#     max_iterations=50,
#     inlier_threshold=40
# )

# aligned_freemocap_data, transformation_matrix = run_ransac_spatial_alignment(alignment_config)
# plot_3d_scatter(freemocap_data=aligned_freemocap_data, qualisys_data=qualisys_model.marker_data_as_numpy)
# np.save(path_to_save_aligned_data, aligned_freemocap_data)

    
# # def run_spatial_alignment(SpatialAlignmentConfig):
    


f =2 