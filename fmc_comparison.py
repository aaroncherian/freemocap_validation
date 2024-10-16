from skellyalign.models.alignment_config import SpatialAlignmentConfig
from skellyalign.run_alignment import run_ransac_spatial_alignment
from skellyalign.plots.scatter_3d import plot_3d_scatter
from pathlib import Path
import numpy as np
from skellymodels.create_model_skeleton import create_mediapipe_skeleton_model
from skellymodels.model_info.mediapipe_model_info import MediapipeModelInfo

path_to_first_freemocap_recording_folder = Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_13_37_32_MDN_treadmill_1')
path_to_second_freemocap_recording_folder = Path(r'D:\mdn_treadmill_for_testing')


p01_walkrun_trial_1_alignment_config = SpatialAlignmentConfig(
    path_to_freemocap_recording_folder=path_to_first_freemocap_recording_folder,
    path_to_freemocap_output_data = path_to_second_freemocap_recording_folder/'output_data'/'mediapipe_body_3d_xyz.npy',
    freemocap_skeleton_function = create_mediapipe_skeleton_model,
    path_to_qualisys_output_data = path_to_first_freemocap_recording_folder/'output_data'/'mediapipe_body_3d_xyz.npy',
    qualisys_skeleton_function= create_mediapipe_skeleton_model,
    markers_for_alignment=MediapipeModelInfo.landmark_names,
    frames_to_sample=20,
    max_iterations=50,
    inlier_threshold=60
)

if __name__ == '__main__':
    from skellyalign.run_alignment import run_ransac_spatial_alignment

    aligned_freemocap_skeleton_model, transformation_matrix = run_ransac_spatial_alignment(p01_walkrun_trial_1_alignment_config)

    plot_3d_scatter(freemocap_data=aligned_freemocap_skeleton_model.marker_data_as_numpy[:,:,:], qualisys_data= np.load(p01_walkrun_trial_1_alignment_config.path_to_qualisys_output_data))
    np.save(path_to_second_freemocap_recording_folder/'output_data'/'aligned_mediapipe_body_3d_xyz.npy', aligned_freemocap_skeleton_model.original_marker_data_as_numpy)




