from validation.datatypes.data_component import DataComponent
from validation.utils.io_helpers import load_csv, load_numpy, save_numpy


FREEMOCAP_TIMESTAMPS = DataComponent(
    name="freemocap_timestamps",
    filename=lambda base_dir: f"{base_dir.stem}_timestamps.csv",
    relative_path="synchronized_videos",
    loader=load_csv,
)

FREEMOCAP_PRE_SYNC_JOINT_CENTERS = DataComponent(
    name = "freemocap_pre_synced",
    filename = 'mediapipe_skeleton_3d.npy',
    relative_path="output_data",
    loader=load_numpy
)

TRANSFORMATION_MATRIX = DataComponent(
    name = "transformation_matrix",
    filename = "transformation_3d.npy",
    relative_path = "validation/mediapipe",
    loader= load_numpy,
    saver = save_numpy
)

FREEMOCAP_JOINT_CENTERS = DataComponent(
    name = "freemocap_aligned_3d",
    filename = "mediapipe_3d_skeleton.npy",
    relative_path = "validation/mediapipe",
    loader = load_numpy,
    saver = save_numpy
)

FREEMOCAP_COM = DataComponent(
    name = "mediapipe_center_of_mass",
    filename = "mediapipe_body_total_body_com.npy",
    relative_path = "validation/mediapipe",
    loader=load_numpy,
    saver=save_numpy
)