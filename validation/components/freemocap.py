from validation.datatypes.data_component import DataComponent
from validation.utils.io_helpers import load_csv, load_numpy, save_numpy


FREEMOCAP_TIMESTAMPS = DataComponent(
    name="freemocap_timestamps",
    filename="{recording_name}_timestamps.csv",
    relative_path="synchronized_videos",
    loader=load_csv,
)

FREEMOCAP_PRE_SYNC_JOINT_CENTERS = DataComponent(
    name = "freemocap_pre_synced",
    filename = '{tracker}_body_3d_xyz.npy',
    relative_path="output_data",
    loader=load_numpy
)

TRANSFORMATION_MATRIX = DataComponent(
    name = "transformation_matrix",
    filename = "transformation_3d.npy",
    relative_path = "validation/{tracker}",
    loader= load_numpy,
    saver = save_numpy
)

FREEMOCAP_JOINT_CENTERS = DataComponent(
    name = "freemocap_aligned_3d",
    filename = "{tracker}_body_3d_xyz.npy",
    relative_path = "validation/{tracker}",
    loader = load_numpy,
    saver = save_numpy
)

FREEMOCAP_RIGID_JOINT_CENTERS = DataComponent(
    name = "freemocap_rigid_aligned_3d",
    filename = "{tracker}_body_rigid_3d_xyz.npy",
    relative_path = "validation/{tracker}",
    loader = load_numpy,
    saver = save_numpy
)

FREEMOCAP_COM = DataComponent(
    name = "freemocap_center_of_mass",
    filename = "{tracker}_body_total_body_com.npy",
    relative_path = "validation/{tracker}",
    loader=load_numpy,
    saver=save_numpy
)