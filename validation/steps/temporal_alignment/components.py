from validation.datatypes.data_component import DataComponent
from validation.registry import register
from validation.utils.io_helpers import load_csv, load_qualisys_timestamp_from_tsv, load_qualisys_tsv, save_numpy
import numpy as np

FREEMOCAP_TIMESTAMPS = DataComponent(
    name="freemocap_timestamps",
    filename=lambda base_dir: f"{base_dir.stem}_timestamps.csv",
    relative_path="synchronized_videos",
    loader=load_csv,
)
register(FREEMOCAP_TIMESTAMPS)

QUALISYS_MARKERS = DataComponent(
    name="qualisys_markers",
    filename="qualisys_exported_markers.tsv",
    relative_path="validation/qualisys",
    loader=load_qualisys_tsv,
)
register(QUALISYS_MARKERS)

QUALISYS_START_TIME =  DataComponent(
    name="qualisys_start_time",
    filename="qualisys_exported_markers.tsv",
    relative_path="validation/qualisys",
    loader=load_qualisys_timestamp_from_tsv
)
register(QUALISYS_START_TIME)

QUALISYS_SYNCED_JOINT_CENTERS = DataComponent(
    name="qualisys_synced_joint_centers",
    filename = "qualisys_skeleton_3d.npy",
    relative_path = "validation/qualisys",
    saver= save_numpy
)

FREEMOCAP_ACTOR = DataComponent(
    name = "freemocap_actor",
    filename=None,
    relative_path=None,
)

# if __name__ == '__main__':
#     from pathlib import Path
#     recording_dir = Path(r"D:\2025-03-13_JSM_pilot\freemocap_data\2025-03-13T16_20_37_gmt-4_pilot_jsm_treadmill_walking")
#     timestamps_component = get_component("freemocap_timestamps")
#     print(timestamps_component.full_path(recording_dir))

#     qualisys_markers = get_component('qualisys_markers')
#     print(qualisys_markers.full_path(recording_dir))
