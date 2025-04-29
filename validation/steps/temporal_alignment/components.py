from validation.datatypes.data_component import DataComponent
from validation.registry import register
from validation.utils.io_helpers import load_csv, load_qualisys_timestamp_from_tsv, load_qualisys_tsv
import numpy as np

# TEMPORAL_ALIGNMENT_COMPONENTS = {
# "freemocap_timestamps": DataComponent(
#     name="freemocap_timestamps",
#     filename=lambda base_dir: f"{base_dir.stem}_timestamps.csv",
#     relative_path="synchronized_videos",
#     loader=load_csv
# ),
#    "timestamps_prealpha": DataComponent(
#     name = "freemocap_timestamps",
#     filename = "unix_synced_timestamps.csv",
#     relative_path = "synchronized_videos",
# ),
#     "qualisys_markers": DataComponent(
#     name = "qualisys_markers",
#     filename = "qualisys_exported_markers.tsv",
#     relative_path = "validation/qualisys",
#     loader= load_qualisys_tsv_and_start_timestamp
# )
# }

# def get_component(key: str):
#     if key not in TEMPORAL_ALIGNMENT_COMPONENTS:
#         raise KeyError(f"Unknown component '{key} in temporal alignment registry'")
#     return TEMPORAL_ALIGNMENT_COMPONENTS[key]

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




# if __name__ == '__main__':
#     from pathlib import Path
#     recording_dir = Path(r"D:\2025-03-13_JSM_pilot\freemocap_data\2025-03-13T16_20_37_gmt-4_pilot_jsm_treadmill_walking")
#     timestamps_component = get_component("freemocap_timestamps")
#     print(timestamps_component.full_path(recording_dir))

#     qualisys_markers = get_component('qualisys_markers')
#     print(qualisys_markers.full_path(recording_dir))
