from validation.datatypes.data_component import DataComponent


TEMPORAL_ALIGNMENT_COMPONENTS = {
"timestamps": DataComponent(
    name="freemocap_timestamps_dynamic",
    filename=lambda base_dir: f"{base_dir.stem}_timestamps.csv",
    relative_path="synchronized_videos",
),
   "timestamps_prealpha": DataComponent(
    name = "freemocap_timestamps",
    filename = "unix_synced_timestamps.csv",
    relative_path = "synchronized_videos",
),
    "qualisys_markers": DataComponent(
    name = "qualisys_markers",
    filename = "qualisys_exported_markers.tsv",
    relative_path = "validation/qualisys"
)
}

def get_component(key: str):
    if key not in TEMPORAL_ALIGNMENT_COMPONENTS:
        raise KeyError(f"Unknown component '{key} in temporal alignment registry'")
    return TEMPORAL_ALIGNMENT_COMPONENTS[key]


if __name__ == '__main__':
    from pathlib import Path
    recording_dir = Path(r"D:\2025-03-13_JSM_pilot\freemocap_data\2025-03-13T15_47_11_gmt-4_jsm_calibration_3")
    timestamps_component = get_component("timestamps")
    print(timestamps_component.full_path(recording_dir))

    qualisys_markers = get_component('qualisys_markers')
    print(qualisys_markers.full_path(recording_dir))
