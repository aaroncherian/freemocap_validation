from validation.datatypes.data_component import DataComponent
from validation.utils.io_helpers import load_numpy, save_numpy, load_qualisys_timestamp_from_tsv, load_qualisys_tsv, load_csv, save_csv
from validation.utils.save_trcs import save_as_trc

QUALISYS_MARKERS = DataComponent(
    name="qualisys_markers",
    filename="qualisys_exported_markers.tsv",
    relative_path="validation/qualisys",
    loader=load_qualisys_tsv,
)

QUALISYS_START_TIME =  DataComponent(
    name="qualisys_start_time",
    filename="qualisys_exported_markers.tsv",
    relative_path="validation/qualisys",
    loader=load_qualisys_timestamp_from_tsv
)

QUALISYS_SYNCED_JOINT_CENTERS = DataComponent(
    name="qualisys_synced_joint_centers",
    filename = "qualisys_body_3d_xyz.npy",
    relative_path = "validation/qualisys",
    loader = load_numpy,
    saver = save_numpy
)



QUALISYS_SYNCED_MARKER_DATA = DataComponent(
    name="qualisys_synced_marker_data",
    filename = "qualisys_synced_markers.csv",
    relative_path = "validation/qualisys",
    loader = load_csv,
    saver = save_csv,
)

QUALISYS_ACTOR = DataComponent(
    name = "qualisys_actor",
    filename = None,
    relative_path = None,
)

QUALISYS_COM = DataComponent(
    name = "qualisys_center_of_mass",
    filename = "qualisys_body_total_body_com.npy",
    relative_path = "validation/qualisys",
    loader=load_numpy,
    saver = save_numpy
)

QUALISYS_TRC = DataComponent(
    name = "qualisys_trc",
    filename = "qualisys_body_3d_xyz.trc",
    relative_path = "validation/qualisys",
    loader = None,
    saver = save_as_trc
)