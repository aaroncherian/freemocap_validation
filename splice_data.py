from pathlib import Path
import numpy as np


length_to_splice = 2576
root_path = Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera")
trial_name = r"sesh_2023-06-07_12_03_15_TF01_flexion_neg_2_8_trial_1"

trackers = ["mediapipe", "rtmpose", "rtmpose_dlc"]

for tracker in trackers:
    path_to_output_data = root_path / trial_name/"output_data"/tracker/f"{tracker}_body_3d_xyz.npy"
    loaded_data = np.load(path_to_output_data)
    loaded_data_spliced = loaded_data[:length_to_splice]
    np.save(path_to_output_data, loaded_data_spliced)
    print(f"Spliced {tracker} data to length {length_to_splice} and saved to {root_path / trial_name / f'{tracker}_body_3d_xyz.npy'}")