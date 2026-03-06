from pathlib import Path
import numpy as np

TRACKER = "mediapipe"

path_to_recordings = Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera")
list_of_folders = [p for p in path_to_recordings.iterdir() if p.is_dir()]


list_of_valid_folders: list[Path] = []
for p in list_of_folders:
    if (p / "validation").is_dir():
        list_of_valid_folders.append(p)
    else:
        print(f"Skipping {p}")
scaling_list = []

for valid_folder in list_of_valid_folders:
    print(f"Processing {valid_folder}...")
    path_to_scaling = valid_folder / "validation" / TRACKER / "transformation_3d.npy"
    scaling = np.load(path_to_scaling)[-1]
    scaling_list.append(scaling)

scaling_factor = np.median(scaling_list)

print(f"Median scaling factor across all recordings: {scaling_factor:.4f}")
print(f"Mean scaling factor across all recordings: {np.mean(scaling_list):.4f}")
print(f"Standard deviation of scaling factors across all recordings: {np.std(scaling_list):.4f}")
f = 2