from pathlib import Path
import numpy as np

TRACKER = "rtmpose"

path_to_recordings_list = [
    Path(r"D:\validation\data\2025_07_31_JSM_pilot\freemocap"),
    Path(r"D:\validation\data\2025_09_03_OKK\freemocap"),
    Path(r"D:\validation\data\2025-11-04_ATC"),
    Path(r"D:\validation\data\2026_01_26_KK"),
    Path(r"D:\validation\data\2026-01-30-JTM")
]

path_to_recordings = Path(r"D:\validation\data\2025_07_31_JSM_pilot\freemocap")

complete_list_of_folders = []
for path in path_to_recordings_list:
    folders = [p for p in path.iterdir() if p.is_dir()]
    complete_list_of_folders.extend(folders)
# list_of_folders = [p for p in path_to_recordings.iterdir() if p.is_dir()]


list_of_valid_folders: list[Path] = []
for p in complete_list_of_folders:
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