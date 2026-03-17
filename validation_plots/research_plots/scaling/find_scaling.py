# from pathlib import Path
# import numpy as np

# TRACKER = "vitpose"

# #older out-dated version of the scaling code

# path_to_recordings_list = [
#     Path(r"D:\validation\data\2025_07_31_JSM_pilot\freemocap"),
#     Path(r"D:\validation\data\2025_09_03_OKK\freemocap"),
#     Path(r"D:\validation\data\2025-11-04_ATC"),
#     Path(r"D:\validation\data\2026_01_26_KK"),
#     Path(r"D:\validation\data\2026-01-30-JTM")
# ]

# scaling_by_path: dict[Path, list[float]] = {}

# for starting_path in path_to_recordings_list:
#     print(f"\nChecking starting path: {starting_path}")
    
#     if not starting_path.exists():
#         print(f"  Path does not exist, skipping.")
#         continue

#     folders = [p for p in starting_path.iterdir() if p.is_dir()]
#     valid_folders = []

#     for folder in folders:
#         if (folder / "validation").is_dir():
#             valid_folders.append(folder)
#         else:
#             print(f"  Skipping {folder}")

#     scaling_list = []

#     for valid_folder in valid_folders:
#         path_to_scaling = valid_folder / "validation" / TRACKER / "transformation_3d.npy"

#         if not path_to_scaling.exists():
#             print(f"  Missing scaling file: {path_to_scaling}")
#             continue

#         print(f"  Processing {valid_folder.name}...")
#         scaling = np.load(path_to_scaling)[-1]
#         scaling_list.append(float(scaling))

#     scaling_by_path[starting_path] = scaling_list

# print("\n--- Scaling summary by starting path ---")
# for starting_path, scaling_list in scaling_by_path.items():
#     print(f"\n{starting_path}")
#     if len(scaling_list) == 0:
#         print("  No valid scaling values found.")
#         continue

#     print(f"  n = {len(scaling_list)}")
#     print(f"  Median scaling factor: {np.median(scaling_list):.4f}")
#     print(f"  Mean scaling factor:   {np.mean(scaling_list):.4f}")
#     print(f"  Std scaling factor:    {np.std(scaling_list):.4f}")
#     print(f"  Min scaling factor:    {np.min(scaling_list):.4f}")
#     print(f"  Max scaling factor:    {np.max(scaling_list):.4f}")