from pathlib import Path
import csv
import pandas as pd

def create_trajectory_trc(skeleton_data_frame, keypoints_names, frame_rate, data_array_folder_path, tracker_name: str):
    # Header values
    data_rate = camera_rate = orig_data_rate = frame_rate
    num_frames = len(skeleton_data_frame)
    num_markers = len(keypoints_names)
    units = 'mm'  # ← Change to meters since your data is in meters
    orig_data_start_frame = 0
    orig_num_frames = num_frames - 1

    # Output path
    trc_filename = f'{tracker_name}_body_3d_xyz.trc'
    trc_path = data_array_folder_path / trc_filename

    # Add Frame and Time columns (as numeric)
    skeleton_data_frame = skeleton_data_frame.copy()
    skeleton_data_frame.insert(0, "Frame", list(range(num_frames)))
    skeleton_data_frame.insert(1, "Time", skeleton_data_frame["Frame"] / float(camera_rate))

    # Ensure all entries are numeric
    skeleton_data_frame = skeleton_data_frame.apply(pd.to_numeric, errors='coerce')

    # Write TRC file
    with open(trc_path, 'wt', newline='', encoding='utf-8') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')

        # Header rows (5 total)
        tsv_writer.writerow(["PathFileType", "4", "(X/Y/Z)", trc_filename])
        tsv_writer.writerow(["DataRate", "CameraRate", "NumFrames", "NumMarkers", "Units",
                             "OrigDataRate", "OrigDataStartFrame", "OrigNumFrames"])
        tsv_writer.writerow([data_rate, camera_rate, num_frames, num_markers, units,
                             orig_data_rate, orig_data_start_frame, orig_num_frames])

        header_names = ['Frame#', 'Time']

        for keypoint in keypoints_names:
            header_names.append(keypoint)
            header_names.append("")
            header_names.append("")
        tsv_writer.writerow(header_names)

        axis_names = ["", ""]  # empty for Frame# and Time
        for i in range(1, len(keypoints_names) + 1):
            axis_names.extend([f"X{i}", f"Y{i}", f"Z{i}"])
        tsv_writer.writerow(axis_names)

        # Data rows
        for _, row in skeleton_data_frame.iterrows():
            flat_row = [str(val) for val in row.values]
            tsv_writer.writerow(flat_row)

    print(f"TRC file successfully written to: {trc_path}")
    return trc_path

def flatten_data(skeleton_3d_data):
    num_frames = skeleton_3d_data.shape[0]
    num_markers = skeleton_3d_data.shape[1]

    skeleton_data_flat = skeleton_3d_data.reshape(num_frames,num_markers*3)

    return skeleton_data_flat
