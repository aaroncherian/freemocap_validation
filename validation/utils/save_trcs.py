import pandas as pd
from validation.steps.trc_conversion.core.convert_to_trc import  TRCResult
from pathlib import Path 
import csv

def save_as_trc(path:Path, trc_components:TRCResult):
    skeleton_data_frame = trc_components.dataframe
    keypoints_names = trc_components.landmark_names

    data_rate = camera_rate = orig_data_rate = 30 
    num_frames = len(skeleton_data_frame)
    num_markers = len(keypoints_names)
    units = 'mm'  
    orig_data_start_frame = 0
    orig_num_frames = num_frames - 1

    trc_path = path

    skeleton_data_frame = skeleton_data_frame.copy()
    skeleton_data_frame.insert(0, "Frame", list(range(num_frames)))
    skeleton_data_frame.insert(1, "Time", skeleton_data_frame["Frame"] / float(camera_rate))

    # Ensure all entries are numeric
    skeleton_data_frame = skeleton_data_frame.apply(pd.to_numeric, errors='coerce')

    # Write TRC file
    with open(trc_path, 'wt', newline='', encoding='utf-8') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')

        # Header rows (5 total)
        tsv_writer.writerow(["PathFileType", "4", "(X/Y/Z)", path.stem + '.trc'])
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
