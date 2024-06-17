import pandas as pd
import numpy as np
import csv



def create_trajectory_trc(skeleton_data_frame, keypoints_names, frame_rate, data_array_folder_path):
    
    #Header
    data_rate = camera_rate = orig_data_rate = frame_rate
    num_frames = len(skeleton_data_frame)
    num_frame_range = range(num_frames)
    num_markers = len(keypoints_names)
    units = 'mm'
    orig_data_start_frame = num_frame_range[0]
    orig_num_frames = num_frame_range[-1]
    
    trc_filename = 'mediapipe_body_3d_xyz.trc'
    trc_path = data_array_folder_path/trc_filename

    with open(trc_path, 'wt', newline='', encoding='utf-8') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(["PathFileType",
                            "4", 
                            "(X/Y/Z)",	
                            trc_filename])
        tsv_writer.writerow(["DataRate",
                            "CameraRate",
                            "NumFrames",
                            "NumMarkers", 
                            "Units",
                            "OrigDataRate",
                            "OrigDataStartFrame",
                            "OrigNumFrames"])
        tsv_writer.writerow([data_rate, 
                            camera_rate,
                            num_frames, 
                            num_markers, 
                            units, 
                            orig_data_rate, 
                            orig_data_start_frame, 
                            orig_num_frames])

        header_names = ['Frame#', 'Time']
        for keypoint in keypoints_names:
            header_names.append(keypoint)
            header_names.append("")
            header_names.append("")

        tsv_writer.writerow(header_names)

        header_names = ["",""]
        for i in range(1,len(keypoints_names)+1):
            header_names.append("X"+str(i))
            header_names.append("Y"+str(i))
            header_names.append("Z"+str(i))    
        
        tsv_writer.writerow(header_names)
        tsv_writer.writerow("")    

        skeleton_data_frame.insert(0, "Frame", [str(i) for i in range(0, len(skeleton_data_frame))])
        skeleton_data_frame.insert(1, "Time", skeleton_data_frame["Frame"].astype(float) / float(camera_rate))

                # and finally actually write the trajectories
        for row in range(0, len(skeleton_data_frame)):
            tsv_writer.writerow(skeleton_data_frame.iloc[row].tolist())

        f = 2 


def flatten_mediapipe_data(skeleton_3d_data):
    num_frames = skeleton_3d_data.shape[0]
    num_markers = skeleton_3d_data.shape[1]

    skeleton_data_flat = skeleton_3d_data.reshape(num_frames,num_markers*3)

    return skeleton_data_flat
