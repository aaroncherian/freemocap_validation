from pathlib import Path
from angles.ankle_angles import calculate_joint_angles
from skellymodels.managers.human import Human
from gait.stepfinder import detect_gait_events, save_gait_events_to_csv
from gait.gait_cycler import get_angle_strides
from gait.cycle_trajectories import get_joint_strides

path_to_recording = Path(r'D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_12_36_TF01_flexion_pos_5_6_trial_1')
freemocap_tracker = 'mediapipe_dlc'
neutral_stance_frames = range(180,250)
#for toe neg 6: range(120,215)
#for toe neg 3: range(105, 170)
#for toe neutral: range(110, 180)
#for toe pos 3: range(90, 160)
#for toe pos 6: range(110, 200)
#--- 
#for flexion neg 5-6: range(220,320)
#for flexion neg 2.8: range(170,290)
#for flexion neutral: range(120,200)
#for flexion pos 2.8: range(90,210)
#for flexion pos 5-6: range(180,250)

active_frame_range =  range(600,2350) # frames to search for steps in
#for toe neg 6: range(500,2250)
#for toe neg 3: range(500,2250)
#for toe neutral: range(500,2250)
#for toe pos 3: range(500,2250)
#for toe pos 6: range(500,2250)
#---
#for flexion neg 5-6: range(750,2500)
#for flexion neg 2.8: range(600,2350)
#for flexion neutral: range(450,2200)
#for flexion pos 2.8: range(600,2350)
#for flexion pos 5-6: range(600,2350)

trackers= ['qualisys', freemocap_tracker]
for tracker in trackers: #NOTE - for non prosthetic data don't loop (because of non rigid flag)
    path_to_data = path_to_recording/'validation'/tracker

    human:Human = Human.from_data(path_to_data)
    angles = calculate_joint_angles(
        human,
        neutral_stance_frames,
        use_nonrigid=True
    )
    angles.to_csv(path_to_data/f'{tracker}_joint_angles.csv', index = True)

## get steps from qualisys data
path_to_data = path_to_recording/'validation'/'qualisys'
human:Human = Human.from_data(path_to_data)

gait_events = detect_gait_events(human, sampling_rate=30.0)
save_gait_events_to_csv(gait_events, fs=30.0, out_path=path_to_data/'gait_events.csv')

get_angle_strides(path_to_recording, active_frame_range, tracker)
get_joint_strides(path_to_recording, active_frame_range, tracker)