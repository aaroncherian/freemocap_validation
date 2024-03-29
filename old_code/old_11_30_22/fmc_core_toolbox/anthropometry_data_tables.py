#values for segment weight and segment mass percentages taken from Winter anthropometry tables
#https://imgur.com/a/aD74j
#Winter, D.A. (2005) Biomechanics and Motor Control of Human Movement. 3rd Edition, John Wiley & Sons, Inc., Hoboken.

import pandas as pd

def build_anthropometric_dataframe(segments:list,joint_connections:list,segment_COM_lengths:list,segment_COM_percentages:list) -> pd.DataFrame:
    #load anthropometric data into a pandas dataframe
    df = pd.DataFrame(list(zip(segments,joint_connections,segment_COM_lengths,segment_COM_percentages)),columns = ['Segment_Name','Joint_Connection','Segment_COM_Length','Segment_COM_Percentage'])
    segment_conn_len_perc_dataframe = df.set_index('Segment_Name')
    return segment_conn_len_perc_dataframe

segments = [
'head',
'trunk',
'right_upper_arm',
'left_upper_arm',
'right_forearm',
'left_forearm',
'right_hand',
'left_hand',
'right_thigh',
'left_thigh',
'right_shin',
'left_shin',
'right_foot',
'left_foot'
]

joint_connections = [
['left_ear','right_ear'],
['mid_chest_marker', 'mid_hip_marker'], 
['right_shoulder','right_elbow'],
['left_shoulder','left_elbow'],
['right_elbow', 'right_wrist'],
['left_elbow', 'left_wrist'],
['right_wrist', 'right_hand_marker'], 
['left_wrist', 'left_hand_marker'],
['right_hip', 'right_knee'],
['left_hip', 'left_knee'],
['right_knee', 'right_ankle'],
['left_knee', 'left_ankle'],
['right_back_of_foot_marker', 'right_foot_index'], 
['left_back_of_foot_marker', 'left_foot_index']
]

segment_COM_lengths = [
.5,
.5,
.436,
.436,
.430,
.430,
.506, 
.506, 
.433,
.433,
.433,
.433,
.5, 
.5  
]

segment_COM_percentages = [
.081,
.497,
.028,
.028,
.016,
.016,
.006,
.006,
.1,
.1,
.0465,
.0465,
.0145,
.0145
]
