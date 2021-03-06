import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path 
import socket
import pickle
import scipy.io as sio

from anthropometry_data_tables import segments, joint_connections, segment_COM_lengths, segment_COM_percentages
from openpose_skeleton_builder import openpose_indices, slice_openpose_data, build_openpose_skeleton 
from COM_calculator import calculate_segment_COM, reformat_segment_COM, calculate_total_body_COM



this_computer_name = socket.gethostname()
print(this_computer_name)

#set the path to your FreeMoCap data 
if this_computer_name == 'DESKTOP-V3D343U':
    freemocap_data_path = Path(r"I:\My Drive\HuMoN_Research_Lab\FreeMoCap_Stuff\FreeMoCap_Balance_Validation\data")
elif this_computer_name == 'DESKTOP-F5LCT4Q':
    freemocap_data_path = Path(r"C:\Users\aaron\Documents\HumonLab\Spring2022\ValidationStudy\FreeMocap_Data")
    #freemocap_data_path = Path(r'D:\freemocap2022\FreeMocap_Data')
else:
    freemocap_data_path = Path(r"C:\Users\kiley\Documents\HumonLab\SampleFMC_Data\FreeMocap_Data-20220216T173514Z-001\FreeMocap_Data")

#parameters to adjust 
sessionID = 'session_SER_1_20_22' #name of the sessionID folder
data_array_name = 'openPoseSkel_3d_smoothed.npy'
num_pose_joints = 25 #number of pose joints tracked by mediapipe 

#creating paths to the session and data
this_freemocap_session_path = freemocap_data_path / sessionID
this_freemocap_data_path = this_freemocap_session_path/'DataArrays'
openpose_data_path = this_freemocap_data_path/data_array_name
openpose_skeleton_file_path = this_freemocap_data_path/'openPose_Skelcoordinates_frame_segment_joint_XYZ.pkl'
segmentCOM_data_path = this_freemocap_data_path/'openPose_segmentedCOM_frame_joint_XYZ.npy'
totalBodyCOM_data_path = this_freemocap_data_path/'openPose_totalBodyCOM_frame_XYZ.npy'


#load the mediapipe data
openposeSkel_fr_mar_dim = np.load(openpose_data_path)

#get just the body data for mediapipe 
openpose_pose_data = slice_openpose_data(openposeSkel_fr_mar_dim, num_pose_joints)
num_frame_range = range(len(openpose_pose_data))


#load anthropometric data into a pandas dataframe
df = pd.DataFrame(list(zip(segments,joint_connections,segment_COM_lengths,segment_COM_percentages)),columns = ['Segment Name','Joint Connection','Segment COM Length','Segment COM Percentage'])
segment_conn_len_perc_dataframe = df.set_index('Segment Name')
num_segments = len(segments)

#build a mediapipe skeleton based on the segments defined in the anthropometry_data_tables.py file
skelcoordinates_frame_segment_joint_XYZ = build_openpose_skeleton(openpose_pose_data,segment_conn_len_perc_dataframe, openpose_indices, num_frame_range)

#calculate segment and total body COM data 
segment_COM_frame_dict = calculate_segment_COM(segment_conn_len_perc_dataframe, skelcoordinates_frame_segment_joint_XYZ, num_frame_range)
segment_COM_frame_imgPoint_XYZ = reformat_segment_COM(segment_COM_frame_dict,num_frame_range, num_segments)
totalBodyCOM_frame_XYZ = calculate_total_body_COM(segment_conn_len_perc_dataframe,segment_COM_frame_dict,num_frame_range)

#save out files 
open_file = open(openpose_skeleton_file_path, "wb")
pickle.dump(skelcoordinates_frame_segment_joint_XYZ, open_file)
open_file.close()

np.save(segmentCOM_data_path,segment_COM_frame_imgPoint_XYZ)
np.save(totalBodyCOM_data_path,totalBodyCOM_frame_XYZ,num_frame_range)

f = 2