from typing import List, Dict
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from validation.components.qualisys import QUALISYS_MARKERS, QUALISYS_START_TIME
from validation.steps.temporal_alignment.core.markersets.validation_study_joint_center_weights import joint_center_weights

class QualisysMarkerData:
    def __init__(self, 
                 marker_dataframe: pd.DataFrame,
                 unix_start_time:float):
        self.data = marker_dataframe
        self.unix_start_time = unix_start_time

    def _extract_marker_data(self) -> pd.DataFrame:
        """Extract only marker data columns."""

        columns_of_interest = self.data.columns[
            ~self.data.columns.str.contains(r'^(?:Frame|Time|unix_timestamps|Unnamed)', regex=True)
        ]
        return self.data[columns_of_interest]
    
    @property
    def marker_names(self) -> List[str]:
        marker_columns = self._extract_marker_data().columns
        return list(dict.fromkeys(col.split()[0] for col in marker_columns))
    
    @property
    def marker_array(self) -> np.ndarray:
        marker_data = self._extract_marker_data()
        num_frames = len(marker_data)
        num_markers = int(len(marker_data.columns) / 3)
        return marker_data.to_numpy().reshape(num_frames, num_markers, 3)
    
    @property 
    def time_and_frame_columns(self) -> pd.DataFrame:
        return self.data[['Time', 'Frame']]

    
    def as_dataframe_with_unix_timestamps(self, lag_seconds: float = 0) -> pd.DataFrame:
        """
        Returns a DataFrame with marker data and corresponding Unix timestamps.
        
        Parameters:
            lag_seconds (float): Optional time offset to adjust timestamps.
        
        Returns:
            pd.DataFrame: DataFrame containing frame, time, markers, and Unix timestamps.
        """
        df = self.time_and_frame_columns.copy()
        
        # Extract marker data and add to the DataFrame
        marker_data = self._extract_marker_data()
        df = pd.concat([df, marker_data], axis=1)

        # Compute Unix timestamps
        df['unix_timestamps'] = df['Time'] + self.unix_start_time + lag_seconds

        return df


class QualisysJointCenterData:

    def __init__(self, marker_data_holder:QualisysMarkerData, weights:Dict):
        self.marker_data = marker_data_holder
        self.weights = weights
        self.joint_names = list(weights.keys())
        self.joint_centers = self._calculate_joint_centers(
            marker_data_array=marker_data_holder.marker_array,
            marker_names=marker_data_holder.marker_names,
            joint_center_weights=weights
        )

    def _calculate_joint_centers(self, marker_data_array:np.ndarray, marker_names:List, joint_center_weights:Dict):
        """
        Optimized calculation of joint centers for Qualisys data with 3D weights.

        Parameters:
            marker_array (np.ndarray): Shape (num_frames, num_markers, 3), 3D marker data.
            joint_center_weights (dict): Weights for each joint as {joint_name: {marker_name: [weight_x, weight_y, weight_z]}}.
            marker_names (list): List of marker names corresponding to marker_array.

        Result:
            np.ndarray: Joint centers with shape (num_frames, num_joints, 3).
        """
        print('Calculating joint centers...')
        num_frames, num_markers, _ = marker_data_array.shape
        num_joints = len(joint_center_weights)

        marker_to_index = {marker: i for i, marker in enumerate(marker_names)}
        joints = list(joint_center_weights.keys())

        weights_matrix = np.zeros((num_joints, num_markers, 3))
        for j_idx, (joint, markers_weights) in enumerate(joint_center_weights.items()):
            for marker, weight in markers_weights.items():
                marker_idx = marker_to_index[marker]
                weights_matrix[j_idx, marker_idx, :] = weight  # Assign 3D weight

        joint_centers = np.einsum('fmd,jmd->fjd', marker_data_array, weights_matrix)

        if 'right_hip' in joint_center_weights:
            right_hip_center = self.calculate_hip_center(marker_names, 'right_hip')
            joint_centers[:, joints.index('right_hip'), :] = right_hip_center
        if 'left_hip' in joint_center_weights:
            left_hip_center = self.calculate_hip_center(marker_names, 'left_hip')
            joint_centers[:, joints.index('left_hip'), :] = left_hip_center

        return joint_centers
    
    def calculate_hip_center(self, marker_names, hip_name:str):
        def get_unit_vector(vector: np.ndarray) -> np.ndarray:
            return vector / np.linalg.norm(vector, axis = -1, keepdims = True)
        
        marker_data = self.marker_data.marker_array
        rasis = marker_data[:,marker_names.index('RASIS'),:]
        lasis = marker_data[:,marker_names.index('LASIS'),:]
        rpsis = marker_data[:,marker_names.index('RPSIS'),:]
        lpsis = marker_data[:,marker_names.index('LPSIS'),:]

        asis_midpoint = (rasis + lasis) / 2 #origin
        psis_midpoint = (rpsis + lpsis) / 2 

        right = rasis - lasis
        xhat = get_unit_vector(right)
        forward = get_unit_vector(asis_midpoint - psis_midpoint)
        
        zhat = get_unit_vector(np.cross(xhat,forward))
        flip = (zhat[...,2] < 0)
        zhat[flip] *= -1.0

        yhat = get_unit_vector(np.cross(zhat,xhat))
        zhat = get_unit_vector(np.cross(xhat,yhat))

        R = np.stack([xhat,yhat,zhat],axis=-1) 
        asis_distance = np.linalg.norm(rasis - lasis, axis=-1, keepdims=True)
        pelvic_depth = np.linalg.norm(asis_midpoint - psis_midpoint, axis=-1, keepdims=True)

        if hip_name == 'left_hip':
            ML = -.36* asis_distance
        elif hip_name == 'right_hip':
            ML = .36* asis_distance
        AP = -.19* asis_distance + .5*pelvic_depth - float(8)
        AXIAL = -.3*asis_distance

        offsets = np.concatenate([ML, AP, AXIAL], axis=1)[..., None] 

        hip_center = asis_midpoint + (R @ offsets)[..., 0]

        return hip_center 


    def as_dataframe(self) -> pd.DataFrame:
        df = self.marker_data.time_and_frame_columns.copy()

        for joint_idx, joint_name in enumerate(self.joint_names):
            for axis_idx, axis in enumerate(['x', 'y', 'z']):
                col_name = f"{joint_name} {axis}"
                df[col_name] = self.joint_centers[:, joint_idx, axis_idx]

        return df
    
    def as_dataframe_with_unix_timestamps(self, lag_seconds: float = 0) -> pd.DataFrame:
        df = self.as_dataframe()
        df['unix_timestamps'] = df['Time'] + self.marker_data.unix_start_time + lag_seconds
        return df


path_to_recording = Path(r"D:\2025_07_31_JSM_pilot\freemocap\2025-07-31_16-52-16_GMT-4_jsm_treadmill_2")
qualisys_markers = QUALISYS_MARKERS.load(path_to_recording)
start_time = QUALISYS_START_TIME.load(path_to_recording)

qualisys_marker_data_holder = QualisysMarkerData(
    marker_dataframe=qualisys_markers,
    unix_start_time=start_time
)

qualisys_joint_center_data = QualisysJointCenterData(
    marker_data_holder=qualisys_marker_data_holder,
    weights=joint_center_weights
)

f = 2