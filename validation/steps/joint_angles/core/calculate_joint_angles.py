import numpy as np
import pandas as pd
from skellymodels.managers.human import Human
from skellymodels.models.trajectory import Trajectory
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass

@dataclass
class XZCoordinateReferences:
    z: tuple[str, str]
    x: tuple[str, str]

@dataclass
class FootCoordinateReferences:
    y: tuple[str, str]

coordinate_systems = {
    'right_foot': FootCoordinateReferences(
        y = ('right_heel', 'right_foot_index')
    ),

    'left_foot': FootCoordinateReferences(
        y = ('left_heel', 'left_foot_index')
    ),

    'right_shank': XZCoordinateReferences(
        z = ('right_ankle', 'right_knee'),
        x = ('left_hip', 'right_hip')
    ),

    'left_shank': XZCoordinateReferences(
        z = ('left_ankle', 'left_knee'),
        x = ('left_hip', 'right_hip')
    ),

    'right_thigh': XZCoordinateReferences(
        z = ('right_knee', 'right_hip'),
        x = ('left_hip', 'right_hip')
    ),

    'left_thigh': XZCoordinateReferences(
        z = ('left_knee', 'left_hip'),
        x = ('left_hip', 'right_hip')
    ),

    'left_hip': XZCoordinateReferences(
        z = ('hips_center', 'neck_center'),
        x = ('left_hip', 'right_hip')
    ),
    'right_hip': XZCoordinateReferences(
        z = ('hips_center', 'neck_center'),
        x = ('left_hip', 'right_hip')
    ),
}

joint_angle_setup = {
    'right_knee': ['right_thigh', 'right_shank'],
    'right_ankle': ['right_shank', 'right_foot'],
    'left_knee': ['left_thigh', 'left_shank'],
    'left_ankle': ['left_shank', 'left_foot'],
    'left_hip': ['left_hip', 'left_thigh'],
    'right_hip': ['right_hip', 'right_thigh'],
}

def get_segment_rotation(joints:Trajectory, ref:XZCoordinateReferences):
    num_frames = joints.as_array.shape[0]
    x1 = joints.as_dict[ref.x[0]]
    x2 = joints.as_dict[ref.x[1]]
    z1 = joints.as_dict[ref.z[0]]
    z2 = joints.as_dict[ref.z[1]]

    x = x2 - x1
    z = z2 - z1

    x = norm(x)
    zhat = norm(z)

    yhat = norm(np.cross(zhat,x))
    xhat = norm(np.cross(yhat, zhat))

    R_segment = np.zeros((num_frames, 3, 3))
    R_segment[:,:,0] = xhat
    R_segment[:,:,1] = yhat
    R_segment[:,:,2] = zhat

    return R_segment

def subtract_neutral(angles:np.ndarray, neutral_frames:range) -> np.ndarray:
    neutral_mean = np.mean(angles[neutral_frames], axis=0)
    return angles - neutral_mean

def norm(v, eps=1e-12):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n

def get_foot_coordinate_system(joints:Trajectory, refs:FootCoordinateReferences):
    num_frames = joints.as_array.shape[0]

    y1 = joints.as_dict[refs.y[0]]
    y2 = joints.as_dict[refs.y[1]]
    y_hat = norm((y2 - y1))
    

    # Constrain ML to ground plane:
    # First compute an x candidate from the global up vector
    up = np.array([0.0, 0.0, 1.0]) 
    x_raw = np.cross(y_hat, up)
    x_hat = norm(x_raw)

    z_hat = norm(np.cross(x_hat, y_hat))
    x_hat = norm(np.cross(y_hat, z_hat))

    R_foot = np.empty((num_frames, 3, 3))
    R_foot[:, :, 0] = x_hat  # ML (ground-plane constrained)
    R_foot[:, :, 1] = y_hat  # longitudinal
    R_foot[:, :, 2] = z_hat  # vertical of the foot

    return R_foot

def calculate_cardan_angles(R_proximal:np.ndarray,
                            R_distal: np.ndarray):
    
    num_frames = R_proximal.shape[0]
    R_rel = np.empty_like(R_proximal)
    for i in range(num_frames):
        R_rel[i] = R_proximal[i].T @ R_distal[i]
    
    r = R.from_matrix(R_rel)
    cardan_angles = r.as_euler('XYZ', degrees=True) 

    return cardan_angles

def calculate_angle(proximal_orientation: np.ndarray,
                    distal_orientation: np.ndarray):
    angle = calculate_cardan_angles(R_proximal=proximal_orientation,
                                    R_distal=distal_orientation)
    return angle

def calculate_joint_angles(human: Human,
                           neutral_stance_frames: range|None,
                           use_nonrigid = False):
    if use_nonrigid:
        joints = human.body.xyz
    else:
        joints = human.body.rigid_xyz

    segment_orientations = {}
    for segment_name, refs in coordinate_systems.items():
        if isinstance(refs, FootCoordinateReferences):
            segment_orientations[segment_name] = get_foot_coordinate_system(joints,refs)
        else:
            segment_orientations[segment_name] = get_segment_rotation(joints, refs)

    joint_angles = {}
    for angle_name, segments in joint_angle_setup.items():
        proximal_orientation = segment_orientations[segments[0]]
        distal_orientation = segment_orientations[segments[1]]
        angle = calculate_angle(proximal_orientation, distal_orientation)
        joint_angles[angle_name] = subtract_neutral(angle, neutral_stance_frames) if neutral_stance_frames is not None else angle
    
    
    df = pd.DataFrame({
        # Right Knee - XYZ sequence
        "knee_flex_ext_r":    joint_angles['right_knee'][:, 0],  # X - Flexion/Extension
        "knee_ab_ad_r":       joint_angles['right_knee'][:, 1],  # Y - Abduction/Adduction
        "knee_int_ext_rot_r": joint_angles['right_knee'][:, 2],  # Z - Internal/External Rotation
        
        # Right Ankle - XYZ sequence  
        "ankle_dorsi_plantar_r": joint_angles['right_ankle'][:, 0],  # X - Dorsi/Plantarflexion
        "ankle_inv_ev_r":        joint_angles['right_ankle'][:, 1],  # Y - Inversion/Eversion
        "ankle_int_ext_rot_r":   joint_angles['right_ankle'][:, 2],  # Z - Internal/External Rotation
        
        # Left Knee - XYZ sequence
        "knee_flex_ext_l":    joint_angles['left_knee'][:, 0],  # X - Flexion/Extension
        "knee_ab_ad_l":       joint_angles['left_knee'][:, 1],  # Y - Abduction/Adduction
        "knee_int_ext_rot_l": joint_angles['left_knee'][:, 2],  # Z - Internal/External Rotation
        
        # Left Ankle - XYZ sequence
        "ankle_dorsi_plantar_l": joint_angles['left_ankle'][:, 0],  # X - Dorsi/Plantarflexion
        "ankle_inv_ev_l":        joint_angles['left_ankle'][:, 1],  # Y - Inversion/Eversion
        "ankle_int_ext_rot_l":   joint_angles['left_ankle'][:, 2],  # Z - Internal/External Rotation

        # Left Hip - XYZ sequence
        "hip_flex_ext_l":    joint_angles['left_hip'][:, 0],  # X - Flexion/Extension
        "hip_ab_ad_l":       joint_angles['left_hip'][:, 1],
        "hip_int_ext_rot_l": joint_angles['left_hip'][:, 2],  # Z - Internal/External Rotation

        # Right Hip - XYZ sequence
        "hip_flex_ext_r":    joint_angles['right_hip'][:, 0],  # X - Flexion/Extension
        "hip_ab_ad_r":       joint_angles['right_hip'][:, 1],
        "hip_int_ext_rot_r": joint_angles['right_hip'][:, 2],  # Z - Internal/External Rotation
    })

    df.index.name = "frame"

    return df



if __name__ == "__main__":
    path_to_recording = Path(r'D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1')

    freemocap_tracker = 'mediapipe_dlc'

    trackers= ['qualisys', freemocap_tracker]

    for tracker in trackers: #NOTE - for non prosthetic data don't loop (because of non rigid flag)
        path_to_data = path_to_recording/'validation'/tracker

        human:Human = Human.from_data(path_to_data)
        angles = calculate_joint_angles(
            human,
            range(90,210),
            use_nonrigid=True
        )
        angles.to_csv(path_to_data/f'{tracker}_joint_angles_adjust.csv', index = True)

