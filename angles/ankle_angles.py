import numpy as np
import pandas as pd
from skellymodels.managers.human import Human
from skellymodels.models.trajectory import Trajectory
from pathlib import Path
from scipy.spatial.transform import Rotation as R

def subtract_neutral(angles:np.ndarray, neutral_frames:range) -> np.ndarray:
    neutral_mean = np.mean(angles[neutral_frames], axis=0)
    return angles - neutral_mean

def norm(v, eps=1e-12):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return v / n

def get_thigh_coordinate_system(joints:Trajectory):
    num_frames = joints.as_array.shape[0]
    left_hip = joints.as_dict['left_hip']
    right_hip = joints.as_dict['right_hip']
    knee = joints.as_dict['right_knee']

    hip_ml = norm(left_hip - right_hip)
    zhat = norm(knee - right_hip)
    yhat = norm(np.cross(zhat, hip_ml))
    xhat = norm(np.cross(yhat, zhat))

    R_thigh = np.zeros((num_frames, 3, 3))
    R_thigh[:,:,0] = xhat
    R_thigh[:,:,1] = yhat
    R_thigh[:,:,2] = zhat

    return R_thigh

def get_shank_coordinate_system(joints:Trajectory):
    num_frames = joints.as_array.shape[0]
    left_hip = joints.as_dict['left_hip']
    right_hip = joints.as_dict['right_hip']
    right_knee = joints.as_dict['right_knee']
    right_ankle = joints.as_dict['right_ankle']

    hip_ml = norm(left_hip - right_hip)
    shank_z = norm(right_ankle - right_knee)
    shank_x = hip_ml
    shank_y = norm(np.cross(shank_z, shank_x))
    shank_x = norm(np.cross(shank_y, shank_z))

    R_shank = np.zeros((num_frames, 3, 3))
    for i in range(num_frames):
        R_shank[i] = np.column_stack([shank_x[i], shank_y[i], shank_z[i]])

    return R_shank

def get_foot_coordinate_system(joints:Trajectory):
    num_frames = joints.as_array.shape[0]
    ankle = joints.as_dict['right_ankle']
    toe = joints.as_dict['right_foot_index']
    heel = joints.as_dict['right_heel']
    
    y = toe - heel
    y_hat = norm(y)

    # Gram-Schmidt to get orthogonal x axis
    b = toe - ankle
    by = np.sum(b * y_hat, axis=1, keepdims=True)
    proj = by * y_hat
    x_raw = b - proj
    small = np.linalg.norm(x_raw, axis=1, keepdims=True) < 1e-9
    if np.any(small):
        g = np.tile(np.array([[1.0, 0.0, 0.0]]), (num_frames,1))
        g_proj = g - np.sum(g * y_hat, axis=1, keepdims=True) * y_hat
        x_raw[small[:,0]] = g_proj[small[:,0]]
    
    x_hat = norm(x_raw)

    z_hat = norm(np.cross(x_hat, y_hat))
    x_hat = norm(np.cross(y_hat, z_hat))

    R_foot = np.empty((num_frames, 3, 3))
    R_foot[:, :, 0] = x_hat
    R_foot[:, :, 1] = y_hat
    R_foot[:, :, 2] = z_hat

    return R_foot

def calculate_cardan_angles(R_proximal:np.ndarray,
                            R_distal: np.ndarray):
    
    num_frames = R_proximal.shape[0]
    R_rel = np.empty_like(R_proximal)
    for i in range(num_frames):
        R_rel[i] = R_proximal[i].T @ R_distal[i]
    
    r = R.from_matrix(R_rel)
    cardan_angles = r.as_euler('ZXY', degrees=True) # Cardan sequence: Z (abduction/adduction), X (dorsiflexion/plantarflexion), Y (inversion/eversio)

    return cardan_angles

def calculate_ankle_angles(joints:Trajectory) -> np.ndarray:

    R_shank = get_shank_coordinate_system(joints)
    R_foot = get_foot_coordinate_system(joints)

    ankle_angles = calculate_cardan_angles(
        R_proximal=R_shank,
        R_distal=R_foot
    )

    return ankle_angles

def calculate_knee_angles(joints:Trajectory)-> np.ndarray:

    R_thigh = get_thigh_coordinate_system(joints)
    R_shank = get_shank_coordinate_system(joints)

    knee_angles = calculate_cardan_angles(
        R_proximal=R_thigh,
        R_distal=R_shank
    )

    return knee_angles

def calculate_joint_angles(human: Human,
                           neutral_stance_frames: range|None,
                           use_nonrigid = False):
    if use_nonrigid:
        joints = human.body.xyz
    else:
        joints = human.body.rigid_xyz

    knee_angles = calculate_knee_angles(joints)
    ankle_angles = calculate_ankle_angles(joints)

    
    if neutral_stance_frames is not None:
        knee_angles  = subtract_neutral(knee_angles, neutral_stance_frames)
        ankle_angles = subtract_neutral(ankle_angles, neutral_stance_frames)
    
    df = pd.DataFrame({
        # Knee (right)
        "knee_ab_ad_r":   knee_angles[:, 0],  # Z
        "knee_flex_ext_r":knee_angles[:, 1],  # X
        "knee_inv_ev_r":  knee_angles[:, 2],  # Y

        # Ankle (right)
        "ankle_ab_ad_r":      ankle_angles[:, 0],  # Z
        "ankle_dorsi_plantar_r": ankle_angles[:, 1],  # X (your usual "flexion" index=1)
        "ankle_inv_ev_r":     ankle_angles[:, 2],  # Y
    })

    df.index.name = "frame"

    return df
    f = 2


if __name__ == "__main__":
    path_to_recording = Path(r'D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_09_05_TF01_flexion_pos_2_8_trial_1')

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
        angles.to_csv(path_to_data/f'{tracker}_joint_angles.csv', index = True)


    #for flexion neutral: 120-200
    #for flexion neg 5-6: 220-320
    #for flexion pos 2.8: 90-210
    # ankle_angles = calculate_ankle_angles(human, neutral_stance_frames=range(120,200), use_nonrigid=True)
    # knee_angles = calculate_knee_angles(human, neutral_stance_frames=range(120,200), use_nonrigid=True)

    # import matplotlib.pyplot as plt
    # fs = 30.0
    # time = np.arange(ankle_angles.shape[0])/fs
    # fig, axes = plt.subplots(2,1, figsize=(8,6), sharex=True)
    # axes[0].plot(time, knee_angles[:,1], label="Knee flexion/extension")
    # axes[0].set_ylabel("Angle (deg)")   
    # axes[0].legend(loc="upper right")
    # axes[0].grid()
    # axes[1].plot(time, ankle_angles[:,1], label="Ankle dorsiflexion/plantarflexion", color='orange')
    # axes[1].set_ylabel("Angle (deg)")
    # axes[1].set_xlabel("Time (s)")
    # axes[1].legend(loc="upper right")
    # axes[1].grid()
    # plt.tight_layout()
    # plt.show()
    # f = 2
    
