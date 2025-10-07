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
    n = np.linalg.norm(v, axis=-1, keepdims=True)
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

    hip_ml = norm(right_hip - left_hip)
    shank_z = norm(right_knee - right_ankle)
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
    
    # Constrain ML to ground plane:
    # First compute an x candidate from the global up vector
    up = np.tile(np.array([0.0, 0.0, 1.0]).reshape(1, 3), (num_frames, 1))
    x_raw = np.cross(y_hat, up)

    # Handle near-degenerate frames (when ŷ ≈ ±up, very rare for feet but safe to guard)
    small = np.linalg.norm(x_raw, axis=1, keepdims=True) < 1e-9
    if np.any(small):
        # Use ankle→toe as a guide: remove its component along ŷ, then cross with up again
        b = toe - ankle
        b_proj = b - np.sum(b * y_hat, axis=1, keepdims=True) * y_hat
        # If still degenerate, nudge with a fixed lateral guess in the ground plane
        b_proj[small[:, 0] & (np.linalg.norm(b_proj, axis=1) < 1e-9)] = np.array([1.0, 0.0, 0.0])
        # Recompute x_raw with the cleaned guide
        x_raw[small[:, 0]] = np.cross(up[small[:, 0]], norm(b_proj[small[:, 0]]))

    x_hat = norm(x_raw)

    # ẑ from right-handedness; will be close to 'up' but adapted to the foot (orthonormal)
    z_hat = norm(np.cross(x_hat, y_hat))

    # Re-orthogonalize x̂ to kill any accumulated numeric drift
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
    cardan_angles = r.as_euler('ZXY', degrees=True) # Cardan sequence: Z (abduction/adduction), X (dorsiflexion/plantarflexion), Y (inversion/eversio)

    return cardan_angles

def get_relative_rotation_matrix(R_proximal:np.ndarray,
                                 R_distal:np.ndarray) -> np.ndarray:
    num_frames = R_proximal.shape[0]
    R_rel = np.empty_like(R_proximal)
    for i in range(num_frames):
        R_rel[i] = R_proximal[i].T @ R_distal[i]

    return R_rel

def calculate_alpha(e1:np.ndarray, e2: np.ndarray, e1_ref:np.ndarray) -> float:
    y = np.dot(e1, np.cross(e1_ref,e2))
    x = np.dot(e2, e1_ref)
    return np.degrees(np.arctan2(y, x))

def calculate_beta(e1: np.ndarray, e2:np.ndarray, e3:np.ndarray) -> float:
    y = np.dot(np.cross(e1,e3), e2)
    x = np.dot(e1,e3)
    return np.degrees(np.arctan2(y,x))

def calculate_gamma( e2:np.ndarray, e3: np.ndarray, e3_ref:np.ndarray) -> float:
    x = np.dot(e3_ref, e2)
    y = np.dot(e3, np.cross(e3_ref,e2))
    return np.degrees(np.arctan2(y,x))


def calculate_ankle_angles(joints: Trajectory) -> np.ndarray:
    R_shank = get_shank_coordinate_system(joints)
    R_foot = get_foot_coordinate_system(joints)
    
    num_frames = R_shank.shape[0]
    alpha = np.zeros(num_frames)
    beta = np.zeros(num_frames)
    gamma = np.zeros(num_frames)
    
    for i in range(num_frames):
        e1 = R_shank[i,:,0]
        e3 = R_foot[i,:,1]
        e2 = norm(np.cross(e3, e1))

        e1_ref = R_shank[i,:,1]
        e3_ref = R_foot[i,:,0]

        alpha[i] = calculate_alpha(e1, e2, e1_ref)
        beta[i] = calculate_beta(e1, e2, e3)
        gamma[i] = calculate_gamma(e2, e3, e3_ref)
    
    return np.column_stack([alpha, beta, gamma])

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


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def plot_coordinate_systems_debug(human, frame_range=None):
    """
    Plot the right leg with shank and foot coordinate systems for debugging
    """
    # Get joint positions
    joints = human  # or use .xyz for non-rigid
    
    # Get the coordinate systems
    R_shank = get_shank_coordinate_system(joints)
    R_foot = get_foot_coordinate_system(joints)
    
    # Get joint positions for plotting
    right_knee = joints.as_dict['right_knee']
    right_ankle = joints.as_dict['right_ankle']
    right_toe = joints.as_dict['right_foot_index']
    right_heel = joints.as_dict['right_heel']
    
    # If no frame range specified, use all frames
    if frame_range is None:
        frame_range = range(0, len(right_knee), 10)  # Every 10th frame for clarity
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Plot multiple frames in subplots
    n_frames_to_plot = min(6, len(frame_range))
    frames_to_plot = np.linspace(frame_range.start, frame_range.stop-1, n_frames_to_plot, dtype=int)
    
    for idx, frame in enumerate(frames_to_plot):
        ax = fig.add_subplot(2, 3, idx+1, projection='3d')
        
        # Plot the leg segments
        # Shank
        ax.plot([right_knee[frame, 0], right_ankle[frame, 0]], 
                [right_knee[frame, 1], right_ankle[frame, 1]], 
                [right_knee[frame, 2], right_ankle[frame, 2]], 
                'k-', linewidth=2, label='Shank')
        
        # Foot
        ax.plot([right_ankle[frame, 0], right_toe[frame, 0]], 
                [right_ankle[frame, 1], right_toe[frame, 1]], 
                [right_ankle[frame, 2], right_toe[frame, 2]], 
                'g-', linewidth=2, label='Foot (to toe)')
        
        ax.plot([right_ankle[frame, 0], right_heel[frame, 0]], 
                [right_ankle[frame, 1], right_heel[frame, 1]], 
                [right_ankle[frame, 2], right_heel[frame, 2]], 
                'g--', linewidth=1, alpha=0.5, label='Foot (to heel)')
        
        # Plot coordinate systems
        scale = 50  # Scale for visualization of axes
        
        # Shank coordinate system at knee
        shank_origin = right_knee[frame]
        shank_x = R_shank[frame, :, 0] * scale
        shank_y = R_shank[frame, :, 1] * scale
        shank_z = R_shank[frame, :, 2] * scale
        
        ax.quiver(shank_origin[0], shank_origin[1], shank_origin[2],
                  shank_x[0], shank_x[1], shank_x[2],
                  color='r', arrow_length_ratio=0.1, linewidth=2, label='Shank X')
        ax.quiver(shank_origin[0], shank_origin[1], shank_origin[2],
                  shank_y[0], shank_y[1], shank_y[2],
                  color='g', arrow_length_ratio=0.1, linewidth=2, label='Shank Y')
        ax.quiver(shank_origin[0], shank_origin[1], shank_origin[2],
                  shank_z[0], shank_z[1], shank_z[2],
                  color='b', arrow_length_ratio=0.1, linewidth=2, label='Shank Z')
        
        # Foot coordinate system at ankle
        foot_origin = right_ankle[frame]
        foot_x = R_foot[frame, :, 0] * scale
        foot_y = R_foot[frame, :, 1] * scale
        foot_z = R_foot[frame, :, 2] * scale
        
        ax.quiver(foot_origin[0], foot_origin[1], foot_origin[2],
                  foot_x[0], foot_x[1], foot_x[2],
                  color='r', arrow_length_ratio=0.1, linewidth=2, linestyle='--', label='Foot X')
        ax.quiver(foot_origin[0], foot_origin[1], foot_origin[2],
                  foot_y[0], foot_y[1], foot_y[2],
                  color='g', arrow_length_ratio=0.1, linewidth=2, linestyle='--', label='Foot Y')
        ax.quiver(foot_origin[0], foot_origin[1], foot_origin[2],
                  foot_z[0], foot_z[1], foot_z[2],
                  color='b', arrow_length_ratio=0.1, linewidth=2, linestyle='--', label='Foot Z')
        
        # Plot joint markers
        ax.scatter(*right_knee[frame], color='red', s=50, label='Knee')
        ax.scatter(*right_ankle[frame], color='blue', s=50, label='Ankle')
        ax.scatter(*right_toe[frame], color='green', s=50, label='Toe')
        ax.scatter(*right_heel[frame], color='orange', s=50, label='Heel')
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Frame {frame}')
        
        # Set equal aspect ratio
        max_range = np.array([
            right_knee[frame] - right_toe[frame],
            right_knee[frame] - right_heel[frame]
        ]).max() * 1.5
        
        mid_x = (right_knee[frame, 0] + right_toe[frame, 0]) / 2
        mid_y = (right_knee[frame, 1] + right_toe[frame, 1]) / 2
        mid_z = (right_knee[frame, 2] + right_toe[frame, 2]) / 2
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        if idx == 0:
            ax.legend(fontsize=8, loc='best')
    
    plt.suptitle('Shank and Foot Coordinate Systems Debug View', fontsize=14)
    plt.tight_layout()
    return fig

def plot_coordinate_alignment(human, frame_range=None):
    """
    Plot to check alignment between coordinate systems
    """
    joints = human
    
    R_shank = get_shank_coordinate_system(joints)
    R_foot = get_foot_coordinate_system(joints)
    
    if frame_range is None:
        frame_range = range(0, R_shank.shape[0])
    
    # Calculate dot products to check orthogonality
    frames = list(frame_range)
    
    # For each frame, check key alignments
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    # Check if shank Z and foot Y are perpendicular (should be ~0 in neutral)
    dot_shankZ_footY = [np.dot(R_shank[i,:,2], R_foot[i,:,1]) for i in frames]
    axes[0,0].plot(frames, dot_shankZ_footY)
    axes[0,0].set_title('Shank Z · Foot Y (should vary around 0)')
    axes[0,0].set_ylabel('Dot product')
    axes[0,0].grid(True)
    axes[0,0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Check if shank X and foot X alignment
    dot_shankX_footX = [np.dot(R_shank[i,:,0], R_foot[i,:,0]) for i in frames]
    axes[0,1].plot(frames, dot_shankX_footX)
    axes[0,1].set_title('Shank X · Foot X (alignment check)')
    axes[0,1].set_ylabel('Dot product')
    axes[0,1].grid(True)
    axes[0,1].axhline(y=1, color='r', linestyle='--', alpha=0.5)
    
    # Calculate the floating axis (e2) and check it's perpendicular to e1 and e3
    e1_e2_dot = []
    e2_e3_dot = []
    e2_norm = []
    
    for i in frames:
        e1 = R_shank[i,:,0]  # Shank X (flexion axis)
        e3 = R_foot[i,:,1]   # Foot Y (internal/external rotation axis)
        e2 = np.cross(e3, e1)
        e2_normalized = e2 / np.linalg.norm(e2)
        
        e1_e2_dot.append(np.dot(e1, e2_normalized))
        e2_e3_dot.append(np.dot(e2_normalized, e3))
        e2_norm.append(np.linalg.norm(e2))
    
    axes[1,0].plot(frames, e1_e2_dot)
    axes[1,0].set_title('e1 · e2 (should be 0)')
    axes[1,0].set_ylabel('Dot product')
    axes[1,0].grid(True)
    axes[1,0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    axes[1,1].plot(frames, e2_e3_dot)
    axes[1,1].set_title('e2 · e3 (should be 0)')
    axes[1,1].set_ylabel('Dot product')
    axes[1,1].grid(True)
    axes[1,1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    axes[2,0].plot(frames, e2_norm)
    axes[2,0].set_title('||e2|| before normalization')
    axes[2,0].set_ylabel('Magnitude')
    axes[2,0].grid(True)
    
    # Plot angle between shank Z and foot Y (this is roughly the flexion angle)
    angle_shankZ_footY = [np.degrees(np.arccos(np.clip(np.dot(R_shank[i,:,2], R_foot[i,:,1]), -1, 1))) for i in frames]
    axes[2,1].plot(frames, angle_shankZ_footY)
    axes[2,1].set_title('Angle between Shank Z and Foot Y')
    axes[2,1].set_ylabel('Degrees')
    axes[2,1].set_xlabel('Frame')
    axes[2,1].grid(True)
    
    for ax in axes.flat:
        ax.set_xlabel('Frame')
    
    plt.suptitle('Coordinate System Alignment Checks', fontsize=14)
    plt.tight_layout()
    return fig



if __name__ == "__main__":
    path_to_recording = Path(r'D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_09_05_TF01_flexion_pos_2_8_trial_1')

    freemocap_tracker = 'mediapipe_dlc'

    trackers= ['qualisys', freemocap_tracker]

    for tracker in trackers: #NOTE - for non prosthetic data don't loop (because of non rigid flag)
        path_to_data = path_to_recording/'validation'/tracker

        human:Human = Human.from_data(path_to_data)
        # angles = calculate_joint_angles(
        #     human,
        #     range(90,210),
        #     use_nonrigid=True
        # )

        fig1 = plot_coordinate_systems_debug(human.body.xyz, frame_range=range(90, 210, 50))
        plt.show()

        # # Check mathematical alignment
        # fig2 = plot_coordinate_alignment(human.body.xyz, frame_range=range(90, 210))
        # plt.show()
        # angles.to_csv(path_to_data/f'{tracker}_joint_angles.csv', index = True)


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
    
