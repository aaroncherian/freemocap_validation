from dataclasses import dataclass
import numpy as np
from skellymodels.managers.human import Human

@dataclass
class FootKinematics:
    left_heel_pos: np.ndarray
    right_heel_pos: np.ndarray
    left_toe_pos: np.ndarray
    right_toe_pos:np.ndarray
    left_heel_vel: np.ndarray
    right_heel_vel: np.ndarray
    left_toe_vel: np.ndarray
    right_toe_vel: np.ndarray

def parse_human(human:Human):
    left_heel = human.body.xyz.as_dict['left_heel']
    right_heel = human.body.xyz.as_dict['right_heel']
    
    left_toe = human.body.xyz.as_dict['left_foot_index']
    right_toe = human.body.xyz.as_dict['right_foot_index']

    return left_heel, right_heel, left_toe, right_toe


def get_velocity(positions:np.ndarray, sampling_rate:float):
    dt = 1.0 / sampling_rate
    velocities = np.gradient(positions, dt, axis=0)

    return velocities


def get_foot_kinematics(human:Human, sampling_rate:float) -> FootKinematics:
    left_heel_pos, right_heel_pos, left_toe_pos, right_toe_pos = parse_human(human)

    left_heel_vel = get_velocity(left_heel_pos, sampling_rate)
    right_heel_vel = get_velocity(right_heel_pos, sampling_rate)
    left_toe_vel = get_velocity(left_toe_pos, sampling_rate)
    right_toe_vel = get_velocity(right_toe_pos, sampling_rate)

    foot_kinematics = FootKinematics(
        left_heel_pos=left_heel_pos,
        right_heel_pos=right_heel_pos,
        left_toe_pos=left_toe_pos,
        right_toe_pos=right_toe_pos,
        left_heel_vel=left_heel_vel,
        right_heel_vel=right_heel_vel,
        left_toe_vel=left_toe_vel,
        right_toe_vel=right_toe_vel,
    )
    return foot_kinematics


