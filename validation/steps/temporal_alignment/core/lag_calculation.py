import numpy as np
from dataclasses import dataclass

@dataclass
class LagCalculatorComponent:
    joint_center_array: np.ndarray
    list_of_joint_center_names: list

    def __post_init__(self):
        if self.joint_center_array.shape[1] != len(self.list_of_joint_center_names):
            raise ValueError(f"Number of joint centers: {self.joint_center_array.shape} must match the number of joint center names: {len(self.list_of_joint_center_names)}")
        