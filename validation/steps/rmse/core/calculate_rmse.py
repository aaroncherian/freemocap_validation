from skellymodels.experimental.model_redo.managers.human import Human
from validation.steps.rmse.config import RMSEConfig
import numpy as np

def add_velocity_to_actor(human:Human):
    velocity_array = np.diff(human.body.trajectories['3d_xyz'].as_numpy, axis = 0)
    human.body.add_trajectory(name = '3d_velocity_xyz',
                                    data = velocity_array,
                                    marker_names=human.body.anatomical_structure.marker_names)

def calculate_rmse(freemocap_actor:Human,
                    qualisys_actor:Human,
                    config: RMSEConfig):
    f = 2
    #think about using timestamps to get true velocity

    add_velocity_to_actor(freemocap_actor)
    add_velocity_to_actor(qualisys_actor)
    f = 2
