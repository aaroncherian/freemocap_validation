from skellymodels.experimental.model_redo.managers.human import Human
from validation.steps.rmse.config import RMSEConfig
from validation.steps.rmse.core.error_metrics_builder import get_error_metrics
import numpy as np
import pandas as pd

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
    markers_for_comparison = config.markers_for_comparison
    add_velocity_to_actor(freemocap_actor)
    add_velocity_to_actor(qualisys_actor)
    
    common_markers_freemocap_df = freemocap_actor.body.trajectories['3d_xyz'].as_dataframe.query('keypoint in @markers_for_comparison')
    common_markers_freemocap_df['system'] = 'freemocap'
    common_markers_qualisys_df = qualisys_actor.body.trajectories['3d_xyz'].as_dataframe.query('keypoint in @markers_for_comparison')
    common_markers_qualisys_df['system'] = 'qualisys'

    combined_df = pd.concat([common_markers_freemocap_df, common_markers_qualisys_df], ignore_index=True)
    position_error_metrics_dict = get_error_metrics(dataframe_of_3d_data=combined_df)

    f = 2
