from skellymodels.experimental.model_redo.managers.human import Human
from validation.steps.rmse.config import RMSEConfig
from validation.steps.rmse.core.error_metrics_builder import get_error_metrics
import numpy as np
import pandas as pd

from dataclasses import dataclass
@dataclass
class RMSEResults:
    position_joint_df: pd.DataFrame
    position_rmse: pd.DataFrame
    position_absolute_error:pd.DataFrame
    velocity_joint_df: pd.DataFrame
    velocity_rmse: pd.DataFrame
    velocity_absolute_error:pd.DataFrame

def add_velocity_to_actor(human:Human):
    velocity_array = np.diff(human.body.trajectories['3d_xyz'].as_numpy, axis = 0)
    human.body.add_trajectory(name = '3d_velocity_xyz',
                                    data = velocity_array,
                                    marker_names=human.body.anatomical_structure.marker_names)

def combine_system_dataframes_on_common_markers(markers_for_comparison: list[str], 
                                                trajectory_name:str,
                                                freemocap_actor: Human,
                                                qualisys_actor: Human) -> pd.DataFrame:
    common_markers_freemocap_df = freemocap_actor.body.trajectories[trajectory_name].as_dataframe.query('keypoint in @markers_for_comparison')
    common_markers_freemocap_df['system'] = 'freemocap'

    common_markers_qualisys_df = qualisys_actor.body.trajectories[trajectory_name].as_dataframe.query('keypoint in @markers_for_comparison')
    common_markers_qualisys_df['system'] = 'qualisys'
    return pd.concat([common_markers_freemocap_df, common_markers_qualisys_df], ignore_index=True)

def calculate_rmse(freemocap_actor:Human,
                    qualisys_actor:Human,
                    config: RMSEConfig) -> RMSEResults:
    f = 2
    #think about using timestamps to get true velocity
    markers_for_comparison = config.markers_for_comparison
    add_velocity_to_actor(freemocap_actor)
    add_velocity_to_actor(qualisys_actor)

    combined_position_df = combine_system_dataframes_on_common_markers(
                                                                    markers_for_comparison=markers_for_comparison,
                                                                    trajectory_name='3d_xyz',
                                                                    freemocap_actor=freemocap_actor,
                                                                    qualisys_actor=qualisys_actor)
    
    start = config.start_frame or 0
    end = config.end_frame or freemocap_actor.body.trajectories['3d_xyz'].as_numpy.shape[0]
    combined_position_df = combined_position_df[
    (combined_position_df['frame'] >= start) &
    (combined_position_df['frame'] <= end)
]

    position_error_metrics_dict = get_error_metrics(dataframe_of_3d_data=combined_position_df)


    combined_velocity_df = combine_system_dataframes_on_common_markers(
                                                                markers_for_comparison=markers_for_comparison,
                                                                trajectory_name='3d_velocity_xyz',
                                                                freemocap_actor=freemocap_actor,
                                                                qualisys_actor=qualisys_actor)

    velocity_error_metrics_dict = get_error_metrics(dataframe_of_3d_data=combined_velocity_df)

    return RMSEResults(
        position_joint_df= combined_position_df,
        position_rmse= position_error_metrics_dict['rmse_dataframe'],
        position_absolute_error= position_error_metrics_dict['absolute_error_dataframe'],
        velocity_joint_df= combined_velocity_df,
        velocity_rmse= velocity_error_metrics_dict['rmse_dataframe'],
        velocity_absolute_error= velocity_error_metrics_dict['absolute_error_dataframe']
    )
    f = 2
