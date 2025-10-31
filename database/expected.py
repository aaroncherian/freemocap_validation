from validation.components import (FREEMOCAP_PARQUET, 
                                   QUALISYS_PARQUET, 
                                   FREEMOCAP_GAIT_EVENTS, 
                                   QUALISYS_GAIT_EVENTS,
                                   FREEMOCAP_JOINT_ANGLES,
                                    QUALISYS_JOINT_ANGLES,
                                    FREEMOCAP_JOINT_ANGLE_CYCLES,
                                    QUALISYS_JOINT_ANGLE_CYCLES,
                                    FREEMOCAP_JOINT_ANGLE_SUMMARY_STATS,
                                    QUALISYS_JOINT_ANGLE_SUMMARY_STATS,
                                    POSITIONRMSE,
                                    VELOCITYRMSE,
                                    FREEMOCAP_TRAJECTORY_CYCLES,
                                    QUALISYS_TRAJECTORY_CYCLES,
                                    FREEMOCAP_TRAJECTORY_SUMMARY_STATS,
                                    QUALISYS_TRAJECTORY_SUMMARY_STATS,
                                    )
from validation.datatypes.data_component import DataComponent


FREEMOCAP_PATH_LENGTH_COM = DataComponent(
    name="path_length_com",
    filename = "condition_data.json",
    relative_path= "validation/{tracker}/path_length_analysis"
)

QUALISYS_PATH_LENGTH_COM = DataComponent(
    name="qualisys_path_length_com",
    filename = "condition_data.json",
    relative_path= "validation/qualisys/path_length_analysis"
)

BALANCE = {
    "synced_data": [
        FREEMOCAP_PARQUET,
        QUALISYS_PARQUET
    ],

    "com_analysis": [
        FREEMOCAP_PATH_LENGTH_COM,
        QUALISYS_PATH_LENGTH_COM
    ]
}


TREADMILL = {
    "synced_data": [
        FREEMOCAP_PARQUET,
        QUALISYS_PARQUET
    ],
    "gait_events": [
        FREEMOCAP_GAIT_EVENTS,
        QUALISYS_GAIT_EVENTS
    ],
    "joint_angles": [
        FREEMOCAP_JOINT_ANGLES,
        QUALISYS_JOINT_ANGLES,
    ],
    "joint_angles_per_stride": [
        FREEMOCAP_JOINT_ANGLE_CYCLES,
        QUALISYS_JOINT_ANGLE_CYCLES,
        FREEMOCAP_JOINT_ANGLE_SUMMARY_STATS,
        QUALISYS_JOINT_ANGLE_SUMMARY_STATS
    ],

    "rmse_metrics": [
        POSITIONRMSE,
        VELOCITYRMSE
    ],
    "trajectories_per_stride": [
        FREEMOCAP_TRAJECTORY_CYCLES,
        QUALISYS_TRAJECTORY_CYCLES,
        FREEMOCAP_TRAJECTORY_SUMMARY_STATS,
        QUALISYS_TRAJECTORY_SUMMARY_STATS
    ]




}
