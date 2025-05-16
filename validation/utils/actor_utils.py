
from skellymodels.experimental.model_redo.tracker_info.model_info import ModelInfo
from skellymodels.experimental.model_redo.managers.human import Human
from validation.pipeline.project_config import ProjectConfig
import numpy as np
from pathlib import Path

def make_qualisys_actor(project_config: ProjectConfig, tracked_points_data:np.ndarray):
    return Human.from_tracked_points_numpy_array(
    name = "qualisys_human",
    model_info = ModelInfo(config_path= project_config.qualisys_model_info_path),
    tracked_points_numpy_array=tracked_points_data)

def get_model_info(project_config: ProjectConfig):
    path_to_model_folder = Path(__file__).parent/'freemocap_model_info'
    match project_config.freemocap_tracker:
        case "mediapipe":
            model_info = ModelInfo(config_path= path_to_model_folder/'mediapipe_model_info.yaml')
        case "yolo":
            model_info = ModelInfo(config_path= path_to_model_folder/'yolo_model_info.yaml')
        case "openpose":
            model_info = ModelInfo(config_path= path_to_model_folder/'openpose_model_info.yaml')
    return model_info

def make_freemocap_actor_from_tracked_points(project_config: ProjectConfig, tracked_points_data:np.ndarray):
    model_info = get_model_info(project_config)
    return Human.from_tracked_points_numpy_array(
        name = f"{project_config.freemocap_tracker}",
        model_info=model_info,
        tracked_points_numpy_array=tracked_points_data
    ) 

def make_freemocap_actor_from_landmarks(project_config: ProjectConfig, landmarks:np.ndarray):
    model_info = get_model_info(project_config)
    return Human.from_landmarks_numpy_array(
        name = f"{project_config.freemocap_tracker}",
        model_info=model_info,
        landmarks_numpy_array=landmarks
    )