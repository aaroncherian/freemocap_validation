
from skellymodels.experimental.model_redo.tracker_info.model_info import ModelInfo
from skellymodels.experimental.model_redo.managers.human import Human
from validation.pipeline.project_config import ProjectConfig
import numpy as np
from pathlib import Path

def make_qualisys_actor(project_config: ProjectConfig, tracked_points_data:np.ndarray):
    return Human.from_numpy_array(
    name = "qualisys_human",
    model_info = ModelInfo(config_path= project_config.qualisys_model_info_path),
    tracked_points_numpy_array=tracked_points_data)

def make_freemocap_actor(project_config: ProjectConfig, tracked_points_data:np.ndarray):
    path_to_model_folder = Path(__file__).parent/'freemocap_model_info'
    match project_config.freemocap_tracker:
        case "mediapipe":
            model_info = ModelInfo(config_path= path_to_model_folder/'mediapipe_model_info.yaml')
    return Human.from_numpy_array(
        name = f"{project_config.freemocap_tracker}",
        model_info=model_info,
        tracked_points_numpy_array=tracked_points_data
    ) 
