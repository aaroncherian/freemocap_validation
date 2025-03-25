from skellymodels.experimental.model_redo.managers.human import Human
from validation.utils.rotation import run_skellyforge_rotation
from validation.steps.temporal_alignment.core.lag_calculation import LagCalculatorComponent
import pandas as pd
class TemporalSyncManager:
    def __init__(self, freemocap_model: Human,
                 freemocap_timestamps: pd.DataFrame):
        self.freemocap_model = freemocap_model
        self.freemocap_timestamps = freemocap_timestamps
        self.run()

    def run(self):
        self._process_freemocap_data()
        self.timestamps, self.framerate = self._get_timestamps()

    def _process_freemocap_data(self):
        freemocap_data = self.freemocap_model.body.trajectories['3d_xyz'].as_numpy
        landmark_names = self.freemocap_model.body.trajectories['3d_xyz'].landmark_names
        origin_aligned_freemocap_data = run_skellyforge_rotation(raw_skeleton_data=freemocap_data,
                                                                 landmark_names=landmark_names)
        self.freemocap_lag_component = LagCalculatorComponent(
            joint_center_array=origin_aligned_freemocap_data,
            list_of_joint_center_names=landmark_names
        )

    def _get_timestamps(self):
        timestamps = self.freemocap_timestamps['timestamp_unix_seconds']
        time_diff = np.diff(timestamps)
        framerate = 1 / np.nanmean(time_diff)
        print(f"Calculated FreeMoCap framerate: {framerate}")
        return timestamps, framerate
        f = 2




if __name__ == '__main__':
    from pathlib import Path
    import numpy as np
    from skellymodels.experimental.model_redo.tracker_info.model_info import MediapipeModelInfo
    from validation.steps.temporal_alignment.components import get_component


    path_to_recording = Path(r"D:\2025-03-13_JSM_pilot\freemocap_data\2025-03-13T16_20_37_gmt-4_pilot_jsm_treadmill_walking")
    path_to_data = path_to_recording/'output_data'/'mediapipe_skeleton_3d.npy'

    data = np.load(path_to_data)

    human = Human.from_numpy_array(name = 'human',
                                   model_info=MediapipeModelInfo(),
                                   tracked_points_numpy_array=data)
    
    freemocap_timestamps = get_component(key = 'freemocap_timestamps')
    
    temp_manager = TemporalSyncManager(freemocap_model = human,
                                       freemocap_timestamps=freemocap_timestamps.load(base_dir= path_to_recording))
    
    f = 2