from validation.steps.temporal_alignment.components import get_component
from validation.steps.temporal_alignment.core.temporal_synchronizer import TemporalSyncManager
from skellymodels.experimental.model_redo.managers.human import Human

from validation.pipeline.base import ValidationStep
from pathlib import Path

class TemporalAlignmentStep(ValidationStep):
    def __init__(self, 
                 recording_dir:Path,
                 freemocap_actor: Human,
                 timestamps_component_key: str = "freemocap_timestamps",
                 qualisys_component_key: str = "qualisys_markers"):
        super().__init__(recording_dir)
        self.freemocap_timestamps = get_component(timestamps_component_key)
        self.qualisys_markers = get_component(qualisys_component_key)
        self.freemocap_actor = freemocap_actor

    def requires(self):
        return [self.freemocap_timestamps, self.qualisys_markers]
        
    def calculate(self):
        freemocap_timestamps = self.freemocap_timestamps.load(self.recording_dir)
        qualisys_data_tuple = self.qualisys_markers.load(self.recording_dir)
        qualisys_df = qualisys_data_tuple.dataframe
        qualisys_unix_start_time = qualisys_data_tuple.unix_start_time
        
        manager = TemporalSyncManager(freemocap_model = self.freemocap_actor,
                                        freemocap_timestamps=freemocap_timestamps,
                                        qualisys_marker_data = qualisys_df,
                                        qualisys_unix_start_time = qualisys_unix_start_time)

        freemocap_lag_component, qualisys_synced_lag_component, qualisys_original_lag_component = manager.run()

    def store(self):
        pass


if __name__ == '__main__':
    from skellymodels.experimental.model_redo.tracker_info.model_info import MediapipeModelInfo
    import numpy as np

    path_to_recording = Path(r"D:\2025-03-13_JSM_pilot\freemocap_data\2025-03-13T16_20_37_gmt-4_pilot_jsm_treadmill_walking")
    path_to_data = path_to_recording/'output_data'/'mediapipe_skeleton_3d.npy'

    data = np.load(path_to_data)

    human = Human.from_numpy_array(name = 'human',
                                    model_info=MediapipeModelInfo(),
                                    tracked_points_numpy_array=data)


    step = TemporalAlignmentStep(recording_dir=path_to_recording,
                                 freemocap_actor=human)

    step.calculate()
    f = 2