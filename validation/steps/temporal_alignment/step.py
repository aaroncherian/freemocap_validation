from validation.steps.temporal_alignment.components import get_component
from validation.steps.temporal_alignment.core.temporal_synchronizer import TemporalSyncManager
from skellymodels.experimental.model_redo.managers.human import Human

from validation.pipeline.base import ValidationStep
from pathlib import Path

from typing import Dict
class TemporalAlignmentStep(ValidationStep):
    def __init__(self, 
                 recording_dir:Path,
                 freemocap_actor: Human,
                 requirements: Dict):
        super().__init__(recording_dir)

        self.freemocap_actor = freemocap_actor
        self.requirements = requirements

        freemocap_timestamps_component, qualisys_data_component = self.get_required_components()

        self.freemocap_timestamps = freemocap_timestamps_component.load(base_dir= path_to_recording)
        qualisys_data_tuple = qualisys_data_component.load(base_dir=path_to_recording)

        self.qualisys_dataframe = qualisys_data_tuple.dataframe
        self.qualisys_unix_start_time = qualisys_data_tuple.unix_start_time

    def requires(self):
        return [self.freemocap_timestamps, self.qualisys_markers]
    
    def get_required_components(self):
        freemocap_timestamps_component = get_component(key = self.requirements['freemocap_timestamps_component'])
        qualisys_data_component = get_component(key = requirements['qualisys_markers_component'])

        return freemocap_timestamps_component, qualisys_data_component

    def calculate(self):
        
        manager = TemporalSyncManager(freemocap_model = self.freemocap_actor,
                                        freemocap_timestamps= self.freemocap_timestamps,
                                        qualisys_marker_data = self.qualisys_dataframe,
                                        qualisys_unix_start_time = self.qualisys_unix_start_time)

        freemocap_lag_component, qualisys_synced_lag_component, qualisys_original_lag_component = manager.run()

        # np.save(path_to_recording/'validation'/'freemocap_3d_xyz.npy', freemocap_lag_component.joint_center_array)
        # np.save(path_to_recording/'validation'/'qualisys_3d_xyz.npy', qualisys_synced_lag_component.joint_center_array)
        f = 2 

    def store(self):
        pass


if __name__ == '__main__':
    from skellymodels.experimental.model_redo.tracker_info.model_info import MediapipeModelInfo
    import numpy as np

    path_to_recording = Path(r"D:\2025-04-23_atc_testing\freemocap\2025-04-23_19-11-05-612Z_atc_test_walk_trial_2")
    path_to_data = path_to_recording/'output_data'/'mediapipe_skeleton_3d.npy'

    data = np.load(path_to_data)

    human = Human.from_numpy_array(name = 'human',
                                    model_info=MediapipeModelInfo(),
                                    tracked_points_numpy_array=data)

    requirements = {'freemocap_timestamps_component':'freemocap_timestamps', 
                     'qualisys_markers_component': 'qualisys_markers'}
    step = TemporalAlignmentStep(recording_dir=path_to_recording,
                                 freemocap_actor=human,
                                 requirements = requirements)

    step.calculate()
    f = 2