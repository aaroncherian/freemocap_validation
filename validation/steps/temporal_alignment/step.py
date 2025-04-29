from validation.steps.temporal_alignment.components import (
    FREEMOCAP_TIMESTAMPS,
    QUALISYS_MARKERS,
    QUALISYS_START_TIME,
)
from validation.steps.temporal_alignment.visualize import SynchronizationVisualizer

from validation.steps.temporal_alignment.core.temporal_synchronizer import TemporalSyncManager
from skellymodels.experimental.model_redo.managers.human import Human

from validation.pipeline.base import ValidationStep
from pathlib import Path
from nicegui import ui

class TemporalAlignmentStep(ValidationStep):
    REQUIRED = [FREEMOCAP_TIMESTAMPS, QUALISYS_MARKERS, QUALISYS_START_TIME]
    def __init__(self, 
                 recording_dir:Path,
                 freemocap_actor: Human):
        super().__init__(recording_dir)

        self.freemocap_actor = freemocap_actor

        self.freemocap_timestamps = self.data["freemocap_timestamps"]
        self.qualisys_dataframe = self.data["qualisys_markers"]
        self.qualisys_unix_start_time = self.data["qualisys_start_time"]


    def calculate(self):
        
        manager = TemporalSyncManager(freemocap_model = self.freemocap_actor,
                                        freemocap_timestamps= self.freemocap_timestamps,
                                        qualisys_marker_data = self.qualisys_dataframe,
                                        qualisys_unix_start_time = self.qualisys_unix_start_time)

        self.freemocap_lag_component, self.qualisys_synced_lag_component, self.qualisys_original_lag_component = manager.run()
        
    def visualize(self):
        sync_gui = SynchronizationVisualizer(
            freemocap_component=self.freemocap_lag_component,
            original_qualisys_component=self.qualisys_original_lag_component,
            corrected_qualisys_component=self.qualisys_synced_lag_component
        )

        sync_gui.create_ui()

        # np.save(path_to_recording/'validation'/'freemocap_3d_xyz.npy', freemocap_lag_component.joint_center_array)
        # np.save(path_to_recording/'validation'/'qualisys_3d_xyz.npy', qualisys_synced_lag_component.joint_center_array)
        f = 2 

    def store(self):
        pass


if __name__ in {"__main__", "__mp_main__"}:
    from skellymodels.experimental.model_redo.tracker_info.model_info import MediapipeModelInfo
    import numpy as np

    path_to_recording = Path(r"D:\2025-04-23_atc_testing\freemocap\2025-04-23_19-11-05-612Z_atc_test_walk_trial_2")
    path_to_data = path_to_recording/'output_data'/'mediapipe_skeleton_3d.npy'

    data = np.load(path_to_data)

    human = Human.from_numpy_array(name = 'human',
                                    model_info=MediapipeModelInfo(),
                                    tracked_points_numpy_array=data)

    step = TemporalAlignmentStep(recording_dir=path_to_recording,
                                 freemocap_actor=human)

    step.calculate()
    step.visualize()
    ui.run()

    f = 2