from validation.steps.temporal_alignment.components import REQUIRES, PRODUCES
from validation.components import QUALISYS_MARKERS,QUALISYS_START_TIME, FREEMOCAP_TIMESTAMPS, QUALISYS_SYNCED_JOINT_CENTERS, FREEMOCAP_ACTOR
#revisit whether this import implementation above is worth it
from validation.steps.temporal_alignment.visualize import SynchronizationVisualizer
from validation.steps.temporal_alignment.core.temporal_synchronizer import TemporalSyncManager
from skellymodels.experimental.model_redo.managers.human import Human

from validation.pipeline.base import ValidationStep
from pathlib import Path
from nicegui import ui


class TemporalAlignmentStep(ValidationStep):
    REQUIRES = REQUIRES
    PRODUCES = PRODUCES

    def calculate(self):
        self.logger.info("Starting temporal alignment")

        freemocap_timestamps   = self.data[FREEMOCAP_TIMESTAMPS.name]
        qualisys_dataframe = self.data[QUALISYS_MARKERS.name]
        qualisys_unix_start_time = self.data[QUALISYS_START_TIME.name]
        freemocap_actor = self.data[FREEMOCAP_ACTOR.name]

        manager = TemporalSyncManager(freemocap_model = freemocap_actor,
                                freemocap_timestamps= freemocap_timestamps,
                                qualisys_marker_data = qualisys_dataframe,
                                qualisys_unix_start_time = qualisys_unix_start_time)
        
        (self.freemocap_lag_component,
        self.qualisys_synced_lag_component,
        self.qualisys_original_lag_component,
        ) = manager.run()
        
        self.outputs[QUALISYS_SYNCED_JOINT_CENTERS.name] = self.qualisys_synced_lag_component.joint_center_array

    def visualize(self):
        sync_gui = SynchronizationVisualizer(
            freemocap_component=self.freemocap_lag_component,
            original_qualisys_component=self.qualisys_original_lag_component,
            corrected_qualisys_component=self.qualisys_synced_lag_component
        )

        sync_gui.create_ui()


if __name__ in {"__main__", "__mp_main__"}:
    
    from skellymodels.experimental.model_redo.tracker_info.model_info import MediapipeModelInfo
    import numpy as np
    from validation.pipeline.base import PipelineContext
    import logging 

    logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] %(message)s')

    path_to_recording = Path(r"D:\2025-04-23_atc_testing\freemocap\2025-04-23_19-11-05-612Z_atc_test_walk_trial_2")
    path_to_data = path_to_recording/'output_data'/'mediapipe_skeleton_3d.npy'

    ctx = PipelineContext(recording_dir=path_to_recording)
    ctx.put(FREEMOCAP_TIMESTAMPS.name,
            FREEMOCAP_TIMESTAMPS.load(path_to_recording))
    ctx.put(QUALISYS_MARKERS.name,
            QUALISYS_MARKERS.load(path_to_recording))
    ctx.put(QUALISYS_START_TIME.name,
            QUALISYS_START_TIME.load(path_to_recording))

    data = np.load(path_to_data)
    human = Human.from_numpy_array(name = 'human',
                                    model_info=MediapipeModelInfo(),
                                    tracked_points_numpy_array=data)
    
    ctx.put("freemocap_actor", human)

    step = TemporalAlignmentStep(ctx, logger=logging.getLogger('temporal_synchronization'))
    step.calculate()
    step.visualize()
    step.store()

    ui.run()

    f = 2