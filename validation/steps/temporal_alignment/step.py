from validation.steps.temporal_alignment.components import get_component
from validation.pipeline.base import ValidationStep
from pathlib import Path

class TemporalAlignmentStep(ValidationStep):
    def __init__(self, 
                 recording_dir:Path,
                 timestamps_component_key: str = "freemocap_timestamps",
                 qualisys_component_key: str = "qualisys_markers"):
        super().__init__(recording_dir)
        self.timestamps = get_component(timestamps_component_key)
        self.qualisys_markers = get_component(qualisys_component_key)

    def requires(self):
        return [self.timestamps, self.qualisys_markers]
        
    def calculate(self):
        timestamps = self.timestamps.load(self.recording_dir)
        qualisys_markers = self.qualisys_markers.load(self.recording_dir)
        f = 2

    def store(self):
        pass


if __name__ == '__main__':
    recording_path = Path(r"D:\2025-03-13_JSM_pilot\freemocap_data\2025-03-13T16_20_37_gmt-4_pilot_jsm_treadmill_walking")

    step = TemporalAlignmentStep(recording_dir=recording_path)

    step.calculate()
    f = 2