from components import freemocap_timestamps, qualisys_markers
from pipeline.base import ValidationStep
from pathlib import Path

class TemporalAlignmentStep(ValidationStep):
    def __init__(self, recording_dir:Path):
        super().__init__(recording_dir)
        self.timestamps = freemocap_timestamps
        self.qualisys_markers = qualisys_markers

    def requires(self):
        return [self.timestamps, self.qualisys_markers]
    
    def calculate(self):
        timestamps = self.timestamps.load(self.recording_dir)
        qualisys_markers = self.qualisys_markers.load(self.recording_dir)

