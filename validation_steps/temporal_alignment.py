from models.recording_config import ValidationStep


class FileComponent:
    name: str
    files: dict


timestamps_component = FileComponent(
    name = 'freemocap_timestamps',
    files = {
        'timestamps': 'unix_synced_timestamps.csv'
    }
)

qualisys_tsv_component = FileComponent(
    name = 'qualisys_exported_markers',
    files = {
        'markers': 'qualisys_exported_markers.tsv'
    }
)

class TemporalAlignmentStep(ValidationStep):
    def requires(self):
        