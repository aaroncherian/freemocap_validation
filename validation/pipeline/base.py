from abc import ABC, abstractmethod
from pathlib import Path
import logging
from validation.datatypes.data_component import DataComponent

class ValidationPipeline:
    def __init__(self, tasks):
        self.tasks = tasks


class ValidationStep(ABC):
    REQUIRED: list[DataComponent] = []
    
    def __init__(self, recording_dir: Path, logger=None):
        self.recording_dir = recording_dir
        self.logger = logger or logging.getLogger(__name__)

        self.data = {}
        missing_requirements = []
        for required_component in self.REQUIRED:
            if required_component.exists(recording_dir):
                self.data[required_component.name] = required_component.load(recording_dir)
            else:
                missing_requirements.append(required_component.name)

        if missing_requirements:
            raise FileNotFoundError(f"Missing inputs: {missing_requirements}")


    @abstractmethod
    def calculate(self):
        "Perform calculation and pass results"
        pass
    
    @abstractmethod
    def store(self):
        "Add data to config"
        pass

