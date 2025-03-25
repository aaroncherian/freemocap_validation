from abc import ABC, abstractmethod
from pathlib import Path
import logging

class ValidationPipeline:
    def __init__(self, tasks):
        self.tasks = tasks


class ValidationStep(ABC):
    def __init__(self, recording_dir: Path, logger=None):
        self.recording_dir = recording_dir
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def calculate(self):
        "Perform calculation and pass results"
        pass
    
    @abstractmethod
    def store(self):
        "Add data to config"
        pass

    @abstractmethod
    def requires(self):
        "Check requirements for step"
        pass