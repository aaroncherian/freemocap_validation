from pathlib import Path
from typing import Union
import logging 

from abc import ABC, abstractmethod


class RecordingConfig:
    def __init__(self, path_to_recording: Union[str, Path]):
        self.path_to_recording = Path(path_to_recording)
        
    def create_validation_folder(self):
        validation_folder_name = 'validation'
        path_to_validation = self.path_to_recording/validation_folder_name
        path_to_validation.mkdir(exist_ok=True)

        logging.info(f'Path to validation folder created at {path_to_validation}')



class ValidationPipeline:
    def __init__(self, tasks):
        self.tasks = tasks


class ValidationStep(ABC):
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