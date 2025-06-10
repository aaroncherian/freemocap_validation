from abc import ABC, abstractmethod
from pathlib import Path
import logging
from validation.datatypes.data_component import DataComponent
from typing import List, Type
from validation.pipeline.project_config import ProjectConfig



logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

from dataclasses import dataclass, field
from typing import Any

@dataclass
class PipelineContext:
    recording_dir: Path
    project_config: ProjectConfig
    backpack: dict[str, Any] = field(default_factory = dict)

    def get(self, name:str) -> Any:
        return self.backpack.get(name)
    
    def put(self, name:str, value: Any) -> Any:
        self.backpack[name] = value

    @property
    def data_component_context(self) -> dict:
        return {
            "tracker": self.project_config.freemocap_tracker,
            "recording_name": self.recording_dir.stem,
        }

class ValidationStep(ABC):
    REQUIRES: list[DataComponent] = []
    PRODUCES: list[DataComponent] = []
    CONFIG = None
    CONFIG_KEY: str | None = None

    def __init__(self, context: PipelineContext, logger=None):
        self.ctx = context
        self.logger = logger or logging.getLogger(__name__)
        self.data = {}
        self.outputs = {}

        if self.CONFIG is not None:
            key = self.CONFIG_KEY or self.__class__.__name__
            config = self.ctx.get(f"{key}.config")

            if config is None:
                raise RuntimeError(
                    f"{self.__class__.__name__} requires a {self.CONFIG.__name__}, "
                    "but none was provided in the context. Check the pipeline_config.yaml")
            if isinstance(config, dict):
                config = self.CONFIG(**config)
                self.cfg = config
            self.logger.info(f"Step {self.__class__.__name__} using config: {self.cfg}")  
        else:
            self.cfg = None

        for requirement in self.REQUIRES:
            val = self.ctx.get(requirement.name)
            if val is None:
                raise RuntimeError(
                f"{self.__class__.__name__} needs {requirement.name}, "
                "but it isn’t in the context.")                    
            self.data[requirement.name] = val

    @abstractmethod
    def calculate(self):
        "Perform calculation and pass results"
        pass
    
    def store(self):
        """Save every output to disk and cache it in context."""
        for result in self.PRODUCES:
            data = self.outputs[result.name]
            if result.saver is not None:
                result.save(self.ctx.recording_dir, data, **self.ctx.data_component_context)
            else: 
                self.logger.warning(f'No saver found for {result.name}, skipping save to disk')  
            self.ctx.put(result.name, data)
            self.logger.info(f'Added {result.name} to context')
           

ValidationStepClass = Type[ValidationStep]
ValidationStepList = List[ValidationStepClass]

class ValidationPipeline:
    def __init__(
            self,
            context:PipelineContext,
            steps: ValidationStepList,
            logger: logging.Logger | None = None,
    ):  
        self.ctx = context
        self.logger = logger or logging.getLogger(__name__)
        self.step_classes = steps
    
    def _load_outputs_into_context(self, step_cls:ValidationStep):
        for result in step_cls.PRODUCES:
            val = result.load(self.ctx.recording_dir, **self.ctx.data_component_context)
            self.ctx.put(result.name, val)

    def _outputs_exist(self, step:ValidationStep) -> bool:
        return all(c.exists(self.ctx.recording_dir, **self.ctx.data_component_context) for c in step.PRODUCES)
    
    def _preload_step_requirements(self, step:ValidationStep):
        for requirement in step.REQUIRES:
            if self.ctx.get(requirement.name) is None:
                if not requirement.exists(self.ctx.recording_dir, **self.ctx.data_component_context):
                     raise RuntimeError(f"{requirement.name} is required but not found on disk at {requirement.full_path(self.ctx.recording_dir, **self.ctx.data_component_context)}")
                self.ctx.put(requirement.name, requirement.load(self.ctx.recording_dir, **self.ctx.data_component_context))

    def _check_requirements_before_running(self, start_at:int):
        
        produced: set[str] = set(self.ctx.backpack.keys())

        for step_cls in self.step_classes[start_at:]:
            missing = [
                req.name for req in step_cls.REQUIRES
                if req.name not in produced
            ]
            if missing:
                raise FileNotFoundError(
                    f"{step_cls.__name__} is missing requirements: {missing}"
            )
            produced.update(component.name for component in step_cls.PRODUCES)

            
    def run(self, *, start_at: int =0):
        if not (0 <= start_at < len(self.step_classes)):
            raise IndexError(f"start_at={start_at} is outside valid step range (0–{len(self.step_classes) - 1})")

        #loads outputs of any skipped stages
        for step_cls in self.step_classes[:start_at]:
            if not self._outputs_exist(step_cls):
                raise RuntimeError(
                    f"Cannot start at step {start_at}: "
                    f"{step_cls.__name__} outputs missing on disk"
                )
            self._load_outputs_into_context(step_cls)

        #preloads the inputs of the first step into context
        self._preload_step_requirements(self.step_classes[start_at])        
        self._check_requirements_before_running(start_at=start_at)    

        
        #run the pipeline
        for step_cls in self.step_classes[start_at:]:
            step = step_cls(self.ctx, logger=self.logger)

            self.logger.info(f"Running {step_cls.__name__}")
            step.calculate()
            step.store()

            if hasattr(step, 'visualize'):
                step.visualize()


if __name__ == "__main__":
    import logging
    from pathlib import Path

    from validation.pipeline.project_config import ProjectConfig
    from validation.pipeline.builder import build_pipeline
 
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    path_to_recording = Path(r"D:\2025-04-23_atc_testing\freemocap\2025-04-23_19-11-05-612Z_atc_test_walk_trial_2")


    cfg_path= Path(r"C:\Users\aaron\Documents\GitHub\freemocap_validation\rtmpose_pipeline_config.yaml")

    ctx, step_classes = build_pipeline(cfg_path, path_to_recording)
    
    pipe = ValidationPipeline(
        context=ctx,
        steps= step_classes, 
        logger=logging.getLogger("pipeline"),
    )

    pipe.run(start_at=0)