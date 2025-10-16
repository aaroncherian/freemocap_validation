from abc import ABC, abstractmethod
from pathlib import Path
import logging
from validation.datatypes.data_component import DataComponent
from typing import List, Type
from validation.pipeline.context import PipelineContext
from validation.pipeline.frame_loop_clause import FrameLoopClause



logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

from dataclasses import dataclass, field
from typing import Any


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

        self.cfg = self._check_config(self.ctx)
        self.frame_loop_clause = FrameLoopClause(self.ctx, self.CONFIG_KEY or self.__class__.__name__)
        self._resolve_requirements()
        self.PRODUCES = self._resolve_products()

    def _check_config(self, ctx:PipelineContext):
        if self.CONFIG is not None:
            key = self.CONFIG_KEY or self.__class__.__name__
            config = ctx.get(f"{key}.config")

            if config is None:
                raise RuntimeError(
                    f"{self.__class__.__name__} requires a {self.CONFIG.__name__}, "
                    "but none was provided in the context. Check the pipeline_config.yaml")
            if isinstance(config, dict):
                config = self.CONFIG(**config)
            self.logger.info(f"Step {self.__class__.__name__} using config: {config}")  
        else:
            config = None
        return config

    def _resolve_requirements(self):
        for requirement in self.REQUIRES:
            val = self.ctx.get(requirement.name)
            if val is None:
                raise RuntimeError(
                    f"{self.__class__.__name__} needs {requirement.name}, "
                    "but it isn’t in the context.")
            self.data[requirement.name] = val

    def _resolve_products(self):
        if not self.frame_loop_clause.do:
            return self.PRODUCES
        return [p.clone_with_prefix(condition) 
                for p in self.PRODUCES 
                for condition in self.ctx.conditions.keys()]

        f = 2

    @abstractmethod
    def calculate(self):
        "Perform calculation and pass results"
        pass
    
    def store(self):
        """Save every output to disk and cache it in context."""
        for result in self.PRODUCES:
            if result.name in self.outputs:
                data = self.outputs[result.name]
                if result.saver is not None:
                    result.save(self.ctx.recording_dir, data, **self.ctx.data_component_context)
                    self.logger.info(f'Saved {result.name} to disk at {result.full_path(self.ctx.recording_dir, **self.ctx.data_component_context)}')
                else: 
                    self.logger.warning(f'No saver found for {result.name}, skipping save to disk')  
                self.ctx.put(result.name, data)
                self.logger.debug(f'Added {result.name} to context')
            else:
                self.logger.warning(f'No output found for {result.name}, skipping save to disk')
    @property
    def requirements(self):
        return self.REQUIRES
    
    @property
    def produced(self):
        return self.PRODUCES

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

    def _preflight_check(self, start_at:int):
        required = []
        produced = []
        for step_cls in self.step_classes[start_at:]:
            required.extend([r for r in step_cls.REQUIRES])
            produced.extend([p for p in step_cls.PRODUCES])

        required = list(set(required))
        required_and_not_produced = [r for r in required if r not in produced]
        self.logger.info("Need to have the following components available to start: " +
                         ", ".join([c.name for c in required_and_not_produced]) + " checking to see if they are on disk...")
        
        for component in required_and_not_produced:
            if self.ctx.get(component.name) is None:
                if not component.exists(self.ctx.recording_dir, **self.ctx.data_component_context):
                    raise FileNotFoundError(f"Preflight check failed: {component.name} is required but not found on disk at {component.full_path(self.ctx.recording_dir, **self.ctx.data_component_context)}")
                self.ctx.put(component.name, component.load(self.ctx.recording_dir, **self.ctx.data_component_context))
                self.logger.info("Found and loaded " + component.name)
        
        
        self._check_requirements_before_running(start_at=start_at)
        self.logger.info("Preflight check passed")
        f = 2

    def run(self, *, start_at: int =0):
        if not (0 <= start_at < len(self.step_classes)):
            raise IndexError(f"start_at={start_at} is outside valid step range (0–{len(self.step_classes) - 1})")

        self._preflight_check(start_at=start_at)
        
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

    path_to_recording = Path(r"D:\2023-06-07_TF01\1.0_recordings\four_camera\sesh_2023-06-07_12_06_15_TF01_flexion_neutral_trial_1")
    cfg_path= Path(r"C:\Users\aaron\Documents\GitHub\freemocap_validation\pipeline_config.yaml")
    cfg_path= Path(r"C:\Users\aaron\Documents\GitHub\freemocap_validation\config_yamls\prosthetic_data\pipeline_config.yaml")

    ctx, step_classes = build_pipeline(cfg_path, path_to_recording)
    
    pipe = ValidationPipeline(
        context=ctx,
        steps= step_classes, 
        logger=logging.getLogger("pipeline"),
    )

    pipe.run(start_at=4)