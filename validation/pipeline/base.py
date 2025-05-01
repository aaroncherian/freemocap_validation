from abc import ABC, abstractmethod
from pathlib import Path
import logging
from validation.datatypes.data_component import DataComponent
from typing import List, Type


logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

from dataclasses import dataclass, field
from typing import Any

@dataclass
class PipelineContext:
    recording_dir: Path
    backpack: dict[str, Any] = field(default_factory = dict)

    def get(self, name:str) -> Any:
        return self.backpack.get(name)
    
    def put(self, name:str, value: Any) -> Any:
        self.backpack[name] = value

class ValidationStep(ABC):
    REQUIRES: list[DataComponent] = []
    PRODUCES: list[DataComponent] = []

    def __init__(self, context: PipelineContext, logger=None):
        self.ctx = context
        self.logger = logger or logging.getLogger(__name__)
        self.data = {}
        self.outputs = {}

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
            result.save(self.ctx.recording_dir, data)  
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
            val = result.load(self.ctx.recording_dir)
            self.ctx.put(result.name, val)

    def _outputs_exist(self, step:ValidationStep) -> bool:
        return all(c.exists(self.ctx.recording_dir) for c in step.PRODUCES)
    
    def _preload_step_requirements(self, step:ValidationStep):
        for requirement in step.REQUIRES:
            if self.ctx.get(requirement.name) is None:
                if not requirement.exists(self.ctx.recording_dir):
                     raise RuntimeError(f"{requirement.name} is required but not found on disk")
                self.ctx.put(requirement.name, requirement.load(self.ctx.recording_dir))

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


    
    # def _check_requires_on_disk(self, step:ValidationStep, produced: set[str]):
    #     missing = [
    #         c.name
    #         for c in step.REQUIRES
    #         if (c.name not in produced) and (not c.exists(self.recording_dir))
    #     ]

    #     if missing:
    #         raise FileNotFoundError(
    #             f"{step.__class__.__name__} requires {missing}, "
    #             "but they are not on disk or produced in the run"
    #         )
        
    # def _check_needed_files_exist_at_start(self, steps:ValidationStepList, start_at:int, produced:set[str]):
    #     #check outputs from skipped steps already exist
    #     for step in steps[:start_at]:
    #         if not self._outputs_exist(step):
    #             raise RuntimeError(
    #                 f"Cannot start at step {start_at}: "
    #                 f"{step.__class__.__name__} outputs are missing."
    #             )
    #         produced.update(c.name for c in step.PRODUCES)

    # def _check_requirements_before_running(self, steps:ValidationStepList, start_at:int):
    #     produced: set[str] = set()

    #     self._check_needed_files_exist_at_start(steps=steps, start_at = start_at, produced = produced)

    #     for step in self.steps[start_at:]:
    #         self._check_requires_on_disk(step,produced)
    #         produced.update(c.name for c in step.PRODUCES)

    # def run(self, *, start_at: int = 0):
    #     if not (0 <= start_at < len(self.steps)):
    #         raise IndexError(f"start_at={start_at} is outside valid step range (0–{len(self.steps) - 1})")

    #     self._check_requirements_before_running(steps = self.steps, start_at=start_at)

    #     produced: set[str] = set()
    #     self._check_needed_files_exist_at_start(steps=self.steps, start_at = start_at, produced = produced)

    #     for step in self.steps[start_at:]:
    #         self._check_requires_on_disk(step,produced)
    #         self.logger.info(f"Running {step.__class__.__name__}")
    #         step.calculate()
    #         step.store()
    #         if hasattr(step, "visualize"):
    #             step.visualize()
    #         produced.update(c.name for c in step.PRODUCES)

if __name__ == "__main__":
    import logging
    from pathlib import Path
    from validation.steps.temporal_alignment.step import TemporalAlignmentStep


    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    path_to_recording = Path(r"D:\2025-04-23_atc_testing\freemocap\2025-04-23_19-11-05-612Z_atc_test_walk_trial_2")
    ctx = PipelineContext(recording_dir=path_to_recording
                          )
    from skellymodels.experimental.model_redo.managers.human import Human
    from skellymodels.experimental.model_redo.tracker_info.model_info import MediapipeModelInfo
    import numpy as np

    skeleton_data = np.load(path_to_recording / "output_data/mediapipe_skeleton_3d.npy")
    human = Human.from_numpy_array(name="human", model_info=MediapipeModelInfo(), tracked_points_numpy_array=skeleton_data)

    ctx.put("freemocap_actor", Human.from_numpy_array(
        name="actor",
        model_info=MediapipeModelInfo(),
        tracked_points_numpy_array=skeleton_data,
    ))

    pipe = ValidationPipeline(
        context=ctx,
        steps=[TemporalAlignmentStep],  # add more as needed
        logger=logging.getLogger("pipeline"),
    )

    pipe.run(start_at=0)