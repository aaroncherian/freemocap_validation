from validation.pipeline.base import ValidationStep
from validation.steps.create_qualisys_actor.components import REQUIRES, PRODUCES
from validation.components import QUALISYS_SYNCED_JOINT_CENTERS, QUALISYS_ACTOR

from skellymodels.experimental.model_redo.tracker_info.model_info import ModelInfo
from skellymodels.experimental.model_redo.managers.human import Human

class QualisysActorStep(ValidationStep):
    REQUIRES = REQUIRES
    PRODUCES = PRODUCES

    def calculate(self):
        qualisys_joint_centers = self.data[QUALISYS_SYNCED_JOINT_CENTERS.name]
        qualisys_model_info = ModelInfo(config_path=r"C:\Users\aaron\Documents\GitHub\freemocap_validation\qualisys_markerset\qualisys_model_info.yaml")

        qualisys_actor = Human.from_numpy_array(name = "qualisys_human",
                                                model_info= qualisys_model_info,
                                                tracked_points_numpy_array=qualisys_joint_centers)
        self.outputs[QUALISYS_ACTOR.name] = qualisys_actor
        f = 2

