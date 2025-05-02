from validation.steps.spatial_alignment.components import REQUIRES, PRODUCES
from validation.steps.spatial_alignment.config import SpatialAlignmentConfig

from validation.components import QUALISYS_SYNCED_JOINT_CENTERS, FREEMOCAP_ACTOR


from validation.pipeline.base import ValidationStep


class SpatialAlignmentStep(ValidationStep):
    REQUIRES = REQUIRES
    PRODUCES = PRODUCES
    CONFIG = SpatialAlignmentConfig

    def calculate(self):
        self.logger.info("Starting spatial alignment")

        qualisys_joint_centers = self.data[QUALISYS_SYNCED_JOINT_CENTERS.name]
        freemocap_actor = self.data[FREEMOCAP_ACTOR.name]
        
    def store(self):
        pass

