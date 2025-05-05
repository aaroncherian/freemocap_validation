from validation.steps.spatial_alignment.components import REQUIRES, PRODUCES
from validation.steps.spatial_alignment.config import SpatialAlignmentConfig
from validation.steps.spatial_alignment.core.ransac_spatial_alignment import run_ransac_spatial_alignment


from validation.components import FREEMOCAP_ACTOR, QUALISYS_ACTOR, TRANSFORMATION_MATRIX, FREEMOCAP_ALIGNED
from validation.pipeline.base import ValidationStep


class SpatialAlignmentStep(ValidationStep):
    REQUIRES = REQUIRES
    PRODUCES = PRODUCES
    CONFIG = SpatialAlignmentConfig

    def calculate(self):
        self.logger.info("Starting spatial alignment")

        qualisys_actor = self.ctx.get(QUALISYS_ACTOR.name)
        freemocap_actor = self.data[FREEMOCAP_ACTOR.name]

        aligned_freemocap_data, transformation_matrix = run_ransac_spatial_alignment(
                                    freemocap_actor=freemocap_actor,
                                     qualisys_actor=qualisys_actor,
                                     config=self.cfg,
                                     logger = self.logger)
        
        self.outputs[FREEMOCAP_ALIGNED.name] = aligned_freemocap_data
        self.outputs[TRANSFORMATION_MATRIX.name] = transformation_matrix
                


