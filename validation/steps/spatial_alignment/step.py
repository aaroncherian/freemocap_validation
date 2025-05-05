from validation.steps.spatial_alignment.components import REQUIRES, PRODUCES
from validation.steps.spatial_alignment.config import SpatialAlignmentConfig
from validation.steps.spatial_alignment.core.ransac_spatial_alignment import run_ransac_spatial_alignment
from validation.steps.spatial_alignment.visualize import visualize_spatial_alignment

from validation.utils.actor_utils import make_qualisys_actor, make_freemocap_actor
from validation.components import FREEMOCAP_PRE_SYNC_JOINT_CENTERS, TRANSFORMATION_MATRIX, QUALISYS_SYNCED_JOINT_CENTERS
from validation.pipeline.base import ValidationStep


class SpatialAlignmentStep(ValidationStep):
    REQUIRES = REQUIRES
    PRODUCES = PRODUCES
    CONFIG = SpatialAlignmentConfig

    def calculate(self):
        self.logger.info("Starting spatial alignment")

        qualisys_actor = make_qualisys_actor(project_config=self.ctx.project_config,
                                             tracked_points_data=self.data[QUALISYS_SYNCED_JOINT_CENTERS.name])
        
        freemocap_actor = make_freemocap_actor(project_config=self.ctx.project_config,
                                               tracked_points_data=self.data[FREEMOCAP_PRE_SYNC_JOINT_CENTERS.name])

        aligned_freemocap_data, transformation_matrix = run_ransac_spatial_alignment(
                                    freemocap_actor=freemocap_actor,
                                     qualisys_actor=qualisys_actor,
                                     config=self.cfg,
                                     logger = self.logger)
        f = 2
        # self.outputs[FREEMOCAP_ALIGNED.name] = aligned_freemocap_data
        self.outputs[TRANSFORMATION_MATRIX.name] = transformation_matrix

    def visualize(self):
        self.logger.info('Starting up Plotly visualization for spatial alignment')
        visualize_spatial_alignment(
            freemocap_actor=self.data[FREEMOCAP_PRE_SYNC_JOINT_CENTERS.name],
            qualisys_actor=self.ctx.get(QUALISYS_ACTOR.name),
            aligned_freemocap_array=self.outputs[FREEMOCAP_ALIGNED.name]
        )


