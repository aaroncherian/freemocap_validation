from validation.pipeline.base import ValidationStep
from validation.steps.rmse.components import REQUIRES, PRODUCES
from validation.steps.rmse.config import RMSEConfig
from validation.steps.rmse.core.calculate_rmse import calculate_rmse
from validation.components import (
    FREEMOCAP_JOINT_CENTERS, QUALISYS_SYNCED_JOINT_CENTERS, 
    POSITIONABSOLUTEERROR, POSITIONRMSE,
    VELOCITYABSOLUTEERROR, VELOCITYRMSE
)
from validation.utils.actor_utils import make_freemocap_actor_from_landmarks, make_qualisys_actor



class RMSEStep(ValidationStep):
    REQUIRES = REQUIRES
    PRODUCES = PRODUCES
    CONFIG = RMSEConfig
    
    def calculate(self):
        self.logger.info("Starting RMSE calculation")

        freemocap_joint_centers = self.data[FREEMOCAP_JOINT_CENTERS.name]
        qualisys_joint_centers = self.data[QUALISYS_SYNCED_JOINT_CENTERS.name]

        freemocap_actor = make_freemocap_actor_from_landmarks(project_config=self.ctx.project_config,
                                                              landmarks=freemocap_joint_centers)
        qualisys_actor = make_qualisys_actor(project_config=self.ctx.project_config,
                                             tracked_points_data=qualisys_joint_centers)
        
        rmse_results = calculate_rmse(freemocap_actor=freemocap_actor,
                       qualisys_actor=qualisys_actor,
                       config = self.cfg)
        
        self.outputs[POSITIONRMSE.name] = rmse_results.position_rmse
        self.outputs[POSITIONABSOLUTEERROR.name] = rmse_results.position_absolute_error
        self.outputs[VELOCITYRMSE.name] = rmse_results.velocity_rmse
        self.outputs[VELOCITYABSOLUTEERROR.name] = rmse_results.velocity_absolute_error

        



