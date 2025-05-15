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
from dataclasses import dataclass
import pandas as pd
from validation.steps.rmse.dash_app.run_dash_app import run_dash_app
from validation.pipeline.supports_variants import SupportsVariantsMixin
from validation.variant_registry import MetricsVariant, VARIANT_TO_COMPONENT
from enum import Enum
@dataclass
class RMSEVisualizationData:
    joint_dataframe: pd.DataFrame
    rmse_dataframe: pd.DataFrame
    absolute_error_dataframe: pd.DataFrame

class BaseRMSEStep(SupportsVariantsMixin, ValidationStep):
    CONFIG_KEY = "RMSEStep"
    VARIANT_ENUM = MetricsVariant
    VARIANT_TO_COMPONENT = VARIANT_TO_COMPONENT

    REQUIRES = REQUIRES
    PRODUCES = PRODUCES
    CONFIG = RMSEConfig
    
    def calculate(self):
        self.logger.info("Starting RMSE calculation")

        freemocap_joint_centers = self.data[self.FREEMOCAP_COMPONENT.name]
        qualisys_joint_centers = self.data[QUALISYS_SYNCED_JOINT_CENTERS.name]

        freemocap_actor = make_freemocap_actor_from_landmarks(project_config=self.ctx.project_config,
                                                              landmarks=freemocap_joint_centers)
        qualisys_actor = make_qualisys_actor(project_config=self.ctx.project_config,
                                             tracked_points_data=qualisys_joint_centers)
        
        self.rmse_results = calculate_rmse(freemocap_actor=freemocap_actor,
                       qualisys_actor=qualisys_actor,
                       config = self.cfg)
        
        self.outputs[POSITIONRMSE.name] = self.rmse_results.position_rmse
        self.outputs[POSITIONABSOLUTEERROR.name] = self.rmse_results.position_absolute_error
        self.outputs[VELOCITYRMSE.name] = self.rmse_results.velocity_rmse
        self.outputs[VELOCITYABSOLUTEERROR.name] = self.rmse_results.velocity_absolute_error

    def visualize(self):
        position_rmse_gui = RMSEVisualizationData(
            joint_dataframe=self.rmse_results.position_joint_df,
            rmse_dataframe=self.rmse_results.position_rmse,
            absolute_error_dataframe=self.rmse_results.position_absolute_error
        )

        velocity_rmse_gui = RMSEVisualizationData(
            joint_dataframe=self.rmse_results.velocity_joint_df,
            rmse_dataframe=self.rmse_results.velocity_rmse,
            absolute_error_dataframe=self.rmse_results.velocity_absolute_error
        )

        run_dash_app(
            data_and_error=self.rmse_results,
            recording_name = self.ctx.recording_dir.stem
        )

    @classmethod
    def make_variant(cls, variant_enum):
        freemocap_component = cls.VARIANT_TO_COMPONENT[variant_enum]
        prefix       = f"{variant_enum.value}"

        class _Variant(cls):
            variant_prefix       = prefix
            FREEMOCAP_COMPONENT  = freemocap_component
            REQUIRES = [freemocap_component, QUALISYS_SYNCED_JOINT_CENTERS]
            PRODUCES = cls.PRODUCES        # keep base list; mix-in handles cloning

        _Variant.__name__ = f"{cls.__name__}_{variant_enum.value}"
        return _Variant