from validation.pipeline.base import ValidationStep
from validation.steps.joint_angles.components import REQUIRES
from validation.components import FREEMOCAP_PARQUET
from validation.utils.actor_utils import make_freemocap_actor_from_parquet

class JointAnglesStep(ValidationStep):
    REQUIRES = REQUIRES

    def calculate(self):
        self.logger.info("Starting joint angles calculation")

        freemocap_parquet_path = self.data[FREEMOCAP_PARQUET.name]
        freemocap_actor = make_freemocap_actor_from_parquet(parquet_path=freemocap_parquet_path)
        
