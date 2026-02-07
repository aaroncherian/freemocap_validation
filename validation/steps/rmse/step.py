from pathlib import Path

from validation.pipeline.base import ValidationStep
from validation.steps.rmse.components import REQUIRES, PRODUCES
from validation.steps.rmse.config import RMSEConfig
from validation.steps.rmse.core.calculate_rmse import calculate_rmse
from validation.components import (
    QUALISYS_PARQUET, FREEMOCAP_PARQUET,
    POSITIONABSOLUTEERROR, POSITIONRMSE,
    VELOCITYABSOLUTEERROR, VELOCITYRMSE
)
from validation.utils.actor_utils import make_freemocap_actor_from_parquet


class RMSEStep(ValidationStep):
    REQUIRES = REQUIRES
    PRODUCES = PRODUCES
    CONFIG = RMSEConfig

    def calculate(self, condition_frame_range: list[int] = None, condition_name: str | None = None):
        self.logger.info("Starting RMSE calculation")

        # IMPORTANT: use sweep-aware parquet paths (ignore DataComponent-saved paths)
        freemocap_parquet_path = self.ctx.freemocap_path / "freemocap_data_by_frame.parquet"
        qualisys_parquet_path  = self.ctx.qualisys_path  / "freemocap_data_by_frame.parquet"

        if not freemocap_parquet_path.exists():
            raise FileNotFoundError(f"Missing FreeMoCap parquet: {freemocap_parquet_path}")
        if not qualisys_parquet_path.exists():
            raise FileNotFoundError(f"Missing Qualisys parquet: {qualisys_parquet_path}")

        freemocap_actor = make_freemocap_actor_from_parquet(parquet_path=freemocap_parquet_path)
        qualisys_actor  = make_freemocap_actor_from_parquet(parquet_path=qualisys_parquet_path)

        self.rmse_results = calculate_rmse(
            freemocap_actor=freemocap_actor,
            qualisys_actor=qualisys_actor,
            config=self.cfg,
            frame_range=condition_frame_range,
        )

        # still populate outputs if you want
        self.outputs[POSITIONRMSE.name] = self.rmse_results.position_rmse
        self.outputs[POSITIONABSOLUTEERROR.name] = self.rmse_results.position_absolute_error
        self.outputs[VELOCITYRMSE.name] = self.rmse_results.velocity_rmse
        self.outputs[VELOCITYABSOLUTEERROR.name] = self.rmse_results.velocity_absolute_error

        # Manual save (sweep-safe)
        cond = condition_name or "overall"
        rmse_root = self.ctx.freemocap_path / "rmse"

        pos_dir = rmse_root / "position" / cond
        vel_dir = rmse_root / "velocity" / cond
        pos_dir.mkdir(parents=True, exist_ok=True)
        vel_dir.mkdir(parents=True, exist_ok=True)

        self.rmse_results.position_absolute_error.to_csv(pos_dir / "position_absolute_error.csv", index=False)
        self.rmse_results.position_rmse.to_csv(pos_dir / "position_rmse.csv", index=False)
        self.rmse_results.velocity_absolute_error.to_csv(vel_dir / "velocity_absolute_error.csv", index=False)
        self.rmse_results.velocity_rmse.to_csv(vel_dir / "velocity_rmse.csv", index=False)

        self.logger.info(f"Saved {cond}_position_rmse to {pos_dir / 'position_rmse.csv'}")
        self.logger.info(f"Saved {cond}_velocity_rmse to {vel_dir / 'velocity_rmse.csv'}")

    # override so we can pass condition_name without touching the base ABC
    def calculate_and_store(self):
        if not self.loop_enabled:
            self.calculate(condition_frame_range=None, condition_name=None)
            self.outputs.clear()
            return

        for condition_name, frames in self.ctx.conditions.items():
            self.logger.info(f"Running RMSE for condition '{condition_name}' with frames {frames}")
            self.calculate(condition_frame_range=frames["frames"], condition_name=condition_name)
            self.outputs.clear()
