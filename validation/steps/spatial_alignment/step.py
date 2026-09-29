import numpy as np
from skellymodels.managers.human import Human

from validation.components import (
    FREEMOCAP_PARQUET,
    FREEMOCAP_PRE_SYNC_JOINT_CENTERS,
    QUALISYS_PARQUET,
    QUALISYS_SYNCED_JOINT_CENTERS,
    TRANSFORMATION_MATRIX,
)
from validation.pipeline.base import ValidationStep
from validation.steps.spatial_alignment.components import PRODUCES, REQUIRES
from validation.steps.spatial_alignment.config import SpatialAlignmentConfig
from validation.steps.spatial_alignment.core.alignment_utils import apply_transformation
from validation.steps.spatial_alignment.core.ransac_spatial_alignment import (
    run_ransac_spatial_alignment,
)
from validation.utils.actor_utils import (
    make_freemocap_actor_from_landmarks,
    make_freemocap_actor_from_tracked_points,
    make_qualisys_actor,
)


class SpatialAlignmentStep(ValidationStep):
    REQUIRES = REQUIRES
    PRODUCES = PRODUCES
    CONFIG = SpatialAlignmentConfig

    def calculate(self):
        self.logger.info("Starting spatial alignment")

        qualisys_actor = make_qualisys_actor(
            project_config=self.ctx.project_config,
            tracked_points_data=self.data[QUALISYS_SYNCED_JOINT_CENTERS.name],
        )

        current_tracker = self.ctx.project_config.freemocap_tracker

        freemocap_actor = make_freemocap_actor_from_tracked_points(
            freemocap_tracker=current_tracker,
            tracked_points_data=self.data[FREEMOCAP_PRE_SYNC_JOINT_CENTERS.name],
        )
        self.freemocap_actor = freemocap_actor

        source_tracker = self.cfg.transform_source_tracker

        if source_tracker is not None:
            if source_tracker == current_tracker:
                raise ValueError(
                    "transform_source_tracker cannot be the same as "
                    f"the current tracker ({current_tracker})."
                )

            transform_path = (
                self.ctx.recording_dir
                / "validation"
                / source_tracker
                / TRANSFORMATION_MATRIX.filename
            )

            if not transform_path.exists():
                raise FileNotFoundError(
                    "Requested shared spatial transform does not exist. "
                    f"Run '{source_tracker}' for this recording first:\n"
                    f"{transform_path}"
                )

            transformation_matrix = np.load(transform_path)

            aligned_freemocap_data = apply_transformation(
                transformation_matrix,
                freemocap_actor.body.xyz.as_array,
            )

            self.logger.info(
                "Reusing spatial transform from tracker "
                f"'{source_tracker}': {transform_path}"
            )

        else:
            aligned_freemocap_data, transformation_matrix = (
                run_ransac_spatial_alignment(
                    freemocap_actor=freemocap_actor,
                    qualisys_actor=qualisys_actor,
                    config=self.cfg,
                    logger=self.logger,
                )
            )

        aligned_freemocap_actor: Human = make_freemocap_actor_from_landmarks(
            freemocap_tracker=current_tracker,
            landmarks=aligned_freemocap_data,
        )
        aligned_freemocap_actor.calculate()

        self.outputs[TRANSFORMATION_MATRIX.name] = transformation_matrix

        self.ctx.freemocap_path.mkdir(parents=True, exist_ok=True)

        aligned_freemocap_actor.save_out_all_xyz_numpy_data(
            self.ctx.freemocap_path
        )
        aligned_freemocap_actor.save_out_all_data_csv(
            self.ctx.freemocap_path
        )
        aligned_freemocap_actor.save_out_all_data_parquet(
            self.ctx.freemocap_path
        )
        self.outputs[FREEMOCAP_PARQUET.name] = (
            self.ctx.freemocap_path / FREEMOCAP_PARQUET.filename
        )

        qualisys_actor.save_out_all_xyz_numpy_data(self.ctx.qualisys_path)
        qualisys_actor.save_out_all_data_csv(self.ctx.qualisys_path)
        qualisys_actor.save_out_all_data_parquet(self.ctx.qualisys_path)
        self.outputs[QUALISYS_PARQUET.name] = (
            self.ctx.qualisys_path / QUALISYS_PARQUET.filename
        )

        self.qualisys_actor = qualisys_actor
        self.freemocap_actor = freemocap_actor
