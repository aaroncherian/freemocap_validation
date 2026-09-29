import logging
from typing import List

import numpy as np
from skellymodels.managers.human import Human

from validation.steps.spatial_alignment.config import SpatialAlignmentConfig
from validation.steps.spatial_alignment.core.alignment_utils import (
    apply_transformation,
    get_best_transformation_matrix_ransac,
)


def run_marker_check(
    freemocap_actor: Human,
    qualisys_actor: Human,
    markers_for_alignment: List[str],
):
    freemocap_markers = set(
        freemocap_actor.body.anatomical_structure.tracked_point_names
    )
    qualisys_markers = set(
        qualisys_actor.body.anatomical_structure.tracked_point_names
    )

    missing_in_freemocap = set(markers_for_alignment) - freemocap_markers
    missing_in_qualisys = set(markers_for_alignment) - qualisys_markers

    if missing_in_freemocap:
        raise ValueError(
            "These markers for alignment were not found in FreeMoCap markers: "
            f"{missing_in_freemocap}"
        )

    if missing_in_qualisys:
        raise ValueError(
            "These markers for alignment were not found in Qualisys markers: "
            f"{missing_in_qualisys}"
        )


def run_ransac_spatial_alignment(
    freemocap_actor: Human,
    qualisys_actor: Human,
    config: SpatialAlignmentConfig,
    logger=None,
):
    if logger is None:
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
        logger = logging.getLogger(__name__)

    markers_for_alignment = config.markers_for_alignment

    run_marker_check(
        freemocap_actor=freemocap_actor,
        qualisys_actor=qualisys_actor,
        markers_for_alignment=markers_for_alignment,
    )

    freemocap_indices = [
        freemocap_actor.body.xyz.landmark_names.index(marker)
        for marker in markers_for_alignment
    ]
    qualisys_indices = [
        qualisys_actor.body.xyz.landmark_names.index(marker)
        for marker in markers_for_alignment
    ]

    freemocap_data_for_alignment = freemocap_actor.body.xyz.as_array[
        :, freemocap_indices, :
    ]
    qualisys_data_for_alignment = qualisys_actor.body.xyz.as_array[
        :, qualisys_indices, :
    ]

    transformation_matrix = get_best_transformation_matrix_ransac(
        freemocap_data=freemocap_data_for_alignment,
        qualisys_data=qualisys_data_for_alignment,
        frames_to_sample=config.frames_to_sample,
        max_iterations=config.max_iterations,
        inlier_threshold=config.inlier_threshold,
        random_seed=config.random_seed,
    )

    logger.info(f"Found rigid transformation matrix as {transformation_matrix}")

    prealigned_freemocap_data = apply_transformation(
        transformation_matrix,
        freemocap_actor.body.xyz.as_array,
    )

    if config.neutral_frames is not None and config.vertical_offset_markers:
        z_offset = compute_vertical_offset(
            freemocap_data=prealigned_freemocap_data,
            qualisys_data=qualisys_actor.body.xyz.as_array,
            freemocap_actor=freemocap_actor,
            qualisys_actor=qualisys_actor,
            neutral_frames=range(
                config.neutral_frames[0],
                config.neutral_frames[1],
            ),
            foot_markers=tuple(config.vertical_offset_markers),
        )

        transformation_matrix = transformation_matrix.copy()
        transformation_matrix[2] += z_offset

        logger.info(f"Added vertical offset of {z_offset:.3f} to transform tz")

    aligned_freemocap_data = apply_transformation(
        transformation_matrix,
        freemocap_actor.body.xyz.as_array,
    )

    return aligned_freemocap_data, transformation_matrix


def compute_vertical_offset(
    freemocap_data,
    qualisys_data,
    freemocap_actor,
    qualisys_actor,
    neutral_frames,
    foot_markers=("left_heel", "right_heel"),
    vertical_axis=2,
):
    freemocap_indices = [
        freemocap_actor.body.xyz.landmark_names.index(marker)
        for marker in foot_markers
    ]
    qualisys_indices = [
        qualisys_actor.body.xyz.landmark_names.index(marker)
        for marker in foot_markers
    ]

    freemocap_heights = freemocap_data[
        neutral_frames
    ][:, freemocap_indices, vertical_axis]

    qualisys_heights = qualisys_data[
        neutral_frames
    ][:, qualisys_indices, vertical_axis]

    freemocap_height = np.nanmedian(freemocap_heights)
    qualisys_height = np.nanmedian(qualisys_heights)

    return qualisys_height - freemocap_height
