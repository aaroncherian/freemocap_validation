from skellymodels.managers.human import Human
from validation.steps.spatial_alignment.config import SpatialAlignmentConfig
from validation.steps.spatial_alignment.core.alignment_utils import get_best_transformation_matrix_ransac, apply_transformation
import numpy as np
from typing import List
import logging


def run_marker_check(freemocap_actor:Human,
                     qualisys_actor:Human,
                     markers_for_alignment: List[str]):
    """
    Validates the presence of alignment markers in both FreeMoCap and Qualisys skeleton models.

    This function checks whether all the markers specified for alignment are present in the marker lists
    of both the FreeMoCap and Qualisys skeleton models. If any markers are missing, it raises a ValueError
    with a descriptive message indicating which markers are missing in which model.
    """
    
    freemocap_markers = set(freemocap_actor.body.anatomical_structure.tracked_point_names)
    qualisys_markers = set(qualisys_actor.body.anatomical_structure.tracked_point_names)

    missing_in_freemocap = set(markers_for_alignment) - freemocap_markers
    missing_in_qualisys = set(markers_for_alignment) - qualisys_markers

    if missing_in_freemocap:
        raise ValueError(f"These markers for alignment were not found in FreeMoCap markers: {missing_in_freemocap}")

    if missing_in_qualisys:
        raise ValueError(f"These markers for alignment were not found in Qualisys markers: {missing_in_qualisys}")

def run_ransac_spatial_alignment(freemocap_actor:Human, 
                                 qualisys_actor:Human,
                                 config: SpatialAlignmentConfig,
                                 logger = None,
                                 ):
    
    if logger is None:
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
        logger = logging.getLogger(__name__)

    markers_for_alignment = config.markers_for_alignment

    run_marker_check(freemocap_actor=freemocap_actor,
                     qualisys_actor=qualisys_actor,
                     markers_for_alignment= config.markers_for_alignment)
    
    freemocap_indices = [freemocap_actor.body.xyz.landmark_names.index(marker) for marker in markers_for_alignment]
    qualisys_indices = [qualisys_actor.body.xyz.landmark_names.index(marker) for marker in markers_for_alignment]
    
    freemocap_data_for_alignment = freemocap_actor.body.xyz.as_array[:,freemocap_indices,:]
    qualisys_data_for_alignment = qualisys_actor.body.xyz.as_array[:, qualisys_indices, :]
    
    best_transformation_matrix = get_best_transformation_matrix_ransac(freemocap_data= freemocap_data_for_alignment,
                                                                       qualisys_data= qualisys_data_for_alignment,
                                                                       frames_to_sample= config.frames_to_sample,
                                                                       max_iterations=config.max_iterations,
                                                                       inlier_threshold=config.inlier_threshold)
    logger.info(f"Found best transformation matrix as {best_transformation_matrix}")

    aligned_freemocap_data = apply_transformation(transformation_matrix=best_transformation_matrix,
                                                  data = freemocap_actor.body.xyz.as_array)
    
    z_offset = compute_vertical_offset(
    freemocap_data=aligned_freemocap_data,
    qualisys_data=qualisys_actor.body.xyz.as_array,
    freemocap_actor=freemocap_actor,
    qualisys_actor=qualisys_actor,
    neutral_frames=range(config.neutral_frames[0], config.neutral_frames[1]) 
)

    aligned_freemocap_data = apply_vertical_translation(
        aligned_freemocap_data,
        z_offset
    )

    return aligned_freemocap_data, best_transformation_matrix


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
        freemocap_actor.body.xyz.landmark_names.index(m)
        for m in foot_markers
    ]
    
    qualisys_indices = [
        qualisys_actor.body.xyz.landmark_names.index(m)
        for m in foot_markers
    ]

    freemocap_heights = freemocap_data[neutral_frames][:, freemocap_indices, vertical_axis]
    qualisys_heights = qualisys_data[neutral_frames][:, qualisys_indices, vertical_axis]

    freemocap_height = np.nanmedian(freemocap_heights)
    qualisys_height = np.nanmedian(qualisys_heights)

    offset = qualisys_height - freemocap_height

    return offset

def apply_vertical_translation(data, offset, vertical_axis=2):
    corrected = data.copy()
    corrected[..., vertical_axis] += offset
    return corrected