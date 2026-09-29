from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


class SpatialAlignmentConfig(BaseModel):
    markers_for_alignment: List[str]

    frames_to_sample: int = Field(
        20,
        gt=0,
        description="Number of frames to sample in each RANSAC iteration",
    )
    max_iterations: int = Field(
        20,
        gt=0,
        description="Maximum number of RANSAC iterations",
    )
    inlier_threshold: float = Field(
        50,
        gt=0,
        description="Inlier threshold for RANSAC",
    )

    neutral_frames: Optional[Tuple[int, int]] = Field(default=None)

    transform_source_tracker: Optional[str] = Field(
        default=None,
        description="Tracker whose saved spatial transform should be reused",
    )

    vertical_offset_markers: List[str] = Field(
        default_factory=lambda: ["left_heel", "right_heel"],
        description="Markers used to estimate the final vertical offset",
    )

    random_seed: int = Field(
        default=0,
        description="Random seed used for RANSAC frame sampling",
    )
