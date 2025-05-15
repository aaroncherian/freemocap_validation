
from validation.steps.temporal_alignment.step import TemporalAlignmentStep
from validation.steps.spatial_alignment.step   import SpatialAlignmentStep
from validation.steps.rmse.step                import BaseRMSEStep

STEP_REGISTRY = {
    "TemporalAlignmentStep": TemporalAlignmentStep,
    "SpatialAlignmentStep":  SpatialAlignmentStep,
    "RMSEStep":              BaseRMSEStep, 
}