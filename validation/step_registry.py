
from validation.steps.temporal_alignment.step import TemporalAlignmentStep
from validation.steps.spatial_alignment.step   import SpatialAlignmentStep
from validation.steps.rmse.step                import BaseRMSEStep
from validation.steps.trc_conversion.step import TRCConversionStep

STEP_REGISTRY = {
    "TemporalAlignmentStep": TemporalAlignmentStep,
    "SpatialAlignmentStep":  SpatialAlignmentStep,
    "RMSEStep":              BaseRMSEStep,
    "TRCConversionStep":     TRCConversionStep, 
}