from typing import Optional
from pydantic import BaseModel, Field

class TemporalAlignmentConfig(BaseModel):
    start_frame: Optional[int] = Field(default=None, description="First frame to include (inclusive)")
    end_frame: Optional[int] = Field(default=None, description="Last frame to include (exclusive)")
    qualisys_joint_weights_file: str

    # NEW (for sweep branch)
    manual_lag_frames: Optional[float] = Field(
        default=None,
        description="If set, skip GUI/auto-lag and force this lag (in FreeMoCap frames).",
    )
    interactive: bool = Field(
        default=True,
        description="If False, do not launch NiceGUI (used for automated sweeps).",
    )
