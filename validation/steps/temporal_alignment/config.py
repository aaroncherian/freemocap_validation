from typing import Optional
from pydantic import BaseModel, Field

class TemporalAlignmentConfig(BaseModel):
    start_frame: Optional[int] = Field(default=None, description="First frame to include (inclusive)")
    end_frame: Optional[int] = Field(default=None, description="Last frame to include (exclusive)")