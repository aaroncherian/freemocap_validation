from pydantic import BaseModel, Field
from typing import List, Optional

class RMSEConfig(BaseModel):
    start_frame: Optional[int] = Field(default=None, description="First frame to include (inclusive)")
    end_frame: Optional[int] = Field(default=None, description="Last frame to include (exclusive)")
    markers_for_comparison: List[str]