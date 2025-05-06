from pydantic import BaseModel
from typing import List

class RMSEConfig(BaseModel):
    markers_for_comparison: List[str]