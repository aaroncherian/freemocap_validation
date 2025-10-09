from pydantic import BaseModel

class StepFinderConfig(BaseModel):
    sampling_rate: float
    min_event_interval_seconds: float 
    