from pydantic import BaseModel

class TrajectoryStridesConfig(BaseModel):
    frame_range: tuple[int, int]|None = None