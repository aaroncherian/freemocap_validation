from pydantic import BaseModel

class StrideSeparatorConfig(BaseModel):
    frame_range: tuple[int, int]|None = None