from validation.pipeline.context import PipelineContext
class FrameLoopClause:
    def __init__(self, context:PipelineContext, cls_name:str):
        self.loop_over_frames = False
        config = context.get(f"{cls_name}.config")
        
        if not config:
            return
        self.loop_over_frames = bool(config.get("loop_over_frames", False))