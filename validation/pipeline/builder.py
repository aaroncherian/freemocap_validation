from pathlib import Path
import yaml
from validation.step_registry import STEP_REGISTRY
from validation.pipeline.project_config import ProjectConfig
from validation.pipeline.base import PipelineContext
from validation.pipeline.supports_variants import SupportsVariantsMixin

def build_pipeline(config_path:Path, recording_dir:Path) -> PipelineContext:
    """
    Build the pipeline from the config file and recording directory.
    """
    with open(config_path, 'r') as f:
        pipeline_config = yaml.safe_load(f)
    
    project_cfg = ProjectConfig(**pipeline_config.pop("ProjectConfig"))
    ctx = PipelineContext(recording_dir=recording_dir,
                          project_config=project_cfg)
    
    for step_name, step_parameters in pipeline_config.items():
        if step_name == "pipeline":
            continue
        ctx.put(f"{step_name}.config", step_parameters)

    step_classes = []
    for step_name in pipeline_config["pipeline"]:
        base_cls = STEP_REGISTRY[step_name]

        if issubclass(base_cls, SupportsVariantsMixin):
            step_config = ctx.get(f"{step_name}.config") or {}
            variant_names = step_config.get("variants",
                                         [list(base_cls.VARIANT_ENUM)[0].value])
            for v in variant_names:
                enum_val = base_cls.VARIANT_ENUM(v)
                step_classes.append(base_cls.make_variant(enum_val))
        else:
            step_classes.append(base_cls)
    
    return ctx, step_classes