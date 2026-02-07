from dataclasses import dataclass, field
from pathlib import Path
from validation.pipeline.project_config import ProjectConfig
from typing import Any, Optional

@dataclass
class PipelineContext:
    recording_dir: Path
    project_config: ProjectConfig
    backpack: dict[str, Any] = field(default_factory=dict)

    output_root: Optional[Path] = None

    def get(self, name: str) -> Any:
        return self.backpack.get(name)

    def put(self, name: str, value: Any) -> Any:
        self.backpack[name] = value

    @property
    def _root(self) -> Path:
        return self.output_root or self.recording_dir

    @property
    def data_component_context(self) -> dict:
        # This is what DataComponents use to locate files.
        return {
            "tracker": self.project_config.freemocap_tracker,
            "recording_name": self.recording_dir.stem,
        }

    @property
    def freemocap_path(self) -> Path:
        return self._root / "validation" / self.project_config.freemocap_tracker

    @property
    def qualisys_path(self) -> Path:
        return self._root / "validation" / "qualisys"

    @property
    def conditions(self) -> dict:
        return self.project_config.conditions or {}
