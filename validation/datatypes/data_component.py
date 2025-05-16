from pathlib import Path
from typing import Callable, Any, Union
from copy import copy
class DataComponent:
    def __init__(
            self,
            name: str,
            filename: str = "",
            relative_path:str = "",
            loader: Callable = None,
            saver: Callable = None
            ):
        
        self.name = name
        self.filename = filename
        self.relative_path = relative_path
        self.loader = loader
        self.saver = saver
    
    def clone_with_prefix(self, prefix:str) -> "DataComponent":
        """Return a shallow copy whose *name* & *filename* are prefixed."""
        new_dc = copy(self)
        new_dc.name = f"{prefix}_{self.name}"
        new_dc.filename = f"{prefix}_{self.filename}"
        return new_dc

    def _resolve_filename(self, base_dir:Path):
        """ In case the filename is a function (like for adding the recording name to the path, for example)"""
        if callable(self.filename):
            return self.filename(base_dir)
        return self.filename

    def _resolve_relative_path(self, **params):
        template_string = self.relative_path
        return template_string.format(**params)

    def full_path(self, base_dir: Path, **params) -> Path:
        relative_path_formatted = self.relative_path.format(**params)
        filename_formatted = self.filename.format(**params)
        return base_dir/relative_path_formatted/filename_formatted
    
    def exists(self, base_dir:Path, **params) -> bool:
        return self.full_path(base_dir, **params).exists()
    
    def load(self, base_dir:Path, **params):
        if self.loader is None:
            raise ValueError(f'No loader defined for {self.name}')
        return self.loader(self.full_path(base_dir, **params))
    
    def save(self, base_dir:Path, data:Any, **params):
        if self.saver is None:
            raise ValueError(f"No saver defined for {self.name}")
        self.full_path(base_dir, **params).parent.mkdir(exist_ok=True,parents=True)
        return self.saver(self.full_path(base_dir,**params), data)