from pathlib import Path
from typing import Callable, Any, Union

class DataComponent:
    def __init__(
            self,
            name: str,
            filename: Union[str, Callable[[Path], str]],
            relative_path:str = "",
            loader: Callable = None,
            saver: Callable = None
            ):
        
        self.name = name
        self.filename = filename
        self.relative_path = relative_path
        self.loader = loader
        self.saver = saver

    def _resolve_filename(self, base_dir:Path):
        """ In case the filename is a function (like for adding the recording name to the path, for example)"""
        if callable(self.filename):
            return self.filename(base_dir)
        return self.filename

    def full_path(self, base_dir: Path) -> Path:
        return base_dir/self.relative_path/self._resolve_filename(base_dir)
    
    def exists(self, base_dir:Path) -> bool:
        return self.full_path(base_dir).exists()
    
    def load(self, base_dir:Path):
        if self.loader is None:
            raise ValueError(f'No loader defined for {self.name}')
        return self.loader(self.full_path(base_dir))
    
    def save(self, base_dir:Path, data:Any):
        if self.saver is None:
            raise ValueError(f"No saver defined for {self.name}")
        self.full_path(base_dir).parent.mkdir(exist_ok=True,parents=True)
        return self.saver(self.full_path(base_dir), data)