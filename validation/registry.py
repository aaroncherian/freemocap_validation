from __future__ import annotations
from typing import Dict
from validation.datatypes.data_component import DataComponent

_REGISTRY: Dict[str, DataComponent] = {}

def register(component: DataComponent) -> None:
    if component.name in _REGISTRY:
        raise KeyError(f"Component '{component.name}' already registered")
    _REGISTRY[component.name] = component

def get(name: str) -> DataComponent:
    return _REGISTRY[name]

