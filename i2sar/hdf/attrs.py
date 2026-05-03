from __future__ import annotations

from enum import Enum
from typing import Any


def attr_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return value


def set_attrs(obj: Any, attrs: dict[str, Any]) -> None:
    for key, value in attrs.items():
        obj.attrs[key] = attr_value(value)
