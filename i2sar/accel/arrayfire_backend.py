from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ArrayFireBackend:
    available: bool = False
    reason: str = ""
    module: Any = None

    def __init__(self):
        status = _load_arrayfire()
        object.__setattr__(self, "available", status[0])
        object.__setattr__(self, "reason", status[1])
        object.__setattr__(self, "module", status[2])


@lru_cache(maxsize=1)
def _load_arrayfire() -> tuple[bool, str, Any]:
    try:
        import arrayfire as af
    except Exception as exc:
        return False, str(exc), None

    try:
        af.eval(af.randu(1))
        af.sync()
    except Exception as exc:
        return False, str(exc), None

    return True, "", af


def arrayfire_available() -> bool:
    return _load_arrayfire()[0]


def asarray(af: Any, value: np.ndarray):
    return af.from_ndarray(np.asarray(value))


def to_numpy(value: Any) -> np.ndarray:
    return np.asarray(value.to_ndarray())
