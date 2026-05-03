from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class LookSide(StrEnum):
    LEFT = "left"
    RIGHT = "right"

    @property
    def sign(self) -> int:
        return 1 if self == LookSide.LEFT else -1

    def __repr__(self) -> str:
        return f"LookSide.{self.name}"


class EntityType(StrEnum):
    PROJECT = "project"
    SCENE = "scene"
    PAIR = "pair"
    PRODUCT = "product"


class AcquisitionMode(StrEnum):
    STRIPMAP = "stripmap"
    TOPS = "tops"


class WorkflowType(StrEnum):
    SCENE = "scene"
    INSAR = "insar"
    RTC = "rtc"
    PROJECT = "project"


class ProductType(StrEnum):
    DEM = "dem"
    RTC = "rtc"
    LOS = "los"
    INTERFEROGRAM = "interferogram"
    EXPORT = "export"


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    INVALIDATED = "invalidated"
    BLOCKED = "blocked"
