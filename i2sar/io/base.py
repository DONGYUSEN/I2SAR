from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from i2sar.core.enums import AcquisitionMode


@dataclass(frozen=True)
class SourceRef:
    path: str
    storage: str = "file"
    member: str | None = None

    def vsi_path(self) -> str:
        if self.storage == "zip":
            if self.member is None:
                raise ValueError("zip SourceRef requires member")
            return f"/vsizip/{self.path}/{self.member}"
        if self.storage == "tar":
            if self.member is None:
                raise ValueError("tar SourceRef requires member")
            return f"/vsitar/{self.path}/{self.member}"
        return self.path

    def to_attrs(self, prefix: str = "") -> dict[str, str]:
        attrs = {
            f"{prefix}path": self.path,
            f"{prefix}storage": self.storage,
        }
        if self.member is not None:
            attrs[f"{prefix}member"] = self.member
        return attrs


@dataclass(frozen=True)
class ParsedScene:
    scene_id: str
    sensor: str
    acquisition_mode: AcquisitionMode
    acquisition_time: str
    acquisition: dict[str, Any] = field(default_factory=dict)
    scene: dict[str, Any] = field(default_factory=dict)
    radar_grid: dict[str, Any] = field(default_factory=dict)
    orbit: dict[str, Any] = field(default_factory=dict)
    orbit_raw: dict[str, Any] | None = None
    doppler: dict[str, Any] = field(default_factory=dict)
    slc: SourceRef | None = None
    slc_attrs: dict[str, Any] = field(default_factory=dict)
    source_refs: dict[str, SourceRef] = field(default_factory=dict)


@dataclass(frozen=True)
class ImportResult:
    scene_id: str
    scene_path: Path
    sensor: str
    acquisition_mode: AcquisitionMode
    slc: SourceRef
