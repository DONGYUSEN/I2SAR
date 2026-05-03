from pathlib import Path

from i2sar.core.enums import AcquisitionMode
from i2sar.io.base import ImportResult, ParsedScene, SourceRef
from i2sar.io.lutan import LutanImporter
from i2sar.io.safe_like import SafeLikeImporter
from i2sar.io.scene_writer import write_parsed_scene
from i2sar.io.slc import load_slc_as_compound_complex
from i2sar.io.source import build_member_ref, list_product_members


def _coerce_acquisition_mode(value: str | AcquisitionMode) -> AcquisitionMode:
    if isinstance(value, AcquisitionMode):
        return value
    return AcquisitionMode(value)


def _default_acquisition_mode(sensor: str) -> AcquisitionMode:
    if sensor == "sentinel1":
        return AcquisitionMode.TOPS
    return AcquisitionMode.STRIPMAP


def import_scene(project, source, *, sensor: str = "auto", acquisition_mode: str | AcquisitionMode = "auto"):
    source_path = Path(source)
    if sensor == "lutan":
        return LutanImporter(source_path).import_to_project(project)
    if sensor in {"tianyi", "sentinel1"}:
        mode = _default_acquisition_mode(sensor) if acquisition_mode == "auto" else _coerce_acquisition_mode(acquisition_mode)
        return SafeLikeImporter(source_path, sensor=sensor, acquisition_mode=mode).import_to_project(project)
    raise ValueError(f"could not detect importer for source: {source}")

__all__ = [
    "ImportResult",
    "LutanImporter",
    "ParsedScene",
    "SafeLikeImporter",
    "SourceRef",
    "build_member_ref",
    "import_scene",
    "list_product_members",
    "load_slc_as_compound_complex",
    "write_parsed_scene",
]
