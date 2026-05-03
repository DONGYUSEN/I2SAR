from __future__ import annotations

from pathlib import Path

import h5py

from i2sar.core.errors import SchemaError

SCHEMA_VERSION = "0.1.0"


def _require_attrs(h5: h5py.File, names: list[str]) -> None:
    missing = [name for name in names if name not in h5.attrs]
    if missing:
        raise SchemaError(f"missing required attrs: {', '.join(missing)}")


def _require_schema_version(h5: h5py.File) -> None:
    version = h5.attrs["schema_version"]
    if version != SCHEMA_VERSION:
        raise SchemaError(f"unsupported schema_version: {version}")


def _require_groups(h5: h5py.File, names: list[str]) -> None:
    missing = [name for name in names if name not in h5]
    if missing:
        raise SchemaError(f"missing required groups: {', '.join(missing)}")
    for name in names:
        if not isinstance(h5[name], h5py.Group):
            raise SchemaError(f"required group is not a group: {name}")


def validate_scene_file(path: str | Path) -> None:
    with h5py.File(path, "r") as h5:
        _require_attrs(h5, ["schema_version", "entity_type", "scene_id", "sensor", "acquisition_mode"])
        _require_schema_version(h5)
        if h5.attrs["entity_type"] != "scene":
            raise SchemaError("not a scene file")
        _require_groups(h5, ["metadata", "radar_grid", "orbit", "orbit_raw", "doppler", "slc", "derived", "provenance"])
        mode = h5.attrs["acquisition_mode"]
        if mode == "stripmap":
            _require_groups(h5, ["stripmap"])
        elif mode == "tops":
            _require_groups(h5, ["tops"])
            _require_groups(h5["tops"], ["swaths", "bursts", "burst_grid", "azimuth_steering"])
        else:
            raise SchemaError(f"unsupported acquisition_mode: {mode}")


def validate_pair_file(path: str | Path) -> None:
    with h5py.File(path, "r") as h5:
        _require_attrs(h5, ["schema_version", "entity_type", "pair_id", "master_scene_id", "slave_scene_id"])
        _require_schema_version(h5)
        if h5.attrs["entity_type"] != "pair":
            raise SchemaError("not a pair file")
        _require_groups(h5, ["metadata", "registration", "geometry", "interferometry", "unwrapping", "displacement", "provenance"])


def validate_product_file(path: str | Path) -> None:
    with h5py.File(path, "r") as h5:
        _require_attrs(h5, ["schema_version", "entity_type", "product_id", "product_type", "owner_type", "owner_id"])
        _require_schema_version(h5)
        if h5.attrs["entity_type"] != "product":
            raise SchemaError("not a product file")
        _require_groups(h5, ["grid", "provenance"])
        if h5.attrs["product_type"] == "dem":
            _require_groups(h5, ["dem"])
        if h5.attrs["product_type"] == "rtc":
            _require_groups(h5, ["rtc"])
