from __future__ import annotations

import json
from typing import Any

import h5py
import numpy as np

from i2sar.io.base import ImportResult, ParsedScene
from i2sar.io.slc import load_slc_as_compound_complex
from i2sar.model import SceneInfo
from i2sar.project import Project


def _write_json_attr(group: h5py.Group, payload: dict[str, Any]) -> None:
    group.attrs["json"] = json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _write_orbit(group: h5py.Group, orbit: dict[str, Any]) -> None:
    vectors = orbit.get("stateVectors", [])
    time = np.array([float(sv.get("gpsTime", 0.0)) for sv in vectors], dtype=np.float64)
    position = np.array(
        [[sv.get("posX", 0.0), sv.get("posY", 0.0), sv.get("posZ", 0.0)] for sv in vectors],
        dtype=np.float64,
    ).reshape((-1, 3))
    velocity = np.array(
        [[sv.get("velX", 0.0), sv.get("velY", 0.0), sv.get("velZ", 0.0)] for sv in vectors],
        dtype=np.float64,
    ).reshape((-1, 3))
    for name in ["time", "position", "velocity"]:
        if name in group:
            del group[name]
    group.create_dataset("time", data=time)
    group.create_dataset("position", data=position)
    group.create_dataset("velocity", data=velocity)
    group.attrs["json"] = json.dumps(orbit, sort_keys=True, ensure_ascii=False)
    if "smoothed" in orbit:
        group.attrs["smoothed"] = bool(orbit["smoothed"])
    if "smoothing" in orbit:
        group.attrs["smoothing_json"] = json.dumps(orbit["smoothing"], sort_keys=True, ensure_ascii=False)


def _write_slc(group: h5py.Group, parsed: ParsedScene) -> None:
    if parsed.slc is None:
        raise ValueError("parsed scene requires slc SourceRef")
    for key, value in parsed.slc.to_attrs().items():
        group.attrs[key] = value
    for key, value in parsed.slc_attrs.items():
        group.attrs[key] = value
    data = load_slc_as_compound_complex(parsed.slc, parsed.slc_attrs)
    if "data" in group:
        del group["data"]
    group.create_dataset("data", data=data)
    group.attrs["hdf5_complex_layout"] = "compound_real_imag"
    if "storage_layout" in parsed.slc_attrs:
        group.attrs["source_storage_layout"] = parsed.slc_attrs["storage_layout"]


def write_parsed_scene(project: Project, parsed: ParsedScene) -> ImportResult:
    if parsed.slc is None:
        raise ValueError("parsed scene requires slc SourceRef")

    scene = SceneInfo(parsed.scene_id, parsed.sensor, parsed.acquisition_mode, parsed.acquisition_time)
    scene_path = project.create_scene(scene)
    with h5py.File(scene_path, "a") as h5:
        _write_json_attr(h5["metadata/source"], {key: ref.to_attrs() for key, ref in parsed.source_refs.items()})
        _write_json_attr(h5["metadata/acquisition"], parsed.acquisition)
        _write_json_attr(h5["metadata/quality"], {"imported": True})
        _write_json_attr(h5["radar_grid"], parsed.radar_grid)
        _write_json_attr(h5["doppler"], parsed.doppler)
        _write_json_attr(h5["derived"], parsed.scene)
        _write_orbit(h5["orbit"], parsed.orbit)
        if parsed.orbit_raw is not None:
            _write_orbit(h5["orbit_raw"], parsed.orbit_raw)
        _write_slc(h5["slc"], parsed)
    return ImportResult(parsed.scene_id, scene_path, parsed.sensor, parsed.acquisition_mode, parsed.slc)
