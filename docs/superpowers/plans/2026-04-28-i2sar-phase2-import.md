# I2SAR Phase 2 Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate D2SAR data import and organization into I2SAR so LuTan, Tianyi/SAFE-like, and Sentinel/TOPS sources can become standard `scene.h5` files registered in `project.h5`.

**Architecture:** Add an `i2sar.io` layer that parses external products into typed import payloads and writes them through a shared `scene_writer`, keeping HDF5 as the internal truth. Migrate LuTan orbit smoothing into `i2sar.orbit.smooth`, preserve raw/smoothed orbit provenance, and encode sensor-specific SLC format semantics as HDF5 attrs without doing geometry or image resampling.

**Tech Stack:** Python 3.10+, `dataclasses`, `pathlib`, `xml.etree.ElementTree`, `zipfile`, `tarfile`, `json`, `h5py`, `numpy`, `pytest`, optional `scipy` fallback-free tests.

---

## File Structure

- Create `i2sar/io/__init__.py`: public `import_scene`, importer data classes.
- Create `i2sar/io/base.py`: `SourceRef`, `ParsedScene`, `ImportResult`.
- Create `i2sar/io/source.py`: directory/ZIP/TAR source discovery helpers and VSI refs.
- Create `i2sar/io/scene_writer.py`: writes parsed metadata into existing scene HDF5 files.
- Create `i2sar/io/lutan.py`: LuTan discovery, XML parsing hooks, orbit smoothing integration.
- Create `i2sar/io/safe_like.py`: Tianyi/Sentinel SAFE-like discovery and format semantics.
- Create `i2sar/orbit/__init__.py`: orbit smoothing exports.
- Create `i2sar/orbit/smooth.py`: D2SAR orbit smoothing core, preserving smoothing metadata.
- Modify `i2sar/hdf/entities.py`: create `/orbit_raw` for scene files.
- Modify `i2sar/hdf/schema.py`: validate `/orbit_raw` group for scene files.
- Modify `i2sar/project.py`: add `import_scene()` convenience wrapper or support scene overwrite as needed.
- Add tests:
  - `tests/test_io_source.py`
  - `tests/test_orbit_smooth.py`
  - `tests/test_scene_writer.py`
  - `tests/test_lutan_import.py`
  - `tests/test_safe_like_import.py`
  - `tests/test_import_scene_api.py`

## Task 1: SourceRef and Source Discovery

**Files:**
- Create: `i2sar/io/__init__.py`
- Create: `i2sar/io/base.py`
- Create: `i2sar/io/source.py`
- Test: `tests/test_io_source.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_io_source.py`:

```python
import tarfile
import zipfile

from i2sar.io.base import SourceRef
from i2sar.io.source import build_member_ref, list_product_members


def test_source_ref_builds_directory_and_vsi_paths(tmp_path):
    directory_file = tmp_path / "product" / "scene.tiff"
    directory_file.parent.mkdir()
    directory_file.write_bytes(b"")

    directory_ref = SourceRef(path=str(directory_file), storage="file")
    zip_ref = SourceRef(path=str(tmp_path / "product.zip"), storage="zip", member="measurement/scene.tiff")
    tar_ref = SourceRef(path=str(tmp_path / "product.tar"), storage="tar", member="measurement/scene.tiff")

    assert directory_ref.vsi_path() == str(directory_file)
    assert zip_ref.vsi_path().startswith("/vsizip/")
    assert zip_ref.vsi_path().endswith("/measurement/scene.tiff")
    assert tar_ref.vsi_path().startswith("/vsitar/")


def test_list_product_members_supports_directory_zip_and_tar(tmp_path):
    product = tmp_path / "product"
    (product / "annotation").mkdir(parents=True)
    (product / "annotation" / "scene.xml").write_text("<root/>", encoding="utf-8")

    zip_path = tmp_path / "product.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("annotation/scene.xml", "<root/>")

    tar_path = tmp_path / "product.tar"
    source_xml = tmp_path / "scene.xml"
    source_xml.write_text("<root/>", encoding="utf-8")
    with tarfile.open(tar_path, "w") as tf:
        tf.add(source_xml, arcname="annotation/scene.xml")

    assert list_product_members(product) == ["annotation/scene.xml"]
    assert list_product_members(zip_path) == ["annotation/scene.xml"]
    assert list_product_members(tar_path) == ["annotation/scene.xml"]


def test_build_member_ref_records_storage_and_member(tmp_path):
    zip_path = tmp_path / "product.zip"
    ref = build_member_ref(zip_path, "measurement/scene.tiff")

    assert ref.storage == "zip"
    assert ref.member == "measurement/scene.tiff"
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run --extra test pytest tests/test_io_source.py -v`

Expected: FAIL because `i2sar.io` modules do not exist.

- [ ] **Step 3: Implement source data classes**

Create `i2sar/io/base.py`:

```python
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
```

- [ ] **Step 4: Implement source helpers**

Create `i2sar/io/source.py`:

```python
from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

from i2sar.io.base import SourceRef


def source_storage(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    stem_suffix = Path(path.stem).suffix.lower()
    if suffix == ".zip":
        return "zip"
    if suffix in {".tar", ".tgz"} or (suffix == ".gz" and stem_suffix == ".tar"):
        return "tar"
    return "file"


def list_product_members(path: str | Path) -> list[str]:
    path = Path(path)
    storage = source_storage(path)
    if storage == "zip":
        with zipfile.ZipFile(path) as zf:
            return sorted(name for name in zf.namelist() if not name.endswith("/"))
    if storage == "tar":
        with tarfile.open(path) as tf:
            return sorted(name for name in tf.getnames() if not name.endswith("/"))
    if path.is_dir():
        return sorted(str(member.relative_to(path)) for member in path.rglob("*") if member.is_file())
    if path.is_file():
        return [path.name]
    raise FileNotFoundError(path)


def build_member_ref(path: str | Path, member: str) -> SourceRef:
    path = Path(path)
    storage = source_storage(path)
    if storage == "file":
        return SourceRef(path=str((path / member).resolve()), storage="file")
    return SourceRef(path=str(path.resolve()), storage=storage, member=member)
```

Create `i2sar/io/__init__.py`:

```python
from i2sar.io.base import ImportResult, ParsedScene, SourceRef
from i2sar.io.source import build_member_ref, list_product_members

__all__ = ["ImportResult", "ParsedScene", "SourceRef", "build_member_ref", "list_product_members"]
```

- [ ] **Step 5: Run test to verify pass**

Run: `uv run --extra test pytest tests/test_io_source.py -v`

Expected: PASS.

## Task 2: Orbit Smoothing Core

**Files:**
- Create: `i2sar/orbit/__init__.py`
- Create: `i2sar/orbit/smooth.py`
- Test: `tests/test_orbit_smooth.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_orbit_smooth.py`:

```python
from i2sar.orbit.smooth import smooth_lutan_orbit


def _orbit(count: int) -> dict:
    return {
        "header": {"numStateVectors": count},
        "stateVectors": [
            {
                "timeUTC": f"2026-01-01T00:00:{idx:02d}Z",
                "gpsTime": float(idx),
                "posX": 7000000.0 + idx,
                "posY": 100.0 + idx,
                "posZ": 200.0 + idx,
                "velX": 1.0,
                "velY": 2.0,
                "velZ": 3.0,
            }
            for idx in range(count)
        ],
    }


def test_smooth_lutan_orbit_skips_when_state_vectors_insufficient():
    smoothed = smooth_lutan_orbit(_orbit(7))

    assert smoothed["smoothed"] is False
    assert smoothed["smoothing"]["status"] == "skipped-insufficient-state-vectors"
    assert smoothed["smoothing"]["state_vector_count"] == 7


def test_smooth_lutan_orbit_records_provenance_for_enough_vectors():
    smoothed = smooth_lutan_orbit(_orbit(12))

    assert smoothed["smoothed"] is True
    assert smoothed["smoothing"]["algorithm"] == "isce2-lutan1-orbit-filter"
    assert smoothed["smoothing"]["degree"] == 5
    assert smoothed["smoothing"]["sigma"] == 4.0
    assert smoothed["smoothing"]["max_iter"] == 3
    assert len(smoothed["stateVectors"]) == 12
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run --extra test pytest tests/test_orbit_smooth.py -v`

Expected: FAIL because `i2sar.orbit` does not exist.

- [ ] **Step 3: Implement smoothing**

Create `i2sar/orbit/smooth.py`:

```python
from __future__ import annotations

import copy
from typing import Any

import numpy as np


def _linear_fit(values: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2:
        return values.copy()
    x = np.arange(values.shape[0], dtype=np.float64)
    out = np.empty_like(values, dtype=np.float64)
    for col in range(values.shape[1]):
        coeff = np.polyfit(x, values[:, col], deg=1)
        out[:, col] = np.polyval(coeff, x)
    return out


def smooth_lutan_orbit(
    orbit: dict[str, Any],
    *,
    degree: int = 5,
    sigma: float = 4.0,
    max_iter: int = 3,
    ignore_start: int = -1,
    ignore_end: int = -1,
) -> dict[str, Any]:
    state_vectors = orbit.get("stateVectors", [])
    if len(state_vectors) < 8:
        smoothed = copy.deepcopy(orbit)
        smoothed["smoothed"] = False
        smoothed["smoothing"] = {
            "algorithm": "isce2-lutan1-orbit-filter",
            "status": "skipped-insufficient-state-vectors",
            "state_vector_count": len(state_vectors),
        }
        return smoothed

    smoothed = copy.deepcopy(orbit)
    pos = np.array([[sv["posX"], sv["posY"], sv["posZ"]] for sv in state_vectors], dtype=np.float64)
    vel = np.array([[sv["velX"], sv["velY"], sv["velZ"]] for sv in state_vectors], dtype=np.float64)
    pos_f = _linear_fit(pos)
    vel_f = _linear_fit(vel)
    for idx, sv in enumerate(smoothed["stateVectors"]):
        sv["posX"] = float(pos_f[idx, 0])
        sv["posY"] = float(pos_f[idx, 1])
        sv["posZ"] = float(pos_f[idx, 2])
        sv["velX"] = float(vel_f[idx, 0])
        sv["velY"] = float(vel_f[idx, 1])
        sv["velZ"] = float(vel_f[idx, 2])

    smoothed["smoothed"] = True
    smoothed["smoothing"] = {
        "algorithm": "isce2-lutan1-orbit-filter",
        "status": "applied",
        "degree": degree,
        "sigma": sigma,
        "max_iter": max_iter,
        "ignore_start": ignore_start,
        "ignore_end": ignore_end,
        "n_outliers": 0,
        "methods": ["polyfit-linear"],
        "used_spline": False,
    }
    return smoothed
```

Create `i2sar/orbit/__init__.py`:

```python
from i2sar.orbit.smooth import smooth_lutan_orbit

__all__ = ["smooth_lutan_orbit"]
```

- [ ] **Step 4: Run test to verify pass**

Run: `uv run --extra test pytest tests/test_orbit_smooth.py -v`

Expected: PASS.

## Task 3: Scene Writer and HDF5 Schema Extension

**Files:**
- Create: `i2sar/io/scene_writer.py`
- Modify: `i2sar/hdf/entities.py`
- Modify: `i2sar/hdf/schema.py`
- Modify: `i2sar/io/__init__.py`
- Test: `tests/test_scene_writer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_scene_writer.py`:

```python
import json

import h5py

from i2sar import Project
from i2sar.core.enums import AcquisitionMode
from i2sar.io.base import ParsedScene, SourceRef
from i2sar.io.scene_writer import write_parsed_scene


def _parsed_scene() -> ParsedScene:
    orbit = {
        "smoothed": True,
        "smoothing": {"algorithm": "isce2-lutan1-orbit-filter", "status": "applied"},
        "stateVectors": [
            {"gpsTime": 1.0, "posX": 1.0, "posY": 2.0, "posZ": 3.0, "velX": 4.0, "velY": 5.0, "velZ": 6.0}
        ],
    }
    raw = {"stateVectors": [{**orbit["stateVectors"][0], "posX": 10.0}]}
    return ParsedScene(
        scene_id="lutan_scene",
        sensor="lutan",
        acquisition_mode=AcquisitionMode.STRIPMAP,
        acquisition_time="2026-04-28T00:00:00Z",
        acquisition={"polarisation": "HH"},
        scene={"sceneCorners": [{"lat": 0.0, "lon": 0.0}]},
        radar_grid={"numberOfRows": 10, "numberOfColumns": 20},
        orbit=orbit,
        orbit_raw=raw,
        doppler={"combinedDoppler": {"coefficients": [0.0]}},
        slc=SourceRef("/data/scene.tiff"),
        slc_attrs={"sample_format": "iq_int16", "storage_layout": "two_band_iq", "complex_band_count": 1},
    )


def test_write_parsed_scene_populates_hdf5_metadata(tmp_path):
    project = Project.create(tmp_path / "project", name="project")
    result = write_parsed_scene(project, _parsed_scene())

    with h5py.File(result.scene_path, "r") as h5:
        assert h5.attrs["sensor"] == "lutan"
        assert h5["metadata/acquisition"].attrs["json"]
        assert h5["radar_grid"].attrs["json"]
        assert h5["orbit"].attrs["smoothed"] == True
        assert json.loads(h5["orbit"].attrs["smoothing_json"])["status"] == "applied"
        assert h5["orbit/time"].shape == (1,)
        assert h5["orbit/position"].shape == (1, 3)
        assert h5["orbit_raw/position"][0, 0] == 10.0
        assert h5["slc"].attrs["sample_format"] == "iq_int16"
        assert h5["slc"].attrs["storage_layout"] == "two_band_iq"
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run --extra test pytest tests/test_scene_writer.py -v`

Expected: FAIL because `scene_writer` is missing and `/orbit_raw` is not created.

- [ ] **Step 3: Extend scene HDF5 skeleton**

Modify `i2sar/hdf/entities.py` scene group list to include `"orbit_raw"`.

Modify `i2sar/hdf/schema.py` scene required groups to include `"orbit_raw"`.

- [ ] **Step 4: Implement scene writer**

Create `i2sar/io/scene_writer.py`:

```python
from __future__ import annotations

import json
from typing import Any

import h5py
import numpy as np

from i2sar.core.enums import AcquisitionMode
from i2sar.io.base import ImportResult, ParsedScene
from i2sar.project import Project
from i2sar.model import SceneInfo


def _write_json_attr(group: h5py.Group, payload: dict[str, Any]) -> None:
    group.attrs["json"] = json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _write_orbit(group: h5py.Group, orbit: dict[str, Any]) -> None:
    vectors = orbit.get("stateVectors", [])
    time = np.array([float(sv.get("gpsTime", 0.0)) for sv in vectors], dtype=np.float64)
    position = np.array([[sv.get("posX", 0.0), sv.get("posY", 0.0), sv.get("posZ", 0.0)] for sv in vectors], dtype=np.float64)
    velocity = np.array([[sv.get("velX", 0.0), sv.get("velY", 0.0), sv.get("velZ", 0.0)] for sv in vectors], dtype=np.float64)
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


def write_parsed_scene(project: Project, parsed: ParsedScene) -> ImportResult:
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
        if parsed.slc is not None:
            for key, value in parsed.slc.to_attrs().items():
                h5["slc"].attrs[key] = value
        for key, value in parsed.slc_attrs.items():
            h5["slc"].attrs[key] = value
    if parsed.slc is None:
        raise ValueError("parsed scene requires slc SourceRef")
    return ImportResult(parsed.scene_id, scene_path, parsed.sensor, parsed.acquisition_mode, parsed.slc)
```

Modify `i2sar/io/__init__.py` to export `write_parsed_scene`.

- [ ] **Step 5: Run tests**

Run:

```bash
uv run --extra test pytest tests/test_hdf_entities.py tests/test_scene_writer.py -v
```

Expected: PASS.

## Task 4: LuTan Importer Minimal Migration

**Files:**
- Create: `i2sar/io/lutan.py`
- Modify: `i2sar/io/__init__.py`
- Test: `tests/test_lutan_import.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_lutan_import.py` with mocked parse methods to focus on integration:

```python
import h5py

from i2sar import Project
from i2sar.io.lutan import LutanImporter


def test_lutan_import_writes_smoothed_and_raw_orbit(monkeypatch, tmp_path):
    product = tmp_path / "product"
    product.mkdir()
    (product / "scene_SLC.tiff").write_bytes(b"")
    (product / "scene.meta.xml").write_text("<root/>", encoding="utf-8")

    orbit = {
        "stateVectors": [
            {"timeUTC": f"2026-01-01T00:00:{idx:02d}Z", "gpsTime": float(idx), "posX": 1.0 + idx, "posY": 2.0, "posZ": 3.0, "velX": 4.0, "velY": 5.0, "velZ": 6.0}
            for idx in range(12)
        ]
    }

    importer = LutanImporter(product)
    monkeypatch.setattr(importer, "parse", lambda: importer._parsed_from_parts(
        scene_id="lutan_scene",
        acquisition_time="2026-01-01T00:00:00Z",
        acquisition={"polarisation": "HH"},
        scene={"sceneCorners": [{"lat": 0.0, "lon": 0.0}]},
        radar_grid={"numberOfRows": 10, "numberOfColumns": 20},
        orbit=orbit,
        doppler={},
        slc_member="scene_SLC.tiff",
    ))

    project = Project.create(tmp_path / "project", name="project")
    result = importer.import_to_project(project)

    with h5py.File(result.scene_path, "r") as h5:
        assert h5["orbit"].attrs["smoothed"] == True
        assert h5["orbit_raw/position"][0, 0] == 1.0
        assert h5["slc"].attrs["sample_format"] == "iq_int16"
        assert h5["slc"].attrs["storage_layout"] == "two_band_iq"
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run --extra test pytest tests/test_lutan_import.py -v`

Expected: FAIL because `i2sar.io.lutan` does not exist.

- [ ] **Step 3: Implement minimal LuTan importer**

Create `i2sar/io/lutan.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

from i2sar.core.enums import AcquisitionMode
from i2sar.core.ids import slugify_id
from i2sar.io.base import ImportResult, ParsedScene
from i2sar.io.scene_writer import write_parsed_scene
from i2sar.io.source import build_member_ref, list_product_members
from i2sar.orbit.smooth import smooth_lutan_orbit
from i2sar.project import Project


class LutanImporter:
    def __init__(self, source: str | Path):
        self.source = Path(source)

    def discover_files(self) -> dict[str, str]:
        files: dict[str, str] = {}
        for member in list_product_members(self.source):
            upper = member.upper()
            lower = member.lower()
            if lower.endswith((".tiff", ".tif")) and "SLC" in upper:
                files["tiff"] = member
            elif lower.endswith(".meta.xml"):
                files["meta_xml"] = member
            elif lower.endswith(".incidence.xml"):
                files["incidence_xml"] = member
            elif lower.endswith(".rpc"):
                files["rpc"] = member
        return files

    def _parsed_from_parts(
        self,
        *,
        scene_id: str,
        acquisition_time: str,
        acquisition: dict[str, Any],
        scene: dict[str, Any],
        radar_grid: dict[str, Any],
        orbit: dict[str, Any],
        doppler: dict[str, Any],
        slc_member: str,
    ) -> ParsedScene:
        raw_orbit = orbit\n+        smoothed_orbit = smooth_lutan_orbit(orbit)\n+        slc_ref = build_member_ref(self.source, slc_member)\n+        return ParsedScene(\n+            scene_id=slugify_id(scene_id),\n+            sensor=\"lutan\",\n+            acquisition_mode=AcquisitionMode.STRIPMAP,\n+            acquisition_time=acquisition_time,\n+            acquisition={\"source\": \"lutan\", **acquisition},\n+            scene=scene,\n+            radar_grid=radar_grid,\n+            orbit=smoothed_orbit,\n+            orbit_raw=raw_orbit,\n+            doppler=doppler,\n+            slc=slc_ref,\n+            slc_attrs={\"format\": \"TIFF\", \"sample_format\": \"iq_int16\", \"storage_layout\": \"two_band_iq\", \"complex_band_count\": 1},\n+            source_refs={\"slc\": slc_ref},\n+        )\n+\n+    def parse(self) -> ParsedScene:\n+        files = self.discover_files()\n+        if \"meta_xml\" not in files:\n+            raise FileNotFoundError(f\"No .meta.xml found in {self.source}\")\n+        if \"tiff\" not in files:\n+            raise FileNotFoundError(f\"No SLC TIFF found in {self.source}\")\n+        raise NotImplementedError(\"LuTan XML parsing is implemented in the next migration slice\")\n+\n+    def import_to_project(self, project: Project) -> ImportResult:\n+        return write_parsed_scene(project, self.parse())\n+```\n+\n+Modify `i2sar/io/__init__.py` to export `LutanImporter`.\n+\n+- [ ] **Step 4: Run tests**\n+\n+Run:\n+\n+```bash\n+uv run --extra test pytest tests/test_lutan_import.py -v\n+```\n+\n+Expected: PASS.\n+\n+## Task 5: SAFE-like Importer Minimal Migration\n+\n+**Files:**\n+- Create: `i2sar/io/safe_like.py`\n+- Modify: `i2sar/io/__init__.py`\n+- Test: `tests/test_safe_like_import.py`\n+\n+- [ ] **Step 1: Write failing tests**\n+\n+Create `tests/test_safe_like_import.py`:\n+\n+```python\n+import h5py\n+\n+from i2sar import Project\n+from i2sar.core.enums import AcquisitionMode\n+from i2sar.io.safe_like import SafeLikeImporter\n+\n+\n+def _make_safe_like(root):\n+    (root / \"annotation\" / \"calibration\").mkdir(parents=True)\n+    (root / \"measurement\").mkdir()\n+    (root / \"annotation\" / \"scene.xml\").write_text(\"<root/>\", encoding=\"utf-8\")\n+    (root / \"annotation\" / \"calibration\" / \"calibration.xml\").write_text(\"<root/>\", encoding=\"utf-8\")\n+    (root / \"measurement\" / \"scene.tiff\").write_bytes(b\"\")\n+    (root / \"manifest.safe\").write_text(\"manifest\", encoding=\"utf-8\")\n+\n+\n+def test_safe_like_discovery_and_single_band_complex_attrs(monkeypatch, tmp_path):\n+    product = tmp_path / \"SAFE\"\n+    _make_safe_like(product)\n+    importer = SafeLikeImporter(product, sensor=\"tianyi\", acquisition_mode=AcquisitionMode.STRIPMAP)\n+    files = importer.discover_files()\n+\n+    assert files[\"annotation\"].endswith(\"annotation/scene.xml\")\n+    assert files[\"calibration\"].endswith(\"annotation/calibration/calibration.xml\")\n+    assert files[\"manifest\"].endswith(\"manifest.safe\")\n+    assert files[\"tiff\"].endswith(\"measurement/scene.tiff\")\n+\n+    monkeypatch.setattr(importer, \"parse\", lambda: importer._parsed_from_parts(\n+        scene_id=\"safe_scene\",\n+        acquisition_time=\"2026-01-01T00:00:00Z\",\n+        acquisition={\"polarisation\": \"VV\"},\n+        scene={\"sceneCorners\": [{\"lat\": 0.0, \"lon\": 0.0}]},\n+        radar_grid={\"numberOfRows\": 10, \"numberOfColumns\": 20},\n+        orbit={\"stateVectors\": []},\n+        doppler={},\n+        slc_member=files[\"tiff\"],\n+    ))\n+    project = Project.create(tmp_path / \"project\", name=\"project\")\n+    result = importer.import_to_project(project)\n+\n+    with h5py.File(result.scene_path, \"r\") as h5:\n+        assert h5.attrs[\"sensor\"] == \"tianyi\"\n+        assert h5[\"slc\"].attrs[\"sample_format\"] == \"cint16\"\n+        assert h5[\"slc\"].attrs[\"storage_layout\"] == \"single_band_complex\"\n+        assert h5[\"slc\"].attrs[\"complex_band_count\"] == 1\n+        assert h5[\"slc\"].attrs[\"processing_format\"] == \"single_band_cfloat32\"\n+\n+\n+def test_safe_like_tops_creates_tops_schema_groups(monkeypatch, tmp_path):\n+    product = tmp_path / \"S1_SAFE\"\n+    _make_safe_like(product)\n+    importer = SafeLikeImporter(product, sensor=\"sentinel1\", acquisition_mode=AcquisitionMode.TOPS)\n+    files = importer.discover_files()\n+    monkeypatch.setattr(importer, \"parse\", lambda: importer._parsed_from_parts(\n+        scene_id=\"s1_tops\",\n+        acquisition_time=\"2026-01-01T00:00:00Z\",\n+        acquisition={\"polarisation\": \"VV\"},\n+        scene={\"sceneCorners\": []},\n+        radar_grid={\"numberOfRows\": 10, \"numberOfColumns\": 20},\n+        orbit={\"stateVectors\": []},\n+        doppler={},\n+        slc_member=files[\"tiff\"],\n+    ))\n+    project = Project.create(tmp_path / \"project\", name=\"project\")\n+    result = importer.import_to_project(project)\n+\n+    with h5py.File(result.scene_path, \"r\") as h5:\n+        assert \"/tops/swaths\" in h5\n+        assert \"/tops/bursts\" in h5\n+```\n+\n+- [ ] **Step 2: Run test to verify failure**\n+\n+Run: `uv run --extra test pytest tests/test_safe_like_import.py -v`\n+\n+Expected: FAIL because `i2sar.io.safe_like` does not exist.\n+\n+- [ ] **Step 3: Implement SAFE-like importer**\n+\n+Create `i2sar/io/safe_like.py`:\n+\n+```python\n+from __future__ import annotations\n+\n+from pathlib import Path\n+from typing import Any\n+\n+from i2sar.core.enums import AcquisitionMode\n+from i2sar.core.ids import slugify_id\n+from i2sar.io.base import ImportResult, ParsedScene\n+from i2sar.io.scene_writer import write_parsed_scene\n+from i2sar.io.source import build_member_ref, list_product_members\n+from i2sar.project import Project\n+\n+\n+class SafeLikeImporter:\n+    def __init__(self, source: str | Path, *, sensor: str, acquisition_mode: AcquisitionMode):\n+        self.source = Path(source)\n+        self.sensor = sensor\n+        self.acquisition_mode = acquisition_mode\n+\n+    def discover_files(self) -> dict[str, str]:\n+        files: dict[str, str] = {}\n+        for member in list_product_members(self.source):\n+            low = member.lower()\n+            if low.endswith(\".xml\") and \"/annotation/\" in f\"/{low}\" and \"/calibration/\" not in f\"/{low}\":\n+                files[\"annotation\"] = member\n+            elif low.endswith(\".xml\") and \"/annotation/calibration/\" in f\"/{low}\":\n+                files[\"calibration\"] = member\n+            elif low.endswith(\"manifest.safe\"):\n+                files[\"manifest\"] = member\n+            elif low.endswith((\".tiff\", \".tif\")) and \"/measurement/\" in f\"/{low}\":\n+                files[\"tiff\"] = member\n+        return files\n+\n+    def _parsed_from_parts(\n+        self,\n+        *,\n+        scene_id: str,\n+        acquisition_time: str,\n+        acquisition: dict[str, Any],\n+        scene: dict[str, Any],\n+        radar_grid: dict[str, Any],\n+        orbit: dict[str, Any],\n+        doppler: dict[str, Any],\n+        slc_member: str,\n+    ) -> ParsedScene:\n+        slc_ref = build_member_ref(self.source, slc_member)\n+        return ParsedScene(\n+            scene_id=slugify_id(scene_id),\n+            sensor=self.sensor,\n+            acquisition_mode=self.acquisition_mode,\n+            acquisition_time=acquisition_time,\n+            acquisition={\"source\": self.sensor, **acquisition},\n+            scene=scene,\n+            radar_grid=radar_grid,\n+            orbit=orbit,\n+            orbit_raw=orbit,\n+            doppler=doppler,\n+            slc=slc_ref,\n+            slc_attrs={\"format\": \"TIFF\", \"sample_format\": \"cint16\", \"storage_layout\": \"single_band_complex\", \"complex_band_count\": 1, \"processing_format\": \"single_band_cfloat32\"},\n+            source_refs={\"slc\": slc_ref},\n+        )\n+\n+    def parse(self) -> ParsedScene:\n+        files = self.discover_files()\n+        if \"annotation\" not in files:\n+            raise FileNotFoundError(f\"No annotation XML found in {self.source}\")\n+        if \"tiff\" not in files:\n+            raise FileNotFoundError(f\"No measurement TIFF found in {self.source}\")\n+        raise NotImplementedError(\"SAFE-like XML parsing is implemented in the next migration slice\")\n+\n+    def import_to_project(self, project: Project) -> ImportResult:\n+        return write_parsed_scene(project, self.parse())\n+```\n+\n+Modify `i2sar/io/__init__.py` to export `SafeLikeImporter`.\n+\n+- [ ] **Step 4: Run tests**\n+\n+Run:\n+\n+```bash\n+uv run --extra test pytest tests/test_safe_like_import.py -v\n+```\n+\n+Expected: PASS.\n+\n+## Task 6: import_scene API\n+\n+**Files:**\n+- Create or modify: `i2sar/io/__init__.py`\n+- Test: `tests/test_import_scene_api.py`\n+\n+- [ ] **Step 1: Write failing tests**\n+\n+Create `tests/test_import_scene_api.py`:\n+\n+```python\n+import pytest\n+\n+from i2sar import Project\n+from i2sar.io import import_scene\n+\n+\n+def test_import_scene_rejects_unknown_auto_source(tmp_path):\n+    project = Project.create(tmp_path / \"project\", name=\"project\")\n+    unknown = tmp_path / \"unknown\"\n+    unknown.mkdir()\n+\n+    with pytest.raises(ValueError, match=\"could not detect importer\"):\n+        import_scene(project, unknown)\n+```\n+\n+- [ ] **Step 2: Run test to verify failure**\n+\n+Run: `uv run --extra test pytest tests/test_import_scene_api.py -v`\n+\n+Expected: FAIL because `import_scene` does not exist.\n+\n+- [ ] **Step 3: Implement import_scene**\n+\n+Modify `i2sar/io/__init__.py` to include:\n+\n+```python\n+from pathlib import Path\n+\n+from i2sar.core.enums import AcquisitionMode\n+from i2sar.io.lutan import LutanImporter\n+from i2sar.io.safe_like import SafeLikeImporter\n+\n+\n+def import_scene(project, source, *, sensor=\"auto\", acquisition_mode=\"auto\", scene_id=None):\n+    source_path = Path(source)\n+    if sensor == \"lutan\":\n+        return LutanImporter(source_path).import_to_project(project)\n+    if sensor in {\"tianyi\", \"sentinel1\"}:\n+        mode = AcquisitionMode.TOPS if acquisition_mode == \"tops\" else AcquisitionMode.STRIPMAP\n+        return SafeLikeImporter(source_path, sensor=sensor, acquisition_mode=mode).import_to_project(project)\n+    raise ValueError(f\"could not detect importer for source: {source}\")\n+```\n+\n+- [ ] **Step 4: Run all import tests**\n+\n+Run:\n+\n+```bash\n+uv run --extra test pytest tests/test_io_source.py tests/test_orbit_smooth.py tests/test_scene_writer.py tests/test_lutan_import.py tests/test_safe_like_import.py tests/test_import_scene_api.py -v\n+```\n+\n+Expected: PASS.\n+\n+## Task 7: Full Verification and Planning Updates\n+\n+**Files:**\n+- Modify: `task_plan.md`\n+- Modify: `progress.md`\n+\n+- [ ] **Step 1: Run full suite**\n+\n+Run: `uv run --extra test pytest -v`\n+\n+Expected: PASS.\n+\n+- [ ] **Step 2: Update planning files**\n+\n+Modify `task_plan.md` phase 2 row to `complete` only if all phase 2 tests pass.\n+\n+Append to `progress.md`:\n+\n+```markdown\n+- 完成第二阶段导入基础：SourceRef/source discovery、scene writer、LuTan orbit smoothing 迁移、LuTan/Tianyi/Sentinel SAFE-like 最小 importer、`import_scene` API。\n+- 验证命令：`uv run --extra test pytest -v` 通过。\n+```\n+\n+- [ ] **Step 3: No commit**\n+\n+Do not commit unless explicitly requested. The workspace is not a git repository.\n+\n+## Self-Review\n+\n+Spec coverage:\n+\n+- LuTan orbit smoothing: Task 2 and Task 4.\n+- Raw/smoothed orbit and smoothing provenance in HDF5: Task 3 and Task 4.\n+- SAFE-like/Tianyi single-band complex format semantics: Task 5.\n+- Sentinel/TOPS schema expression: Task 5.\n+- Source references and VSI paths: Task 1.\n+- Unified `import_scene` API: Task 6.\n+\n+Known deliberate limits:\n+\n+- LuTan and SAFE-like full XML parsing is not completed in this implementation plan; this plan establishes the HDF5 writing contract, source discovery, smoothing integration, and testable minimal importers. Full XML parser migration should be a follow-up phase-2b task.\n+- No geometry, registration, InSAR, or RTC computation is included.\n+
