# I2SAR Phase 1 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the I2SAR phase-1 foundation: package skeleton, core data models, HDF5 schema/read-write/validation, project indexing, empty workflows, provenance, resume/skipped logic, and a CLI thin wrapper.

**Architecture:** Implement a standard Python package centered on typed models and HDF5 entity files. `Project` manages `project.h5` as a lightweight index; `Scene`, `Pair`, and `Product` live in separate HDF5 files. Workflow tasks call Python APIs only and write provenance plus task hashes for resumable execution.

**Tech Stack:** Python 3.10+, `dataclasses`, `enum`, `hashlib`, `json`, `pathlib`, `h5py`, `numpy`, `pytest`, `argparse`.

---

## File Structure

- Create `pyproject.toml`: package metadata, pytest config, console script.
- Create `i2sar/__init__.py`: public API exports and version.
- Create `i2sar/core/enums.py`: stable enums for entity type, acquisition mode, workflow type, task status, product type.
- Create `i2sar/core/ids.py`: deterministic ID helpers.
- Create `i2sar/core/time.py`: UTC timestamp helper.
- Create `i2sar/core/errors.py`: domain exceptions.
- Create `i2sar/model/entities.py`: `ProjectInfo`, `SceneInfo`, `PairInfo`, `ProductInfo`.
- Create `i2sar/hdf/attrs.py`: HDF5 attribute helpers.
- Create `i2sar/hdf/schema.py`: schema constants and validators.
- Create `i2sar/hdf/entities.py`: create/read scene, pair, product HDF5 files.
- Create `i2sar/project.py`: `Project` class and `project.h5` index management.
- Create `i2sar/workflow/task.py`: `TaskSpec`, `TaskRecord`, task hash.
- Create `i2sar/workflow/runner.py`: empty task runner and resume/skipped logic.
- Create `i2sar/workflow/presets.py`: empty `SceneWorkflow`, `InSARWorkflow`, `RTCWorkflow`.
- Create `i2sar/cli/main.py`: CLI thin wrapper around Python API.
- Create tests under `tests/` matching each module.

## Task 1: Package Skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `i2sar/__init__.py`
- Create: `i2sar/core/__init__.py`
- Create: `i2sar/hdf/__init__.py`
- Create: `i2sar/model/__init__.py`
- Create: `i2sar/workflow/__init__.py`
- Create: `i2sar/cli/__init__.py`
- Test: `tests/test_package_import.py`

- [ ] **Step 1: Write the import test**

Create `tests/test_package_import.py`:

```python
def test_package_imports():
    import i2sar

    assert i2sar.__version__ == "0.1.0"
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `pytest tests/test_package_import.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'i2sar'` or missing `__version__`.

- [ ] **Step 3: Add package metadata**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "i2sar"
version = "0.1.0"
description = "HDF5-centered SAR/InSAR processing foundation"
requires-python = ">=3.10"
dependencies = [
  "h5py",
  "numpy",
]

[project.optional-dependencies]
test = ["pytest"]

[project.scripts]
i2sar = "i2sar.cli.main:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

Create `i2sar/__init__.py`:

```python
"""I2SAR SAR/InSAR processing foundation."""

__version__ = "0.1.0"
```

Create empty package files:

```python
"""Package module."""
```

for:

- `i2sar/core/__init__.py`
- `i2sar/hdf/__init__.py`
- `i2sar/model/__init__.py`
- `i2sar/workflow/__init__.py`
- `i2sar/cli/__init__.py`

- [ ] **Step 4: Run the test and verify it passes**

Run: `pytest tests/test_package_import.py -v`

Expected: PASS.

## Task 2: Core Enums and Entity Models

**Files:**
- Create: `i2sar/core/enums.py`
- Create: `i2sar/core/errors.py`
- Create: `i2sar/core/time.py`
- Create: `i2sar/core/ids.py`
- Create: `i2sar/model/entities.py`
- Modify: `i2sar/model/__init__.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write model tests**

Create `tests/test_models.py`:

```python
from i2sar.core.enums import AcquisitionMode, EntityType, ProductType, WorkflowType
from i2sar.core.ids import make_pair_id, slugify_id
from i2sar.model import PairInfo, ProductInfo, SceneInfo


def test_scene_info_supports_stripmap_and_tops():
    strip = SceneInfo(
        scene_id="scene-strip",
        sensor="sentinel1",
        acquisition_mode=AcquisitionMode.STRIPMAP,
        acquisition_time="2026-04-28T00:00:00Z",
    )
    tops = SceneInfo(
        scene_id="scene-tops",
        sensor="sentinel1",
        acquisition_mode=AcquisitionMode.TOPS,
        acquisition_time="2026-04-28T00:01:00Z",
    )

    assert strip.entity_type is EntityType.SCENE
    assert tops.acquisition_mode is AcquisitionMode.TOPS


def test_pair_and_product_info():
    pair = PairInfo(pair_id="master__slave", master_scene_id="master", slave_scene_id="slave")
    product = ProductInfo(
        product_id="rtc-master",
        product_type=ProductType.RTC,
        owner_type=EntityType.SCENE,
        owner_id="master",
    )

    assert pair.workflow_type is WorkflowType.INSAR
    assert product.product_type is ProductType.RTC


def test_id_helpers_are_stable():
    assert slugify_id("Scene 01/HH") == "Scene_01_HH"
    assert make_pair_id("master scene", "slave scene") == "master_scene__slave_scene"
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_models.py -v`

Expected: FAIL because modules are missing.

- [ ] **Step 3: Implement enums**

Create `i2sar/core/enums.py`:

```python
from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


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
```

- [ ] **Step 4: Implement helpers and errors**

Create `i2sar/core/errors.py`:

```python
class I2SARError(Exception):
    """Base exception for I2SAR."""


class SchemaError(I2SARError):
    """Raised when an HDF5 file does not match the I2SAR schema."""


class WorkflowError(I2SARError):
    """Raised when a workflow task cannot be executed."""
```

Create `i2sar/core/time.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
```

Create `i2sar/core/ids.py`:

```python
from __future__ import annotations

import re


def slugify_id(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        raise ValueError("identifier cannot be empty")
    return slug


def make_pair_id(master_scene_id: str, slave_scene_id: str) -> str:
    return f"{slugify_id(master_scene_id)}__{slugify_id(slave_scene_id)}"
```

- [ ] **Step 5: Implement entity dataclasses**

Create `i2sar/model/entities.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from i2sar.core.enums import AcquisitionMode, EntityType, ProductType, WorkflowType


@dataclass(frozen=True)
class ProjectInfo:
    project_id: str
    name: str
    root_path: str
    entity_type: EntityType = EntityType.PROJECT


@dataclass(frozen=True)
class SceneInfo:
    scene_id: str
    sensor: str
    acquisition_mode: AcquisitionMode
    acquisition_time: str
    entity_type: EntityType = EntityType.SCENE


@dataclass(frozen=True)
class PairInfo:
    pair_id: str
    master_scene_id: str
    slave_scene_id: str
    entity_type: EntityType = EntityType.PAIR
    workflow_type: WorkflowType = WorkflowType.INSAR


@dataclass(frozen=True)
class ProductInfo:
    product_id: str
    product_type: ProductType
    owner_type: EntityType
    owner_id: str
    entity_type: EntityType = EntityType.PRODUCT
```

Modify `i2sar/model/__init__.py`:

```python
from i2sar.model.entities import PairInfo, ProductInfo, ProjectInfo, SceneInfo

__all__ = ["PairInfo", "ProductInfo", "ProjectInfo", "SceneInfo"]
```

- [ ] **Step 6: Run tests and verify pass**

Run: `pytest tests/test_models.py -v`

Expected: PASS.

## Task 3: HDF5 Entity Creation and Validation

**Files:**
- Create: `i2sar/hdf/attrs.py`
- Create: `i2sar/hdf/schema.py`
- Create: `i2sar/hdf/entities.py`
- Modify: `i2sar/hdf/__init__.py`
- Test: `tests/test_hdf_entities.py`

- [ ] **Step 1: Write HDF5 entity tests**

Create `tests/test_hdf_entities.py`:

```python
import h5py

from i2sar.core.enums import AcquisitionMode, EntityType, ProductType
from i2sar.hdf.entities import create_pair_file, create_product_file, create_scene_file
from i2sar.hdf.schema import validate_pair_file, validate_product_file, validate_scene_file
from i2sar.model import PairInfo, ProductInfo, SceneInfo


def test_create_stripmap_scene_file(tmp_path):
    path = tmp_path / "scenes" / "strip.h5"
    scene = SceneInfo("strip", "lutan", AcquisitionMode.STRIPMAP, "2026-04-28T00:00:00Z")

    create_scene_file(path, scene)
    validate_scene_file(path)

    with h5py.File(path, "r") as h5:
        assert h5.attrs["entity_type"] == "scene"
        assert h5.attrs["acquisition_mode"] == "stripmap"
        assert "/stripmap" in h5
        assert h5["/stripmap"].attrs["continuous_azimuth"] is True


def test_create_tops_scene_file(tmp_path):
    path = tmp_path / "scenes" / "tops.h5"
    scene = SceneInfo("tops", "sentinel1", AcquisitionMode.TOPS, "2026-04-28T00:00:00Z")

    create_scene_file(path, scene)
    validate_scene_file(path)

    with h5py.File(path, "r") as h5:
        assert "/tops/swaths" in h5
        assert "/tops/bursts" in h5
        assert "/tops/burst_grid" in h5
        assert "/tops/azimuth_steering" in h5


def test_create_pair_and_rtc_product_files(tmp_path):
    pair_path = tmp_path / "pairs" / "m__s.h5"
    pair = PairInfo("m__s", "m", "s")
    create_pair_file(pair_path, pair)
    validate_pair_file(pair_path)

    product_path = tmp_path / "products" / "rtc_m.h5"
    product = ProductInfo("rtc_m", ProductType.RTC, EntityType.SCENE, "m")
    create_product_file(product_path, product)
    validate_product_file(product_path)

    with h5py.File(product_path, "r") as h5:
        assert h5.attrs["product_type"] == "rtc"
        assert "/rtc" in h5
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_hdf_entities.py -v`

Expected: FAIL because HDF helpers are missing.

- [ ] **Step 3: Implement attribute helpers**

Create `i2sar/hdf/attrs.py`:

```python
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
```

- [ ] **Step 4: Implement schema validators**

Create `i2sar/hdf/schema.py`:

```python
from __future__ import annotations

from pathlib import Path

import h5py

from i2sar.core.errors import SchemaError

SCHEMA_VERSION = "0.1.0"


def _require_attrs(h5: h5py.File, names: list[str]) -> None:
    missing = [name for name in names if name not in h5.attrs]
    if missing:
        raise SchemaError(f"missing required attrs: {', '.join(missing)}")


def _require_groups(h5: h5py.File, names: list[str]) -> None:
    missing = [name for name in names if name not in h5]
    if missing:
        raise SchemaError(f"missing required groups: {', '.join(missing)}")


def validate_scene_file(path: str | Path) -> None:
    with h5py.File(path, "r") as h5:
        _require_attrs(h5, ["schema_version", "entity_type", "scene_id", "sensor", "acquisition_mode"])
        if h5.attrs["entity_type"] != "scene":
            raise SchemaError("not a scene file")
        _require_groups(h5, ["metadata", "radar_grid", "orbit", "doppler", "slc", "derived", "provenance"])
        mode = h5.attrs["acquisition_mode"]
        if mode == "stripmap":
            _require_groups(h5, ["stripmap"])
        elif mode == "tops":
            _require_groups(h5, ["tops"])
            for name in ["swaths", "bursts", "burst_grid", "azimuth_steering"]:
                if name not in h5["tops"]:
                    raise SchemaError(f"missing TOPS group: /tops/{name}")
        else:
            raise SchemaError(f"unsupported acquisition_mode: {mode}")


def validate_pair_file(path: str | Path) -> None:
    with h5py.File(path, "r") as h5:
        _require_attrs(h5, ["schema_version", "entity_type", "pair_id", "master_scene_id", "slave_scene_id"])
        if h5.attrs["entity_type"] != "pair":
            raise SchemaError("not a pair file")
        _require_groups(h5, ["metadata", "registration", "geometry", "interferometry", "unwrapping", "displacement", "provenance"])


def validate_product_file(path: str | Path) -> None:
    with h5py.File(path, "r") as h5:
        _require_attrs(h5, ["schema_version", "entity_type", "product_id", "product_type", "owner_type", "owner_id"])
        if h5.attrs["entity_type"] != "product":
            raise SchemaError("not a product file")
        _require_groups(h5, ["grid", "provenance"])
        if h5.attrs["product_type"] == "rtc":
            _require_groups(h5, ["rtc"])
```

- [ ] **Step 5: Implement entity file creators**

Create `i2sar/hdf/entities.py`:

```python
from __future__ import annotations

from pathlib import Path

import h5py

from i2sar.core.enums import AcquisitionMode, ProductType
from i2sar.hdf.attrs import set_attrs
from i2sar.hdf.schema import SCHEMA_VERSION
from i2sar.model import PairInfo, ProductInfo, SceneInfo


def create_scene_file(path: str | Path, scene: SceneInfo, overwrite: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with h5py.File(path, mode) as h5:
        set_attrs(
            h5,
            {
                "schema_version": SCHEMA_VERSION,
                "entity_type": scene.entity_type,
                "scene_id": scene.scene_id,
                "sensor": scene.sensor,
                "acquisition_mode": scene.acquisition_mode,
                "acquisition_time": scene.acquisition_time,
            },
        )
        for group in ["metadata/source", "metadata/acquisition", "metadata/quality", "radar_grid", "orbit", "doppler", "slc", "derived", "provenance"]:
            h5.require_group(group)
        if scene.acquisition_mode is AcquisitionMode.STRIPMAP:
            stripmap = h5.require_group("stripmap")
            stripmap.attrs["continuous_azimuth"] = True
        if scene.acquisition_mode is AcquisitionMode.TOPS:
            for group in ["tops/swaths", "tops/bursts", "tops/burst_grid", "tops/azimuth_steering"]:
                h5.require_group(group)
    return path


def create_pair_file(path: str | Path, pair: PairInfo, overwrite: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with h5py.File(path, mode) as h5:
        set_attrs(
            h5,
            {
                "schema_version": SCHEMA_VERSION,
                "entity_type": pair.entity_type,
                "pair_id": pair.pair_id,
                "master_scene_id": pair.master_scene_id,
                "slave_scene_id": pair.slave_scene_id,
            },
        )
        for group in ["metadata", "registration", "geometry", "interferometry", "unwrapping", "displacement", "provenance"]:
            h5.require_group(group)
    return path


def create_product_file(path: str | Path, product: ProductInfo, overwrite: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with h5py.File(path, mode) as h5:
        set_attrs(
            h5,
            {
                "schema_version": SCHEMA_VERSION,
                "entity_type": product.entity_type,
                "product_id": product.product_id,
                "product_type": product.product_type,
                "owner_type": product.owner_type,
                "owner_id": product.owner_id,
            },
        )
        for group in ["grid", "provenance"]:
            h5.require_group(group)
        if product.product_type is ProductType.RTC:
            h5.require_group("rtc")
    return path
```

Modify `i2sar/hdf/__init__.py`:

```python
from i2sar.hdf.entities import create_pair_file, create_product_file, create_scene_file
from i2sar.hdf.schema import validate_pair_file, validate_product_file, validate_scene_file

__all__ = [
    "create_pair_file",
    "create_product_file",
    "create_scene_file",
    "validate_pair_file",
    "validate_product_file",
    "validate_scene_file",
]
```

- [ ] **Step 6: Run tests and verify pass**

Run: `pytest tests/test_hdf_entities.py -v`

Expected: PASS.

## Task 4: Project Index Management

**Files:**
- Create: `i2sar/project.py`
- Modify: `i2sar/__init__.py`
- Test: `tests/test_project.py`

- [ ] **Step 1: Write project tests**

Create `tests/test_project.py`:

```python
import h5py

from i2sar import Project
from i2sar.core.enums import AcquisitionMode, EntityType, ProductType
from i2sar.model import ProductInfo, SceneInfo


def test_create_project_and_add_scene_pair_product(tmp_path):
    project = Project.create(tmp_path / "demo", name="demo")
    scene = SceneInfo("master", "lutan", AcquisitionMode.STRIPMAP, "2026-04-28T00:00:00Z")
    scene_path = project.create_scene(scene)
    pair_path = project.create_pair("master", "slave")
    product = ProductInfo("rtc_master", ProductType.RTC, EntityType.SCENE, "master")
    product_path = project.create_product(product)

    assert scene_path.exists()
    assert pair_path.exists()
    assert product_path.exists()

    with h5py.File(project.path, "r") as h5:
        assert "master" in h5["scenes"]
        assert "master__slave" in h5["pairs"]
        assert "rtc_master" in h5["products"]


def test_open_existing_project(tmp_path):
    created = Project.create(tmp_path / "demo", name="demo")
    opened = Project.open(created.path)

    assert opened.path == created.path
    assert opened.root == created.root
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_project.py -v`

Expected: FAIL because `Project` is missing.

- [ ] **Step 3: Implement Project**

Create `i2sar/project.py`:

```python
from __future__ import annotations

from pathlib import Path

import h5py

from i2sar import __version__
from i2sar.core.enums import AcquisitionMode, EntityType
from i2sar.core.ids import make_pair_id
from i2sar.core.time import utc_now_iso
from i2sar.hdf.entities import create_pair_file, create_product_file, create_scene_file
from i2sar.hdf.schema import SCHEMA_VERSION
from i2sar.model import PairInfo, ProductInfo, ProjectInfo, SceneInfo


class Project:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.root = self.path.parent

    @classmethod
    def create(cls, root: str | Path, name: str) -> "Project":
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        for dirname in ["scenes", "pairs", "products", "exports"]:
            (root / dirname).mkdir(exist_ok=True)
        path = root / "project.h5"
        info = ProjectInfo(project_id=root.name, name=name, root_path=str(root))
        now = utc_now_iso()
        with h5py.File(path, "x") as h5:
            h5.attrs["i2sar_schema_version"] = SCHEMA_VERSION
            h5.attrs["i2sar_version"] = __version__
            h5.attrs["project_id"] = info.project_id
            h5.attrs["created_at"] = now
            h5.attrs["updated_at"] = now
            project = h5.require_group("project")
            project.attrs["name"] = info.name
            project.attrs["root_path"] = info.root_path
            for group in ["scenes", "pairs", "products", "workflow/runs", "workflow/tasks", "workflow/dependencies", "provenance"]:
                h5.require_group(group)
        return cls(path)

    @classmethod
    def open(cls, path: str | Path) -> "Project":
        path = Path(path).resolve()
        if path.is_dir():
            path = path / "project.h5"
        with h5py.File(path, "r") as h5:
            if "project_id" not in h5.attrs:
                raise ValueError(f"not an I2SAR project file: {path}")
        return cls(path)

    def create_scene(self, scene: SceneInfo) -> Path:
        scene_path = self.root / "scenes" / f"{scene.scene_id}.h5"
        create_scene_file(scene_path, scene)
        with h5py.File(self.path, "a") as h5:
            group = h5["scenes"].require_group(scene.scene_id)
            group.attrs["file_path"] = str(scene_path.relative_to(self.root))
            group.attrs["sensor"] = scene.sensor
            group.attrs["acquisition_mode"] = scene.acquisition_mode.value
            group.attrs["acquisition_time"] = scene.acquisition_time
            group.attrs["status"] = "created"
            h5.attrs["updated_at"] = utc_now_iso()
        return scene_path

    def create_pair(self, master_scene_id: str, slave_scene_id: str) -> Path:
        pair_id = make_pair_id(master_scene_id, slave_scene_id)
        pair = PairInfo(pair_id=pair_id, master_scene_id=master_scene_id, slave_scene_id=slave_scene_id)
        pair_path = self.root / "pairs" / f"{pair_id}.h5"
        create_pair_file(pair_path, pair)
        with h5py.File(self.path, "a") as h5:
            group = h5["pairs"].require_group(pair_id)
            group.attrs["file_path"] = str(pair_path.relative_to(self.root))
            group.attrs["master_scene_id"] = master_scene_id
            group.attrs["slave_scene_id"] = slave_scene_id
            group.attrs["status"] = "created"
            h5.attrs["updated_at"] = utc_now_iso()
        return pair_path

    def create_product(self, product: ProductInfo) -> Path:
        product_path = self.root / "products" / f"{product.product_id}.h5"
        create_product_file(product_path, product)
        with h5py.File(self.path, "a") as h5:
            group = h5["products"].require_group(product.product_id)
            group.attrs["file_path"] = str(product_path.relative_to(self.root))
            group.attrs["product_type"] = product.product_type.value
            group.attrs["owner_type"] = product.owner_type.value
            group.attrs["owner_id"] = product.owner_id
            group.attrs["status"] = "created"
            h5.attrs["updated_at"] = utc_now_iso()
        return product_path
```

Modify `i2sar/__init__.py`:

```python
"""I2SAR SAR/InSAR processing foundation."""

__version__ = "0.1.0"

from i2sar.project import Project

__all__ = ["Project", "__version__"]
```

- [ ] **Step 4: Run tests and verify pass**

Run: `pytest tests/test_project.py -v`

Expected: PASS.

## Task 5: Workflow Task Hashing and Runner

**Files:**
- Create: `i2sar/workflow/task.py`
- Create: `i2sar/workflow/runner.py`
- Modify: `i2sar/workflow/__init__.py`
- Test: `tests/test_workflow_runner.py`

- [ ] **Step 1: Write workflow tests**

Create `tests/test_workflow_runner.py`:

```python
import h5py

from i2sar import Project
from i2sar.core.enums import TaskStatus
from i2sar.workflow import TaskSpec, run_empty_task


def test_task_hash_is_stable():
    spec = TaskSpec(
        task_id="scene.validate",
        owner_type="scene",
        owner_id="s1",
        inputs=["scenes/s1.h5:/metadata"],
        outputs=["scenes/s1.h5:/provenance/tasks/scene.validate"],
        params={"strict": True},
    )

    assert spec.task_hash() == spec.task_hash()


def test_empty_task_completes_then_skips(tmp_path):
    project = Project.create(tmp_path / "demo", name="demo")
    spec = TaskSpec(
        task_id="project.empty",
        owner_type="project",
        owner_id="demo",
        inputs=[],
        outputs=["project.h5:/workflow/tasks/project.empty"],
        params={"step": "empty"},
    )

    first = run_empty_task(project, spec, resume=True)
    second = run_empty_task(project, spec, resume=True)

    assert first.status is TaskStatus.COMPLETED
    assert second.status is TaskStatus.SKIPPED

    with h5py.File(project.path, "r") as h5:
        task = h5["workflow/tasks/project.empty"]
        assert task.attrs["status"] == "skipped"
        assert task.attrs["task_hash"] == spec.task_hash()
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_workflow_runner.py -v`

Expected: FAIL because workflow modules are missing.

- [ ] **Step 3: Implement task objects**

Create `i2sar/workflow/task.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json

from i2sar.core.enums import TaskStatus


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    owner_type: str
    owner_id: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)

    def task_hash(self) -> str:
        payload = {
            "task_id": self.task_id,
            "owner_type": self.owner_type,
            "owner_id": self.owner_id,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "params": self.params,
        }
        data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    task_hash: str
    status: TaskStatus
```

- [ ] **Step 4: Implement empty runner**

Create `i2sar/workflow/runner.py`:

```python
from __future__ import annotations

import json

import h5py

from i2sar.core.enums import TaskStatus
from i2sar.core.time import utc_now_iso
from i2sar.project import Project
from i2sar.workflow.task import TaskRecord, TaskSpec


def _task_group_name(task_id: str) -> str:
    return task_id.replace("/", "_")


def run_empty_task(project: Project, spec: TaskSpec, resume: bool = True) -> TaskRecord:
    task_hash = spec.task_hash()
    group_name = _task_group_name(spec.task_id)
    now = utc_now_iso()
    with h5py.File(project.path, "a") as h5:
        tasks = h5.require_group("workflow/tasks")
        if group_name in tasks:
            group = tasks[group_name]
            if resume and group.attrs.get("task_hash") == task_hash and group.attrs.get("status") in {"completed", "skipped"}:
                group.attrs["status"] = TaskStatus.SKIPPED.value
                group.attrs["updated_at"] = now
                return TaskRecord(spec.task_id, task_hash, TaskStatus.SKIPPED)
            group.attrs["status"] = TaskStatus.INVALIDATED.value

        group = tasks.require_group(group_name)
        group.attrs["task_id"] = spec.task_id
        group.attrs["task_hash"] = task_hash
        group.attrs["owner_type"] = spec.owner_type
        group.attrs["owner_id"] = spec.owner_id
        group.attrs["started_at"] = now
        group.attrs["status"] = TaskStatus.RUNNING.value
        group.attrs["inputs_json"] = json.dumps(spec.inputs, sort_keys=True)
        group.attrs["outputs_json"] = json.dumps(spec.outputs, sort_keys=True)
        group.attrs["params_json"] = json.dumps(spec.params, sort_keys=True)
        group.attrs["finished_at"] = utc_now_iso()
        group.attrs["status"] = TaskStatus.COMPLETED.value
        h5.attrs["updated_at"] = utc_now_iso()
    return TaskRecord(spec.task_id, task_hash, TaskStatus.COMPLETED)
```

Modify `i2sar/workflow/__init__.py`:

```python
from i2sar.workflow.runner import run_empty_task
from i2sar.workflow.task import TaskRecord, TaskSpec

__all__ = ["TaskRecord", "TaskSpec", "run_empty_task"]
```

- [ ] **Step 5: Run tests and verify pass**

Run: `pytest tests/test_workflow_runner.py -v`

Expected: PASS.

## Task 6: Empty Workflow Presets for Scene, InSAR, and RTC

**Files:**
- Create: `i2sar/workflow/presets.py`
- Modify: `i2sar/workflow/__init__.py`
- Test: `tests/test_workflow_presets.py`

- [ ] **Step 1: Write preset tests**

Create `tests/test_workflow_presets.py`:

```python
from i2sar.workflow import insar_workflow_specs, rtc_workflow_specs, scene_workflow_specs


def test_scene_workflow_specs():
    specs = scene_workflow_specs("scene1")
    assert [spec.task_id for spec in specs] == [
        "scene.import_scene",
        "scene.validate_scene",
        "scene.multilook",
        "scene.quicklook",
    ]


def test_insar_and_rtc_workflow_specs_are_distinct():
    insar = insar_workflow_specs("m__s")
    rtc = rtc_workflow_specs("scene1")

    assert any(spec.task_id == "insar.form_interferogram" for spec in insar)
    assert any(spec.task_id == "rtc.compute_rtc_factor" for spec in rtc)
    assert all(not spec.task_id.startswith("insar.") for spec in rtc)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_workflow_presets.py -v`

Expected: FAIL because preset functions are missing.

- [ ] **Step 3: Implement presets**

Create `i2sar/workflow/presets.py`:

```python
from __future__ import annotations

from i2sar.workflow.task import TaskSpec


def scene_workflow_specs(scene_id: str) -> list[TaskSpec]:
    return [
        TaskSpec("scene.import_scene", "scene", scene_id, params={"stage": "import"}),
        TaskSpec("scene.validate_scene", "scene", scene_id, params={"stage": "validate"}),
        TaskSpec("scene.multilook", "scene", scene_id, params={"stage": "multilook"}),
        TaskSpec("scene.quicklook", "scene", scene_id, params={"stage": "quicklook"}),
    ]


def insar_workflow_specs(pair_id: str) -> list[TaskSpec]:
    return [
        TaskSpec("insar.create_pair", "pair", pair_id, params={"stage": "create_pair"}),
        TaskSpec("insar.prepare_geometry", "pair", pair_id, params={"stage": "geometry"}),
        TaskSpec("insar.register", "pair", pair_id, params={"stage": "registration"}),
        TaskSpec("insar.form_interferogram", "pair", pair_id, params={"stage": "interferometry"}),
        TaskSpec("insar.unwrap", "pair", pair_id, params={"stage": "unwrapping"}),
        TaskSpec("insar.phase_to_los", "pair", pair_id, params={"stage": "displacement"}),
        TaskSpec("insar.export", "pair", pair_id, params={"stage": "export"}),
    ]


def rtc_workflow_specs(scene_id: str) -> list[TaskSpec]:
    return [
        TaskSpec("rtc.prepare_geometry", "scene", scene_id, params={"stage": "geometry"}),
        TaskSpec("rtc.compute_rtc_factor", "scene", scene_id, params={"stage": "rtc_factor"}),
        TaskSpec("rtc.terrain_correct", "scene", scene_id, params={"stage": "terrain_correction"}),
        TaskSpec("rtc.radiometric_normalize", "scene", scene_id, params={"stage": "radiometry"}),
        TaskSpec("rtc.geocode", "scene", scene_id, params={"stage": "geocode"}),
        TaskSpec("rtc.make_rtc_product", "scene", scene_id, params={"stage": "product"}),
        TaskSpec("rtc.export", "scene", scene_id, params={"stage": "export"}),
    ]
```

Modify `i2sar/workflow/__init__.py`:

```python
from i2sar.workflow.presets import insar_workflow_specs, rtc_workflow_specs, scene_workflow_specs
from i2sar.workflow.runner import run_empty_task
from i2sar.workflow.task import TaskRecord, TaskSpec

__all__ = [
    "TaskRecord",
    "TaskSpec",
    "insar_workflow_specs",
    "rtc_workflow_specs",
    "run_empty_task",
    "scene_workflow_specs",
]
```

- [ ] **Step 4: Run tests and verify pass**

Run: `pytest tests/test_workflow_presets.py -v`

Expected: PASS.

## Task 7: CLI Thin Wrapper

**Files:**
- Create: `i2sar/cli/main.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write CLI tests**

Create `tests/test_cli.py`:

```python
from i2sar.cli.main import main


def test_cli_create_project(tmp_path):
    root = tmp_path / "demo"
    code = main(["create-project", str(root), "--name", "demo"])

    assert code == 0
    assert (root / "project.h5").exists()


def test_cli_run_empty_rtc_workflow(tmp_path):
    root = tmp_path / "demo"
    assert main(["create-project", str(root), "--name", "demo"]) == 0
    assert main(["run-empty", str(root / "project.h5"), "--workflow", "rtc", "--owner-id", "scene1"]) == 0
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/test_cli.py -v`

Expected: FAIL because CLI main is missing.

- [ ] **Step 3: Implement CLI**

Create `i2sar/cli/main.py`:

```python
from __future__ import annotations

import argparse

from i2sar.project import Project
from i2sar.workflow import insar_workflow_specs, rtc_workflow_specs, run_empty_task, scene_workflow_specs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="i2sar")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create-project")
    create.add_argument("root")
    create.add_argument("--name", required=True)

    run_empty = subparsers.add_parser("run-empty")
    run_empty.add_argument("project")
    run_empty.add_argument("--workflow", choices=["scene", "insar", "rtc"], required=True)
    run_empty.add_argument("--owner-id", required=True)
    run_empty.add_argument("--no-resume", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "create-project":
        Project.create(args.root, args.name)
        return 0

    if args.command == "run-empty":
        project = Project.open(args.project)
        if args.workflow == "scene":
            specs = scene_workflow_specs(args.owner_id)
        elif args.workflow == "insar":
            specs = insar_workflow_specs(args.owner_id)
        else:
            specs = rtc_workflow_specs(args.owner_id)
        for spec in specs:
            run_empty_task(project, spec, resume=not args.no_resume)
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests and verify pass**

Run: `pytest tests/test_cli.py -v`

Expected: PASS.

## Task 8: End-to-End MVP Verification

**Files:**
- Test: `tests/test_phase1_e2e.py`
- Modify: `progress.md`
- Modify: `task_plan.md`

- [ ] **Step 1: Write end-to-end test**

Create `tests/test_phase1_e2e.py`:

```python
import h5py

from i2sar import Project
from i2sar.core.enums import AcquisitionMode, EntityType, ProductType
from i2sar.model import ProductInfo, SceneInfo
from i2sar.workflow import insar_workflow_specs, rtc_workflow_specs, run_empty_task


def test_phase1_e2e_project_scene_pair_product_workflows(tmp_path):
    project = Project.create(tmp_path / "demo", name="demo")
    master = SceneInfo("master", "sentinel1", AcquisitionMode.TOPS, "2026-04-28T00:00:00Z")
    slave = SceneInfo("slave", "sentinel1", AcquisitionMode.TOPS, "2026-04-28T00:12:00Z")

    project.create_scene(master)
    project.create_scene(slave)
    project.create_pair("master", "slave")
    project.create_product(ProductInfo("rtc_master", ProductType.RTC, EntityType.SCENE, "master"))

    for spec in insar_workflow_specs("master__slave"):
        run_empty_task(project, spec)
    for spec in rtc_workflow_specs("master"):
        run_empty_task(project, spec)

    with h5py.File(project.path, "r") as h5:
        assert "master" in h5["scenes"]
        assert "master__slave" in h5["pairs"]
        assert "rtc_master" in h5["products"]
        assert "insar.form_interferogram" in h5["workflow/tasks"]
        assert "rtc.compute_rtc_factor" in h5["workflow/tasks"]
```

- [ ] **Step 2: Run all tests**

Run: `pytest -v`

Expected: PASS for all tests.

- [ ] **Step 3: Update planning files**

Modify `task_plan.md`:

```markdown
| 1 | complete | 定义包结构、核心数据模型、项目目录和配置规范 |
```

Modify `progress.md` by appending:

```markdown
- 完成第一阶段基础骨架实现：Python 包、HDF5 schema、Project/Scene/Pair/Product、空 workflow、CLI 和测试。
- 验证命令：`pytest -v` 通过。
```

- [ ] **Step 4: Commit**

Run:

```bash
git add pyproject.toml i2sar tests task_plan.md progress.md
git commit -m "feat: add i2sar phase1 foundation"
```

Expected: commit succeeds if repository is initialized. If this directory is not a git repository, skip commit and record that in `progress.md`.

## Self-Review

Spec coverage:

- Package skeleton: Task 1.
- Data models and explicit `stripmap` / `tops`: Task 2 and Task 3.
- HDF5 entity files and schema validation: Task 3.
- `project.h5` lightweight index and separate scene/pair/product files: Task 4.
- Empty workflow, task hash, provenance, resume/skipped: Task 5.
- Separate InSAR and RTC workflow types: Task 6.
- Python API with CLI thin wrapper: Task 7.
- First-stage end-to-end MVP verification: Task 8.

No placeholders remain in implementation steps. Type names are consistent across tasks: `Project`, `SceneInfo`, `PairInfo`, `ProductInfo`, `TaskSpec`, `TaskRecord`.

