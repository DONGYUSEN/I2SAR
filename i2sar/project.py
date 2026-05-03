from __future__ import annotations

from pathlib import Path

import h5py

from i2sar import __version__
from i2sar.core.enums import EntityType
from i2sar.core.ids import make_pair_id, slugify_id
from i2sar.core.time import utc_now_iso
from i2sar.hdf.entities import create_pair_file, create_product_file, create_scene_file
from i2sar.hdf.schema import SCHEMA_VERSION
from i2sar.model import PairInfo, ProductInfo, ProjectInfo, SceneInfo


def _require_project_attrs(h5: h5py.File | h5py.Group, names: list[str]) -> None:
    missing = [name for name in names if name not in h5.attrs]
    if missing:
        raise ValueError(f"missing required project attrs: {', '.join(missing)}")


def _require_project_groups(h5: h5py.File | h5py.Group, names: list[str]) -> None:
    missing = [name for name in names if name not in h5]
    if missing:
        raise ValueError(f"missing required project groups: {', '.join(missing)}")
    for name in names:
        if not isinstance(h5[name], h5py.Group):
            raise ValueError(f"required project path is not a group: {name}")


def _validate_project_file(path: Path) -> None:
    with h5py.File(path, "r") as h5:
        _require_project_attrs(
            h5,
            ["i2sar_schema_version", "i2sar_version", "project_id", "created_at", "updated_at"],
        )
        if h5.attrs["i2sar_schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"unsupported I2SAR schema version: {h5.attrs['i2sar_schema_version']}")
        _require_project_groups(
            h5,
            ["project", "scenes", "pairs", "products", "workflow", "provenance"],
        )
        _require_project_groups(h5["workflow"], ["runs", "tasks", "dependencies"])
        _require_project_attrs(h5["project"], ["name", "root_path"])


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
        _validate_project_file(path)
        return cls(path)

    def create_scene(self, scene: SceneInfo) -> Path:
        if slugify_id(scene.scene_id) != scene.scene_id:
            raise ValueError(f"scene_id must be normalized: {scene.scene_id}")
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
        with h5py.File(self.path, "a") as h5:
            if master_scene_id not in h5["scenes"]:
                raise ValueError(f"master scene not found: {master_scene_id}")
            if slave_scene_id not in h5["scenes"]:
                raise ValueError(f"slave scene not found: {slave_scene_id}")
            create_pair_file(pair_path, pair)
            group = h5["pairs"].require_group(pair_id)
            group.attrs["file_path"] = str(pair_path.relative_to(self.root))
            group.attrs["master_scene_id"] = master_scene_id
            group.attrs["slave_scene_id"] = slave_scene_id
            group.attrs["status"] = "created"
            h5.attrs["updated_at"] = utc_now_iso()
        return pair_path

    def create_product(self, product: ProductInfo) -> Path:
        if slugify_id(product.product_id) != product.product_id:
            raise ValueError(f"product_id must be normalized: {product.product_id}")
        product_path = self.root / "products" / f"{product.product_id}.h5"
        with h5py.File(self.path, "a") as h5:
            if product.owner_type == EntityType.SCENE:
                if product.owner_id not in h5["scenes"]:
                    raise ValueError(f"scene owner not found: {product.owner_id}")
            elif product.owner_type == EntityType.PAIR:
                if product.owner_id not in h5["pairs"]:
                    raise ValueError(f"pair owner not found: {product.owner_id}")
            elif product.owner_type == EntityType.PROJECT:
                if product.owner_id != h5.attrs["project_id"]:
                    raise ValueError(f"project owner not found: {product.owner_id}")
            else:
                raise ValueError(f"unsupported product owner_type: {product.owner_type}")
            create_product_file(product_path, product)
            group = h5["products"].require_group(product.product_id)
            group.attrs["file_path"] = str(product_path.relative_to(self.root))
            group.attrs["product_type"] = product.product_type.value
            group.attrs["owner_type"] = product.owner_type.value
            group.attrs["owner_id"] = product.owner_id
            group.attrs["status"] = "created"
            h5.attrs["updated_at"] = utc_now_iso()
        return product_path
