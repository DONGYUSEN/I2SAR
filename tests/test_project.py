import h5py
import pytest

from i2sar import Project
from i2sar.core.enums import AcquisitionMode, EntityType, ProductType
from i2sar.model import ProductInfo, SceneInfo


def test_create_project_and_add_scene_pair_product(tmp_path):
    project = Project.create(tmp_path / "demo", name="demo")
    scene = SceneInfo("master", "lutan", AcquisitionMode.STRIPMAP, "2026-04-28T00:00:00Z")
    scene_path = project.create_scene(scene)
    slave = SceneInfo("slave", "lutan", AcquisitionMode.STRIPMAP, "2026-04-28T00:12:00Z")
    project.create_scene(slave)
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


def test_open_rejects_malformed_project_file(tmp_path):
    path = tmp_path / "not_project.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs["project_id"] = "bad"

    with pytest.raises(ValueError, match="missing required project attrs"):
        Project.open(path)


def test_scene_and_product_ids_must_be_normalized(tmp_path):
    project = Project.create(tmp_path / "demo", name="demo")

    with pytest.raises(ValueError, match="scene_id must be normalized"):
        project.create_scene(
            SceneInfo("../bad scene", "lutan", AcquisitionMode.STRIPMAP, "2026-04-28T00:00:00Z")
        )

    with pytest.raises(ValueError, match="product_id must be normalized"):
        project.create_product(ProductInfo("rtc/bad", ProductType.RTC, EntityType.SCENE, "master"))


def test_create_pair_requires_existing_scenes(tmp_path):
    project = Project.create(tmp_path / "demo", name="demo")
    project.create_scene(SceneInfo("master", "lutan", AcquisitionMode.STRIPMAP, "2026-04-28T00:00:00Z"))

    with pytest.raises(ValueError, match="slave scene not found: slave"):
        project.create_pair("master", "slave")


def test_create_product_requires_existing_owner(tmp_path):
    project = Project.create(tmp_path / "demo", name="demo")

    with pytest.raises(ValueError, match="scene owner not found: missing"):
        project.create_product(ProductInfo("rtc_missing", ProductType.RTC, EntityType.SCENE, "missing"))


def test_create_product_allows_project_owner(tmp_path):
    project = Project.create(tmp_path / "demo", name="demo")
    product_path = project.create_product(ProductInfo("project_export", ProductType.EXPORT, EntityType.PROJECT, "demo"))

    assert product_path.exists()
