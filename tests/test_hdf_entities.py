import h5py
import pytest

from i2sar.core.errors import SchemaError
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
        assert bool(h5["/stripmap"].attrs["continuous_azimuth"]) is True


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


def test_validate_rejects_dataset_where_group_required(tmp_path):
    path = tmp_path / "bad_scene.h5"
    scene = SceneInfo("bad", "lutan", AcquisitionMode.STRIPMAP, "2026-04-28T00:00:00Z")
    create_scene_file(path, scene)

    with h5py.File(path, "a") as h5:
        del h5["metadata"]
        h5.create_dataset("metadata", data=[1])

    with pytest.raises(SchemaError, match="required group is not a group: metadata"):
        validate_scene_file(path)


def test_validate_rejects_wrong_schema_version(tmp_path):
    path = tmp_path / "strip.h5"
    scene = SceneInfo("strip", "lutan", AcquisitionMode.STRIPMAP, "2026-04-28T00:00:00Z")
    create_scene_file(path, scene)

    with h5py.File(path, "a") as h5:
        h5.attrs["schema_version"] = "999.0.0"

    with pytest.raises(SchemaError, match="unsupported schema_version: 999.0.0"):
        validate_scene_file(path)
