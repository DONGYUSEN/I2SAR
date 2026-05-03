import pytest

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


def test_entity_type_and_pair_workflow_are_not_overridable():
    with pytest.raises(TypeError):
        SceneInfo(
            scene_id="bad",
            sensor="sentinel1",
            acquisition_mode=AcquisitionMode.STRIPMAP,
            acquisition_time="2026-04-28T00:00:00Z",
            entity_type=EntityType.PROJECT,
        )

    with pytest.raises(TypeError):
        PairInfo(
            pair_id="m__s",
            master_scene_id="m",
            slave_scene_id="s",
            workflow_type=WorkflowType.RTC,
        )


def test_id_helpers_document_edge_behavior():
    with pytest.raises(ValueError, match="identifier cannot be empty"):
        slugify_id("   ")

    assert slugify_id("Scene///01   HH") == "Scene_01_HH"
    assert slugify_id("master__scene") == "master_scene"
    assert make_pair_id("master__scene", "slave scene") == "master_scene__slave_scene"
