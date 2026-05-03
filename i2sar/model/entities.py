from __future__ import annotations

from dataclasses import dataclass, field

from i2sar.core.enums import AcquisitionMode, EntityType, ProductType, WorkflowType


@dataclass(frozen=True)
class ProjectInfo:
    project_id: str
    name: str
    root_path: str
    entity_type: EntityType = field(default=EntityType.PROJECT, init=False)


@dataclass(frozen=True)
class SceneInfo:
    scene_id: str
    sensor: str
    acquisition_mode: AcquisitionMode
    acquisition_time: str
    entity_type: EntityType = field(default=EntityType.SCENE, init=False)


@dataclass(frozen=True)
class PairInfo:
    pair_id: str
    master_scene_id: str
    slave_scene_id: str
    entity_type: EntityType = field(default=EntityType.PAIR, init=False)
    workflow_type: WorkflowType = field(default=WorkflowType.INSAR, init=False)


@dataclass(frozen=True)
class ProductInfo:
    product_id: str
    product_type: ProductType
    owner_type: EntityType
    owner_id: str
    entity_type: EntityType = field(default=EntityType.PRODUCT, init=False)
