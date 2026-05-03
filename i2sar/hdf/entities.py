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
        for group in ["metadata/source", "metadata/acquisition", "metadata/quality", "radar_grid", "orbit", "orbit_raw", "doppler", "slc", "derived", "provenance"]:
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
        if product.product_type is ProductType.DEM:
            h5.require_group("dem")
        if product.product_type is ProductType.RTC:
            h5.require_group("rtc")
    return path
