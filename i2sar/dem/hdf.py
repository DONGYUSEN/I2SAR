from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from i2sar.core.enums import EntityType, ProductType
from i2sar.core.ids import slugify_id
from i2sar.dem.tiles import DEFAULT_SCENE_MARGIN_DEG, scene_bbox_from_corners, srtm_tiles_for_bbox
from i2sar.hdf.entities import create_product_file
from i2sar.model import ProductInfo
from i2sar.project import Project


def _tile_origin(tile_name: str) -> tuple[int, int]:
    if len(tile_name) < 7:
        raise ValueError(f"invalid SRTM tile name: {tile_name}")
    lat = int(tile_name[1:3])
    if tile_name[0].upper() == "S":
        lat = -lat
    lon = int(tile_name[4:7])
    if tile_name[3].upper() == "W":
        lon = -lon
    return lat, lon


def read_hgt(path: str | Path) -> tuple[np.ndarray, list[float], list[float]]:
    path = Path(path)
    raw = np.fromfile(path, dtype=">i2")
    size = int(round(np.sqrt(raw.size)))
    if size * size != raw.size:
        raise ValueError(f"unexpected HGT sample count: {raw.size}")
    elevation = raw.reshape(size, size).astype(np.float32)
    elevation[elevation <= -32768] = np.nan
    lat0, lon0 = _tile_origin(path.name)
    pixel_size = 1.0 / (size - 1)
    geotransform = [float(lon0), pixel_size, 0.0, float(lat0 + 1), 0.0, -pixel_size]
    bbox = [float(lon0), float(lon0 + 1), float(lat0), float(lat0 + 1)]
    return elevation, geotransform, bbox


def write_dem_hdf(
    path: str | Path,
    *,
    dem_id: str,
    elevation: np.ndarray,
    geotransform: list[float],
    bbox: list[float],
    source: dict[str, Any],
    margin_deg: float,
    crs: str = "EPSG:4326",
    vertical_datum: str = "unknown",
    nodata: float = np.nan,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(elevation, dtype=np.float32)
    mask = np.isfinite(arr)
    product = ProductInfo(slugify_id(dem_id), ProductType.DEM, EntityType.PROJECT, "standalone")
    create_product_file(path, product, overwrite=True)
    with h5py.File(path, "a") as h5:
        h5.attrs["product_id"] = slugify_id(dem_id)
        h5["dem"].create_dataset("elevation", data=arr)
        h5["dem"].create_dataset("mask", data=mask)
        h5["dem"].attrs["crs"] = crs
        h5["dem"].attrs["vertical_datum"] = vertical_datum
        h5["dem"].attrs["geotransform"] = np.asarray(geotransform, dtype=np.float64)
        h5["dem"].attrs["bbox"] = np.asarray(bbox, dtype=np.float64)
        h5["dem"].attrs["nodata"] = nodata
        h5["provenance"].attrs["source_json"] = json.dumps(source, sort_keys=True)
        h5["provenance"].attrs["margin_deg"] = float(margin_deg)
    return path


def write_dem_product_from_hgt(project: Project, hgt_path: str | Path, *, product_id: str | None = None) -> Path:
    hgt_path = Path(hgt_path)
    dem_id = slugify_id(product_id or f"dem_{hgt_path.stem.lower()}")
    product_path = project.root / "products" / f"{dem_id}.h5"
    elevation, geotransform, bbox = read_hgt(hgt_path)
    write_dem_hdf(
        product_path,
        dem_id=dem_id,
        elevation=elevation,
        geotransform=geotransform,
        bbox=bbox,
        source={"type": "hgt", "path": str(hgt_path.resolve()), "tile": hgt_path.stem},
        margin_deg=0.0,
    )
    with h5py.File(project.path, "a") as h5:
        group = h5["products"].require_group(dem_id)
        group.attrs["file_path"] = str(product_path.relative_to(project.root))
        group.attrs["product_type"] = ProductType.DEM.value
        group.attrs["owner_type"] = EntityType.PROJECT.value
        group.attrs["owner_id"] = h5.attrs["project_id"]
        group.attrs["status"] = "created"
    return product_path


def _scene_path(project: Project, scene_id: str) -> Path:
    with h5py.File(project.path, "r") as h5:
        if scene_id not in h5["scenes"]:
            raise ValueError(f"scene not found: {scene_id}")
        return project.root / h5["scenes"][scene_id].attrs["file_path"]


def _scene_corners(scene_path: Path) -> list[dict[str, float]]:
    with h5py.File(scene_path, "r") as h5:
        payload = json.loads(h5["derived"].attrs.get("json", "{}"))
    corners = payload.get("sceneCorners")
    if not corners:
        raise ValueError(f"scene has no derived sceneCorners: {scene_path}")
    return corners


def create_dem_product_from_scene_corners(
    project: Project,
    scene_id: str,
    *,
    elevation: np.ndarray,
    geotransform: list[float],
    margin_deg: float = DEFAULT_SCENE_MARGIN_DEG,
    product_id: str | None = None,
) -> Path:
    bbox = scene_bbox_from_corners(_scene_corners(_scene_path(project, scene_id)), margin_deg=margin_deg)
    tiles = srtm_tiles_for_bbox(bbox)
    dem_id = slugify_id(product_id or f"dem_{scene_id}")
    product_path = project.root / "products" / f"{dem_id}.h5"
    write_dem_hdf(
        product_path,
        dem_id=dem_id,
        elevation=elevation,
        geotransform=geotransform,
        bbox=bbox,
        source={"type": "scene_corners", "scene_id": scene_id, "tiles": tiles},
        margin_deg=margin_deg,
    )
    with h5py.File(project.path, "a") as h5:
        group = h5["products"].require_group(dem_id)
        group.attrs["file_path"] = str(product_path.relative_to(project.root))
        group.attrs["product_type"] = ProductType.DEM.value
        group.attrs["owner_type"] = EntityType.SCENE.value
        group.attrs["owner_id"] = scene_id
        group.attrs["status"] = "created"
    return product_path
