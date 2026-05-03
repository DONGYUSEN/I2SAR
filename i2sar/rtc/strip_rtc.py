from __future__ import annotations

import json
import re
import shutil
import struct
import sys
import tempfile
import urllib.request
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import h5py
import numpy as np
from osgeo import gdal
from osgeo import osr

gdal.UseExceptions()

from i2sar.core.enums import LookSide
from i2sar.core.ids import slugify_id
from i2sar.dem.hdf import read_hgt, write_dem_hdf
from i2sar.dem.sampling import read_dem_from_hdf5, sample_dem_at_latlons
from i2sar.dem.tiles import DEFAULT_SCENE_MARGIN_DEG, scene_bbox_from_corners, srtm_tiles_for_bbox
from i2sar.geometry.accelerated_geometry import rdr2geo_fast, geo2rdr_fast
from i2sar.geometry.doppler import Doppler
from i2sar.geometry.ellipsoid import llh_to_ecef, ecef_to_llh
from i2sar.geometry.radar_grid import RadarGrid
from i2sar.io import import_scene
from i2sar.io.base import ImportResult, SourceRef
from i2sar.orbit import OrbitInterpolator
from i2sar.project import Project
from i2sar.rtc.rtc import _calculate_output_resolution, _get_utm_epsg

AF_AVAILABLE = False

def _check_arrayfire() -> bool:
    global AF_AVAILABLE
    if AF_AVAILABLE:
        return True
    try:
        import arrayfire as af
        AF_AVAILABLE = True
        return True
    except ImportError:
        return False


DEFAULT_DEM_CACHE = Path("/home/ysdong/Temp/dem")
NASADEM_SRTM_URL_TEMPLATE = "https://e4ftl01.cr.usgs.gov/MEASURES/NASADEM_HGT.001/2000.02.11/{tile}.hgt.zip"


@dataclass(frozen=True)
class StripRTCResult:
    project_path: Path
    scene_id: str
    scene_h5_path: Path
    dem_h5_path: Path
    output_dir: Path
    rtc_png: Path
    kml_file: Path
    metadata_h5: Path
    epsg: int
    output_resolution: float
    full_resolution: bool


def _scene_path(project: Project, scene_id: str) -> Path:
    with h5py.File(project.path, "r") as h5:
        return project.root / h5["scenes"][scene_id].attrs["file_path"]


def _scene_corners(scene_h5_path: Path) -> list[dict[str, float]]:
    with h5py.File(scene_h5_path, "r") as h5:
        payload = json.loads(h5["derived"].attrs.get("json", "{}"))
    corners = payload.get("sceneCorners")
    if not corners:
        raise ValueError(f"scene has no sceneCorners: {scene_h5_path}")
    return corners


def _scene_center(corners: list[dict[str, float]]) -> tuple[float, float]:
    lats = np.array([float(corner["lat"]) for corner in corners], dtype=np.float64)
    lons = np.array([float(corner["lon"]) for corner in corners], dtype=np.float64)
    return float(np.mean(lats)), float(np.mean(lons))


def _hgt_zip_path(cache_dir: Path, tile: str) -> Path:
    candidates = [
        cache_dir / f"{tile}.SRTMGL1.hgt.zip",
        cache_dir / f"{tile}.hgt.zip",
        cache_dir / f"{tile}.zip",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _ensure_hgt_zip(cache_dir: Path, tile: str, *, download_missing: bool) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _hgt_zip_path(cache_dir, tile)
    if path.exists():
        return path
    if not download_missing:
        raise FileNotFoundError(f"missing DEM tile {tile} in {cache_dir}")
    url = NASADEM_SRTM_URL_TEMPLATE.format(tile=tile)
    with urllib.request.urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())
    return path


def _extract_hgt(zip_path: Path, tile: str, work_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path) as zf:
        members = [name for name in zf.namelist() if name.lower().endswith(".hgt")]
        if not members:
            raise ValueError(f"DEM zip has no HGT member: {zip_path}")
        member = next((name for name in members if Path(name).stem.upper() == tile.upper()), members[0])
        out_path = work_dir / f"{tile}.hgt"
        with zf.open(member) as src, out_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return out_path


def _mosaic_hgt_tiles(tile_paths: list[Path]) -> tuple[np.ndarray, list[float], list[float]]:
    pieces = []
    for tile_path in tile_paths:
        elevation, geotransform, bbox = read_hgt(tile_path)
        pieces.append((tile_path.stem.upper(), elevation, geotransform, bbox))

    if len(pieces) == 1:
        _, elevation, geotransform, bbox = pieces[0]
        return elevation, geotransform, bbox

    pixel_size = float(pieces[0][2][1])
    west = min(float(piece[3][0]) for piece in pieces)
    east = max(float(piece[3][1]) for piece in pieces)
    south = min(float(piece[3][2]) for piece in pieces)
    north = max(float(piece[3][3]) for piece in pieces)
    rows = int(round((north - south) / pixel_size)) + 1
    cols = int(round((east - west) / pixel_size)) + 1
    mosaic = np.full((rows, cols), np.nan, dtype=np.float32)

    for _, elevation, geotransform, _ in pieces:
        tile_west = float(geotransform[0])
        tile_north = float(geotransform[3])
        row0 = int(round((north - tile_north) / pixel_size))
        col0 = int(round((tile_west - west) / pixel_size))
        r1 = row0 + elevation.shape[0]
        c1 = col0 + elevation.shape[1]
        target = mosaic[row0:r1, col0:c1]
        np.copyto(target, elevation, where=np.isfinite(elevation))

    return mosaic, [west, pixel_size, 0.0, north, 0.0, -pixel_size], [west, east, south, north]


def get_dem_for_extent(
    south: float,
    north: float,
    west: float,
    east: float,
    output_path: Path | str,
    dem_cache: Path | str = DEFAULT_DEM_CACHE,
    download_missing: bool = True,
) -> Path:
    """Download DEM for a given geographic extent."""
    output_path = Path(output_path)
    dem_cache = Path(dem_cache)
    
    bbox = [south, north, west, east]
    tiles = srtm_tiles_for_bbox(bbox)
    
    print(f"Downloading {len(tiles)} SRTM tiles...")
    
    with tempfile.TemporaryDirectory(prefix="i2sar-dem-") as tmp:
        tmp_path = Path(tmp)
        hgt_paths = [
            _extract_hgt(_ensure_hgt_zip(dem_cache, tile, download_missing=download_missing), tile, tmp_path)
            for tile in tiles
        ]
        elevation, geotransform, dem_bbox = _mosaic_hgt_tiles(hgt_paths)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_dem_hdf(
        output_path,
        dem_id="dem",
        elevation=elevation,
        geotransform=geotransform,
        bbox=bbox,
        source={"type": "srtm_hgt_cache", "tiles": tiles, "cache": str(dem_cache.resolve())},
    )
    
    return output_path


def ensure_dem_product_for_scene(
    project: Project,
    scene_id: str,
    *,
    dem_cache: str | Path = DEFAULT_DEM_CACHE,
    margin_deg: float = DEFAULT_SCENE_MARGIN_DEG,
    download_missing: bool = True,
) -> Path:
    scene_path = _scene_path(project, scene_id)
    corners = _scene_corners(scene_path)
    bbox = scene_bbox_from_corners(corners, margin_deg=margin_deg)
    tiles = srtm_tiles_for_bbox(bbox)
    dem_cache = Path(dem_cache)

    with tempfile.TemporaryDirectory(prefix="i2sar-strip-dem-") as tmp:
        tmp_path = Path(tmp)
        hgt_paths = [
            _extract_hgt(_ensure_hgt_zip(dem_cache, tile, download_missing=download_missing), tile, tmp_path)
            for tile in tiles
        ]
        elevation, geotransform, dem_bbox = _mosaic_hgt_tiles(hgt_paths)

    dem_id = slugify_id(f"dem_{scene_id}")
    dem_path = project.root / "products" / f"{dem_id}.h5"
    write_dem_hdf(
        dem_path,
        dem_id=dem_id,
        elevation=elevation,
        geotransform=geotransform,
        bbox=bbox,
        source={"type": "srtm_hgt_cache", "tiles": tiles, "cache": str(dem_cache.resolve()), "mosaic_bbox": dem_bbox},
        margin_deg=margin_deg,
    )
    with h5py.File(project.path, "a") as h5:
        group = h5["products"].require_group(dem_id)
        group.attrs["file_path"] = str(dem_path.relative_to(project.root))
        group.attrs["product_type"] = "dem"
        group.attrs["owner_type"] = "scene"
        group.attrs["owner_id"] = scene_id
        group.attrs["status"] = "created"
    return dem_path


def _read_source_ref(scene_h5_path: Path) -> SourceRef:
    with h5py.File(scene_h5_path, "r") as h5:
        attrs = h5["slc"].attrs
        return SourceRef(
            path=str(attrs["path"]),
            storage=str(attrs.get("storage", "file")),
            member=str(attrs["member"]) if "member" in attrs else None,
        )


def _find_imported_scene(project: Project, source: Path) -> ImportResult | None:
    source = source.resolve()
    with h5py.File(project.path, "r") as project_h5:
        scene_items = [
            (scene_id, project.root / group.attrs["file_path"])
            for scene_id, group in project_h5["scenes"].items()
        ]
    for scene_id, scene_path in scene_items:
        try:
            ref = _read_source_ref(scene_path)
        except (KeyError, OSError, ValueError):
            continue
        if Path(ref.path).resolve() != source:
            continue
        with h5py.File(scene_path, "r") as scene_h5:
            sensor = str(scene_h5.attrs["sensor"])
            from i2sar.core.enums import AcquisitionMode

            acquisition_mode = AcquisitionMode(str(scene_h5.attrs["acquisition_mode"]))
        return ImportResult(
            scene_id=scene_id,
            scene_path=scene_path,
            sensor=sensor,
            acquisition_mode=acquisition_mode,
            slc=ref,
        )
    return None


def _radar_grid_payload(scene_h5_path: Path) -> dict[str, Any]:
    with h5py.File(scene_h5_path, "r") as h5:
        return json.loads(h5["radar_grid"].attrs.get("json", "{}"))


def _acquisition_payload(scene_h5_path: Path) -> dict[str, Any]:
    with h5py.File(scene_h5_path, "r") as h5:
        return json.loads(h5["metadata/acquisition"].attrs.get("json", "{}"))


def _scene_root_attrs(scene_h5_path: Path) -> dict[str, Any]:
    with h5py.File(scene_h5_path, "r") as h5:
        return dict(h5.attrs)


def _format_satellite_name(value: Any, fallback: str) -> str:
    raw = str(value or fallback or "Satellite").strip()
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", raw) if part]
    if not parts:
        return "Satellite"
    return "".join(part[:1].upper() + part[1:].lower() for part in parts)


def _yyyymmdd_from_text(value: Any) -> Optional[str]:
    text = str(value or "")
    match = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", text)
    if match is None:
        return None
    return "".join(match.groups())


def _rtc_metadata_filename(scene_h5_path: Path, scene_name: str, acq: dict[str, Any]) -> str:
    root_attrs = _scene_root_attrs(scene_h5_path)
    satellite = _format_satellite_name(root_attrs.get("sensor") or acq.get("source") or acq.get("mission"), scene_name)
    date = (
        _yyyymmdd_from_text(root_attrs.get("acquisition_time"))
        or _yyyymmdd_from_text(acq.get("startTimeUTC"))
        or _yyyymmdd_from_text(scene_name)
        or "unknown_date"
    )
    return f"{satellite}_{date}_RTC.h5"


def _orbit_arrays(scene_h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(scene_h5_path, "r") as h5:
        return h5["orbit/time"][:], h5["orbit/position"][:]


def _build_gcps(corners: list[dict[str, float]], width: int, height: int) -> list[gdal.GCP]:
    gcps = []
    for corner in corners:
        pixel = float(corner.get("pixel", corner.get("refColumn", 0.0)))
        line = float(corner.get("line", corner.get("refRow", 0.0)))
        lon = float(corner["lon"])
        lat = float(corner["lat"])
        gcps.append(gdal.GCP(lon, lat, 0.0, pixel, line))
    if len(gcps) >= 3:
        return gcps
    raise ValueError("strip RTC requires at least three geolocation corner/control points")


def _slc_block_to_power(block: np.ndarray, *, scale: float = 1.0) -> np.ndarray:
    if np.iscomplexobj(block):
        real = block.real.astype(np.float32, copy=False)
        imag = block.imag.astype(np.float32, copy=False)
        power = real * real + imag * imag
    else:
        arr = np.asarray(block)
        if arr.ndim == 3 and arr.shape[-1] == 2:
            real = arr[..., 0].astype(np.float32, copy=False)
            imag = arr[..., 1].astype(np.float32, copy=False)
            power = real * real + imag * imag
        else:
            vals = arr.astype(np.float32, copy=False)
            power = vals * vals
    if scale != 1.0:
        power = power * np.float32(scale)
    return power.astype(np.float32, copy=False)


def _write_power_geotiff(src_ds, output_path: Path, *, scale: float, block_rows: int = 512) -> Path:
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        str(output_path),
        src_ds.RasterXSize,
        src_ds.RasterYSize,
        1,
        gdal.GDT_Float32,
        ["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"],
    )
    out_ds.SetGCPs(src_ds.GetGCPs(), src_ds.GetGCPProjection())
    band = src_ds.GetRasterBand(1)
    out_band = out_ds.GetRasterBand(1)
    for row in range(0, src_ds.RasterYSize, block_rows):
        n_rows = min(block_rows, src_ds.RasterYSize - row)
        block = band.ReadAsArray(0, row, src_ds.RasterXSize, n_rows)
        out_band.WriteArray(_slc_block_to_power(block, scale=scale), 0, row)
    out_band.FlushCache()
    out_ds.FlushCache()
    out_ds = None
    return output_path


def _fill_invalid_dem(elevation: np.ndarray) -> np.ndarray:
    arr = np.asarray(elevation, dtype=np.float32).copy()
    if np.all(np.isfinite(arr)):
        return arr
    fill = float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else 0.0
    arr[~np.isfinite(arr)] = fill
    return arr


def _dem_gradient_arrays(elevation: np.ndarray, geotransform: list[float]) -> tuple[np.ndarray, np.ndarray]:
    filled = _fill_invalid_dem(elevation)
    pixel_deg = abs(float(geotransform[1]))
    center_lat = float(geotransform[3]) - pixel_deg * filled.shape[0] / 2.0
    dx = max(1.0, 111_320.0 * np.cos(np.deg2rad(center_lat)) * pixel_deg)
    dy = max(1.0, 110_574.0 * pixel_deg)
    grad_north, grad_east = np.gradient(filled, dy, dx)
    return grad_east.astype(np.float32), (-grad_north).astype(np.float32)


def _geolocation_corner_model(corners: list[dict[str, float]]) -> dict[str, float]:
    if len(corners) < 4:
        raise ValueError("physical strip RTC requires at least four scene corner/geolocation points")
    rows = np.array([float(corner.get("line", corner.get("refRow", 0.0))) for corner in corners])
    cols = np.array([float(corner.get("pixel", corner.get("refColumn", 0.0))) for corner in corners])
    lats = np.array([float(corner["lat"]) for corner in corners])
    lons = np.array([float(corner["lon"]) for corner in corners])
    row_min, row_max = float(np.min(rows)), float(np.max(rows))
    col_min, col_max = float(np.min(cols)), float(np.max(cols))

    def nearest(row_target: float, col_target: float) -> int:
        dist = (rows - row_target) ** 2 + (cols - col_target) ** 2
        return int(np.argmin(dist))

    ul = nearest(row_min, col_min)
    ur = nearest(row_min, col_max)
    ll = nearest(row_max, col_min)
    lr = nearest(row_max, col_max)
    return {
        "row_min": row_min,
        "row_span": max(1.0, row_max - row_min),
        "col_min": col_min,
        "col_span": max(1.0, col_max - col_min),
        "lat_ul": float(lats[ul]),
        "lat_ur": float(lats[ur]),
        "lat_ll": float(lats[ll]),
        "lat_lr": float(lats[lr]),
        "lon_ul": float(lons[ul]),
        "lon_ur": float(lons[ur]),
        "lon_ll": float(lons[ll]),
        "lon_lr": float(lons[lr]),
    }


def _interpolate_lat_lon(model: dict[str, float], rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = (cols - model["col_min"]) / model["col_span"]
    v = (rows - model["row_min"]) / model["row_span"]
    lat = (
        (1.0 - u) * (1.0 - v) * model["lat_ul"]
        + u * (1.0 - v) * model["lat_ur"]
        + (1.0 - u) * v * model["lat_ll"]
        + u * v * model["lat_lr"]
    )
    lon = (
        (1.0 - u) * (1.0 - v) * model["lon_ul"]
        + u * (1.0 - v) * model["lon_ur"]
        + (1.0 - u) * v * model["lon_ll"]
        + u * v * model["lon_lr"]
    )
    return lat.astype(np.float64), lon.astype(np.float64)


def _compute_satellite_positions_at_aztimes(
    az_times: np.ndarray,
    orbit: OrbitInterpolator,
) -> np.ndarray:
    az_times_flat = np.atleast_1d(az_times).ravel()
    n_times = len(az_times_flat)
    
    positions = np.zeros((n_times, 3), dtype=np.float64)
    for i in range(n_times):
        state = orbit.state_at(az_times_flat[i], allow_extrapolation=True)
        positions[i] = state.position
    
    return positions


def _compute_local_cos_incidence_vectorized(
    lat: np.ndarray,
    lon: np.ndarray,
    height: np.ndarray,
    dem_slope_east: np.ndarray,
    dem_slope_north: np.ndarray,
    sat_pos: np.ndarray,
) -> np.ndarray:
    x, y, z = llh_to_ecef(lat, lon, height)
    target = np.stack([x, y, z], axis=-1)
    
    n_points = lat.size
    
    if sat_pos.ndim == 1:
        sensor = sat_pos[None, :] - target
    elif sat_pos.shape[0] == n_points:
        sensor = sat_pos - target
    else:
        sensor = sat_pos[:, None, :] - target
    
    sensor_norm = np.linalg.norm(sensor, axis=-1, keepdims=True)
    sensor_norm = np.maximum(sensor_norm, 1e-6)
    sensor = sensor / sensor_norm

    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    east = np.stack([-sin_lon, cos_lon, np.zeros_like(lon_rad)], axis=-1)
    north = np.stack([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat], axis=-1)
    up = np.stack([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat], axis=-1)

    normal_enu_e = -dem_slope_east[..., None]
    normal_enu_n = -dem_slope_north[..., None]
    normal_enu_u = np.ones_like(normal_enu_e)
    normal_norm = np.sqrt(normal_enu_e**2 + normal_enu_n**2 + normal_enu_u**2)
    normal_norm = np.maximum(normal_norm, 1e-10)
    
    normal = (
        (normal_enu_e / normal_norm) * east
        + (normal_enu_n / normal_norm) * north
        + (normal_enu_u / normal_norm) * up
    )
    
    return np.sum(normal * sensor, axis=-1)


def _load_orbit_from_scene(scene_h5_path: Path) -> OrbitInterpolator:
    with h5py.File(scene_h5_path, "r") as h5:
        time = h5["orbit/time"][:]
        position = h5["orbit/position"][:]
        velocity = h5["orbit/velocity"][:]
        return OrbitInterpolator(time, position, velocity, method="hermite")


def _load_doppler_from_scene(scene_h5_path: Path) -> float:
    with h5py.File(scene_h5_path, "r") as h5:
        doppler_json = h5["radar_grid"].attrs.get("json", "{}")
        doppler_data = json.loads(doppler_json)
        return doppler_data.get("doppler", 0.0)


def _compute_rtc_block_vectorized(
    block_rows: np.ndarray,
    block_cols: np.ndarray,
    radar_grid: RadarGrid,
    orbit: OrbitInterpolator,
    doppler: float,
    dem_elevation: np.ndarray,
    dem_geotransform: list[float],
    slc_power: np.ndarray,
    calibration_scale: float,
    look_side: LookSide = LookSide.RIGHT,
    use_dem: bool = True,
    use_gpu: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_rows = len(block_rows)
    n_cols = len(block_cols)

    row_grid, col_grid = np.meshgrid(block_rows, block_cols, indexing="ij")
    flat_rows = row_grid.ravel()
    flat_cols = col_grid.ravel()

    if use_gpu and _check_arrayfire():
        llh = rdr2geo_fast(
            line=flat_rows,
            pixel=flat_cols,
            radar_grid=radar_grid,
            satellite_position=orbit,
            velocity=None,
            doppler=doppler,
            dem=0.0,
            method="arrayfire",
        )
    else:
        from i2sar.geometry.rdr2geo import rdr2geo_parallel
        llh = rdr2geo_parallel(
            line=flat_rows,
            pixel=flat_cols,
            radar_grid=radar_grid,
            satellite_position=orbit,
            velocity=None,
            doppler=doppler,
            dem=0.0,
        )

    lat = llh[0].reshape(n_rows, n_cols)
    lon = llh[1].reshape(n_rows, n_cols)
    height = llh[2].reshape(n_rows, n_cols)

    if use_dem:
        dem_height = sample_dem_at_latlons(lat.ravel(), lon.ravel(), dem_elevation, dem_geotransform)
        height = dem_height.reshape(n_rows, n_cols)
    else:
        height = llh[2].reshape(n_rows, n_cols)
        height = np.clip(height, 0, 9000)

    az_times = radar_grid.line_to_azimuth_time(flat_rows)
    sat_positions = _compute_satellite_positions_at_aztimes(az_times, orbit)
    sat_positions_flat = sat_positions.reshape(-1, 3)

    dem_slope_east, dem_slope_north = _dem_gradient_arrays(dem_elevation, dem_geotransform)
    slope_e = sample_dem_at_latlons(lat.ravel(), lon.ravel(), dem_slope_east, dem_geotransform)
    slope_n = sample_dem_at_latlons(lat.ravel(), lon.ravel(), dem_slope_north, dem_geotransform)
    slope_e = slope_e.reshape(n_rows, n_cols)
    slope_n = slope_n.reshape(n_rows, n_cols)

    cos_inc = _compute_local_cos_incidence_vectorized(
        lat.ravel(), lon.ravel(), height.ravel(), 
        slope_e.ravel(), slope_n.ravel(), sat_positions_flat
    )
    cos_inc = cos_inc.reshape(n_rows, n_cols)

    valid = np.isfinite(height) & np.isfinite(slope_e) & np.isfinite(slope_n) & (cos_inc > 1e-4)

    rtc = np.full(slc_power.shape, np.nan, dtype=np.float32)
    if calibration_scale != 1.0:
        power_scaled = slc_power * calibration_scale
    else:
        power_scaled = slc_power
    rtc[valid] = (power_scaled[valid] / cos_inc[valid]).astype(np.float32)

    return rtc, lat, lon, height, cos_inc


def _write_physical_rtc_radar_geotiff(
    src_ds,
    output_path: Path,
    *,
    scene_h5_path: Path,
    dem_h5_path: Path,
    corners: list[dict[str, float]],
    radar_grid: RadarGrid,
    calibration_scale: float,
    block_rows: int = 128,
    use_gpu: bool = True,
) -> Path:
    elevation, _, dem_geotransform = read_dem_from_hdf5(dem_h5_path)
    orbit = _load_orbit_from_scene(scene_h5_path)
    doppler = _load_doppler_from_scene(scene_h5_path)
    
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        str(output_path),
        src_ds.RasterXSize,
        src_ds.RasterYSize,
        1,
        gdal.GDT_Float32,
        ["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"],
    )
    out_ds.SetGCPs(src_ds.GetGCPs(), src_ds.GetGCPProjection())
    band = src_ds.GetRasterBand(1)
    out_band = out_ds.GetRasterBand(1)
    
    all_cols = np.arange(src_ds.RasterXSize, dtype=np.float64)
    
    total_blocks = (src_ds.RasterYSize + block_rows - 1) // block_rows
    print(f"Processing {total_blocks} blocks ({block_rows} rows each)...", file=sys.stderr)
    sys.stderr.flush()
    
    for row_start in range(0, src_ds.RasterYSize, block_rows):
        n_block_rows = min(block_rows, src_ds.RasterYSize - row_start)
        block_row_values = np.arange(row_start, row_start + n_block_rows, dtype=np.float64)
        
        slc_power = _slc_block_to_power(
            band.ReadAsArray(0, row_start, src_ds.RasterXSize, n_block_rows),
            scale=1.0,
        )
        
        rtc, _, _, _, _ = _compute_rtc_block_vectorized(
            block_rows=block_row_values,
            block_cols=all_cols,
            radar_grid=radar_grid,
            orbit=orbit,
            doppler=doppler,
            dem_elevation=elevation,
            dem_geotransform=dem_geotransform,
            slc_power=slc_power,
            calibration_scale=calibration_scale,
            use_gpu=use_gpu,
        )
        
        out_band.WriteArray(rtc, 0, row_start)
        out_band.SetNoDataValue(np.nan)
        out_band.FlushCache()
    
    out_ds.FlushCache()
    out_ds = None
    return output_path


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _write_gray_png(path: Path, image: np.ndarray, geotransform: Optional[list[float]] = None, epsg: Optional[int] = None) -> Path:
    image = np.asarray(image, dtype=np.uint8)
    height, width = image.shape
    
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    payload = [
        b"\x89PNG\r\n\x1a\n",
        _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)),
        _png_chunk(b"IDAT", zlib.compress(raw, level=6)),
        _png_chunk(b"IEND", b""),
    ]
    path.write_bytes(b"".join(payload))
    
    if geotransform is not None:
        _write_world_file(path, geotransform)
    
    if epsg is not None and geotransform is not None:
        _write_png_aux_xml(path, geotransform, epsg)
    
    return path


def _write_world_file(png_path: Path, geotransform: list[float]) -> Path:
    world_path = png_path.with_suffix(".pgw")
    
    pixel_size_x = geotransform[1]
    rotation_x = geotransform[2] if len(geotransform) > 2 else 0.0
    rotation_y = geotransform[4] if len(geotransform) > 4 else 0.0
    pixel_size_y = geotransform[5]
    origin_x = geotransform[0]
    origin_y = geotransform[3]
    
    world_content = f"""{pixel_size_x}
{rotation_x}
{rotation_y}
{pixel_size_y}
{origin_x}
{origin_y}
"""
    world_path.write_text(world_content)
    return world_path


def _write_png_aux_xml(png_path: Path, geotransform: list[float], epsg: int) -> Path:
    aux_path = png_path.with_suffix(png_path.suffix + ".aux.xml")
    
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    wkt = srs.ExportToWkt()
    
    pixel_size_x = geotransform[1]
    rotation_x = geotransform[2] if len(geotransform) > 2 else 0.0
    rotation_y = geotransform[4] if len(geotransform) > 4 else 0.0
    pixel_size_y = geotransform[5]
    origin_x = geotransform[0]
    origin_y = geotransform[3]
    
    xml_content = f"""<PAMDataset>
  <SRS dataAxisToSRSAxisMapping="1,2">{wkt}</SRS>
  <GeoTransform>{origin_x}, {pixel_size_x}, {rotation_x}, {origin_y}, {rotation_y}, {pixel_size_y}</GeoTransform>
</PAMDataset>"""
    aux_path.write_text(xml_content)
    return aux_path


def _epsg_wkt(epsg: int) -> str:
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(int(epsg))
    return srs.ExportToWkt()


def _write_metadata_h5(
    path: Path,
    *,
    scene_id: str,
    scene_h5_path: Path,
    dem_h5_path: Path,
    rtc_png: Path,
    kml_file: Path,
    epsg: int,
    output_resolution: float,
    full_resolution: bool,
) -> None:
    with h5py.File(path, "w") as h5:
        h5.attrs["scene_id"] = scene_id
        h5.attrs["scene_h5_path"] = str(scene_h5_path)
        h5.attrs["dem_h5_path"] = str(dem_h5_path)
        h5.attrs["rtc_png"] = str(rtc_png)
        h5.attrs["kml_file"] = str(kml_file)
        h5.attrs["epsg"] = int(epsg)
        h5.attrs["output_resolution"] = float(output_resolution)
        h5.attrs["full_resolution"] = bool(full_resolution)


def _stretch_valid_to_uint8(data: np.ndarray, valid_mask: np.ndarray, lower_percent: float = 5.0, upper_percent: float = 95.0) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    valid = valid_mask & np.isfinite(data)
    out = np.zeros(data.shape, dtype=np.uint8)
    if not np.any(valid):
        return out
    valid_values = data[valid]
    lo = float(np.percentile(valid_values, lower_percent))
    hi = float(np.percentile(valid_values, upper_percent))
    if hi <= lo:
        out[valid] = 255
        return out
    scaled = np.clip((data[valid] - lo) / (hi - lo), 0.0, 1.0)
    out[valid] = np.round(scaled * 255.0).astype(np.uint8)
    return out


def _log10_power_to_uint8(data: np.ndarray, lower_percent: float = 5.0, upper_percent: float = 95.0) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    valid = np.isfinite(data) & (data > 0.0)
    log_data = np.zeros(data.shape, dtype=np.float32)
    log_data[valid] = np.log10(data[valid])
    return _stretch_valid_to_uint8(log_data, valid, lower_percent=lower_percent, upper_percent=upper_percent)


def _dataset_lon_lat_footprint(ds) -> tuple[list[tuple[float, float]], tuple[float, float, float, float]]:
    width = ds.RasterXSize
    height = ds.RasterYSize
    gt = ds.GetGeoTransform()
    srs_wkt = ds.GetProjection()
    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(srs_wkt)
    src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(4326)
    dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(src_srs, dst_srs)
    points = []
    for px, py in ((0, 0), (width, 0), (width, height), (0, height)):
        x = gt[0] + px * gt[1] + py * gt[2]
        y = gt[3] + px * gt[4] + py * gt[5]
        lon, lat, _ = transform.TransformPoint(x, y)
        points.append((lon, lat))
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    return points, (min(lons), max(lons), min(lats), max(lats))


def _write_png_kml(
    path: Path,
    *,
    png_path: Path,
    corners: list[tuple[float, float]],
    north: float,
    south: float,
    east: float,
    west: float,
    name: str,
) -> Path:
    href = png_path.name
    ul, ur, lr, ll = corners
    quad_coordinates = " ".join(
        f"{lon:.10f},{lat:.10f},0" for lon, lat in (ll, lr, ur, ul)
    )
    text = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2">
  <GroundOverlay>
    <name>{name}</name>
    <Icon>
      <href>{href}</href>
    </Icon>
    <gx:LatLonQuad>
      <coordinates>{quad_coordinates}</coordinates>
    </gx:LatLonQuad>
    <LatLonBox>
      <north>{north:.10f}</north>
      <south>{south:.10f}</south>
      <east>{east:.10f}</east>
      <west>{west:.10f}</west>
      <rotation>0</rotation>
    </LatLonBox>
  </GroundOverlay>
</kml>
"""
    path.write_text(text, encoding="utf-8")
    return path


def process_strip_scene_rtc(
    scene_h5_path: str | Path,
    dem_h5_path: str | Path,
    output_dir: str | Path,
    *,
    output_resolution: Optional[float] = None,
    full_resolution: bool = True,
    scene_id: Optional[str] = None,
    use_gpu: bool = True,
    nalks: int = 1,
    nrlks: int = 1,
) -> StripRTCResult:
    from i2sar.processing import multilook_hdf5

    scene_h5_path = Path(scene_h5_path)
    dem_h5_path = Path(dem_h5_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ml_h5_path = None
    if nalks > 1 or nrlks > 1:
        ml_h5_path = output_dir / f"{scene_h5_path.stem}_ml{nalks}x{nrlks}.h5"
        print(f"Applying multilook: nalks={nalks}, nrlks={nrlks}", file=sys.stderr)
        multilook_hdf5(scene_h5_path, ml_h5_path, nalks=nalks, nrlks=nrlks, preserve_phase=True)
        scene_h5_path = ml_h5_path

    corners = _scene_corners(scene_h5_path)
    center_lat, center_lon = _scene_center(corners)
    epsg, _ = _get_utm_epsg(center_lat, center_lon)
    source_ref = _read_source_ref(scene_h5_path)
    radar_payload = _radar_grid_payload(scene_h5_path)
    radar_grid = RadarGrid.from_mapping(radar_payload)
    acq = _acquisition_payload(scene_h5_path)
    if output_resolution is None:
        output_resolution = _calculate_output_resolution(
            float(radar_payload.get("columnSpacing", radar_grid.range_pixel_spacing_m)),
            float(radar_payload.get("rowSpacing", 1.0)),
        )

    src_ds = gdal.Open(source_ref.vsi_path())
    if src_ds is None:
        raise RuntimeError(f"GDAL failed to open SLC source: {source_ref.vsi_path()}")
    gcps = _build_gcps(corners, src_ds.RasterXSize, src_ds.RasterYSize)
    src_ds.SetGCPs(gcps, _epsg_wkt(4326))

    scene_name = slugify_id(scene_id or scene_h5_path.stem)
    rtc_png = output_dir / f"{scene_name}_strip_rtc.png"
    kml_file = output_dir / f"{scene_name}_strip_rtc.kml"
    metadata_h5 = output_dir / _rtc_metadata_filename(scene_h5_path, scene_name, acq)

    scale = 1.0
    if full_resolution:
        spacing_area = float(radar_payload.get("columnSpacing", radar_grid.range_pixel_spacing_m)) * float(
            radar_payload.get("rowSpacing", 1.0)
        )
        frequency = float(acq.get("centerFrequency", 0.0))
        wavelength = 299792458.0 / frequency if frequency > 0 else 1.0
        scale = spacing_area * wavelength * wavelength / (4.0 * np.pi**3)

    with tempfile.TemporaryDirectory(prefix="i2sar-strip-rtc-") as tmp:
        tmp_dir = Path(tmp)
        radar_rtc_path = tmp_dir / f"{scene_name}_physical_rtc_radar.tif"
        geocoded_rtc_path = tmp_dir / f"{scene_name}_physical_rtc_utm.tif"
        _write_physical_rtc_radar_geotiff(
            src_ds,
            radar_rtc_path,
            scene_h5_path=scene_h5_path,
            dem_h5_path=dem_h5_path,
            corners=corners,
            radar_grid=radar_grid,
            calibration_scale=scale,
            use_gpu=use_gpu,
        )
        src_ds = None

        warp_options = gdal.WarpOptions(
            format="GTiff",
            dstSRS=f"EPSG:{epsg}",
            xRes=float(output_resolution),
            yRes=float(output_resolution),
            resampleAlg="bilinear",
            outputType=gdal.GDT_Float32,
            creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"],
            multithread=True,
            warpOptions=["NUM_THREADS=ALL_CPUS"],
        )
        gdal.Warp(str(geocoded_rtc_path), str(radar_rtc_path), options=warp_options)
        geocoded_ds = gdal.Open(str(geocoded_rtc_path))
        geocoded = geocoded_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        geotransform = list(geocoded_ds.GetGeoTransform())
        corners_lon_lat, (west, east, south, north) = _dataset_lon_lat_footprint(geocoded_ds)
        geocoded_ds = None
        png_data = _log10_power_to_uint8(geocoded)
        _write_gray_png(rtc_png, png_data, geotransform=geotransform, epsg=epsg)
        _write_png_kml(
            kml_file,
            png_path=rtc_png,
            corners=corners_lon_lat,
            north=north,
            south=south,
            east=east,
            west=west,
            name=scene_name,
        )

    _write_metadata_h5(
        metadata_h5,
        scene_id=scene_name,
        scene_h5_path=scene_h5_path,
        dem_h5_path=dem_h5_path,
        rtc_png=rtc_png,
        kml_file=kml_file,
        epsg=epsg,
        output_resolution=float(output_resolution),
        full_resolution=full_resolution,
    )
    return StripRTCResult(
        project_path=scene_h5_path.parents[1] / "project.h5",
        scene_id=scene_name,
        scene_h5_path=scene_h5_path,
        dem_h5_path=dem_h5_path,
        output_dir=output_dir,
        rtc_png=rtc_png,
        kml_file=kml_file,
        metadata_h5=metadata_h5,
        epsg=epsg,
        output_resolution=float(output_resolution),
        full_resolution=full_resolution,
    )


def run_strip_rtc(
    source: str | Path,
    *,
    output_dir: str | Path,
    project_dir: str | Path | None = None,
    sensor: str = "auto",
    acquisition_mode: str = "auto",
    dem_cache: str | Path = DEFAULT_DEM_CACHE,
    output_resolution: Optional[float] = None,
    full_resolution: bool = True,
    download_dem: bool = True,
    use_gpu: bool = True,
    nalks: int = 1,
    nrlks: int = 1,
) -> StripRTCResult:
    source = Path(source)
    output_dir = Path(output_dir)
    project_dir = Path(project_dir) if project_dir is not None else output_dir / "project"
    if project_dir.exists() and (project_dir / "project.h5").exists():
        project = Project.open(project_dir)
    else:
        project = Project.create(project_dir, name="strip_rtc")

    imported = _find_imported_scene(project, source)
    if imported is None:
        imported = import_scene(project, source, sensor=sensor, acquisition_mode=acquisition_mode)
    dem_path = ensure_dem_product_for_scene(
        project,
        imported.scene_id,
        dem_cache=dem_cache,
        download_missing=download_dem,
    )
    return process_strip_scene_rtc(
        imported.scene_path,
        dem_path,
        output_dir,
        output_resolution=output_resolution,
        full_resolution=full_resolution,
        scene_id=imported.scene_id,
        use_gpu=use_gpu,
        nalks=nalks,
        nrlks=nrlks,
    )


__all__ = [
    "DEFAULT_DEM_CACHE",
    "StripRTCResult",
    "ensure_dem_product_for_scene",
    "process_strip_scene_rtc",
    "run_strip_rtc",
]
