from __future__ import annotations

import math
from typing import Iterable, Mapping


DEFAULT_SCENE_MARGIN_DEG = 0.2


def _normalize_corners(corners: Mapping[str, Mapping[str, float]] | Iterable[Mapping[str, float]]) -> list[Mapping[str, float]]:
    if isinstance(corners, Mapping):
        return list(corners.values())
    return list(corners)


def _normalize_lon_360(lon: float) -> float:
    return float(lon) % 360.0


def _canonicalize_unwrapped_west(west: float, east: float) -> tuple[float, float]:
    while west < -180.0:
        west += 360.0
        east += 360.0
    while west >= 180.0:
        west -= 360.0
        east -= 360.0
    return west, east


def _scene_lon_interval(lons: list[float], margin_deg: float) -> tuple[float, float]:
    normalized = sorted(_normalize_lon_360(lon) for lon in lons)
    if not normalized:
        raise ValueError("no longitudes provided")
    if len(normalized) == 1:
        start = normalized[0]
        span = 0.0
    else:
        gaps = []
        for idx, lon in enumerate(normalized):
            nxt = normalized[(idx + 1) % len(normalized)]
            if idx == len(normalized) - 1:
                nxt += 360.0
            gaps.append(nxt - lon)
        max_gap_idx = int(max(range(len(gaps)), key=lambda idx: gaps[idx]))
        start = normalized[(max_gap_idx + 1) % len(normalized)]
        span = 360.0 - gaps[max_gap_idx]
    west, east = _canonicalize_unwrapped_west(start, start + span)
    west -= margin_deg
    east += margin_deg
    return _canonicalize_unwrapped_west(west, east)


def scene_bbox_from_corners(
    corners: Mapping[str, Mapping[str, float]] | Iterable[Mapping[str, float]],
    margin_deg: float = DEFAULT_SCENE_MARGIN_DEG,
) -> list[float]:
    normalized = _normalize_corners(corners)
    if not normalized:
        raise ValueError("no scene corners provided")
    lats = [float(corner["lat"]) for corner in normalized]
    lons = [float(corner["lon"]) for corner in normalized]
    west, east = _scene_lon_interval(lons, margin_deg=margin_deg)
    south = min(lats) - margin_deg
    north = max(lats) + margin_deg
    return [round(west, 10), round(east, 10), round(south, 10), round(north, 10)]


def _tile_id(lat: int, lon: int) -> str:
    lat_tag = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
    lon_tag = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"
    return f"{lat_tag}{lon_tag}"


def srtm_tiles_for_bbox(bbox: list[float] | tuple[float, float, float, float]) -> list[str]:
    west, east, south, north = [float(value) for value in bbox]
    lat_start = math.floor(south)
    lat_stop = math.ceil(north)
    lon_start = math.floor(west)
    lon_stop = math.ceil(east)
    if lat_start >= lat_stop or lon_start >= lon_stop:
        raise ValueError(f"invalid bbox: {bbox}")
    tiles = []
    for lat in range(lat_start, lat_stop):
        for lon in range(lon_start, lon_stop):
            tiles.append(_tile_id(lat, lon))
    return tiles
