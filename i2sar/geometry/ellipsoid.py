from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Ellipsoid:
    a: float
    f: float

    @property
    def b(self) -> float:
        return self.a * (1.0 - self.f)

    @property
    def e2(self) -> float:
        return self.f * (2.0 - self.f)

    @property
    def ep2(self) -> float:
        return self.e2 / (1.0 - self.e2)

    @property
    def eccentricity_squared(self) -> float:
        return self.e2


WGS84 = Ellipsoid(a=6378137.0, f=1.0 / 298.257223563)


def llh_to_ecef(lat_deg, lon_deg, height_m, ellipsoid: Ellipsoid = WGS84):
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    height = np.asarray(height_m, dtype=np.float64)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    prime_vertical = ellipsoid.a / np.sqrt(1.0 - ellipsoid.e2 * sin_lat * sin_lat)
    x = (prime_vertical + height) * cos_lat * cos_lon
    y = (prime_vertical + height) * cos_lat * sin_lon
    z = (prime_vertical * (1.0 - ellipsoid.e2) + height) * sin_lat
    return np.array([x, y, z]) if np.ndim(x) == 0 else (x, y, z)


def ecef_to_llh(x, y, z, ellipsoid: Ellipsoid = WGS84):
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    z_arr = np.asarray(z, dtype=np.float64)

    p = np.hypot(x_arr, y_arr)
    lon = np.arctan2(y_arr, x_arr)

    theta = np.arctan2(z_arr * ellipsoid.a, p * ellipsoid.b)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    lat = np.arctan2(
        z_arr + ellipsoid.ep2 * ellipsoid.b * sin_theta**3,
        p - ellipsoid.e2 * ellipsoid.a * cos_theta**3,
    )

    sin_lat = np.sin(lat)
    prime_vertical = ellipsoid.a / np.sqrt(1.0 - ellipsoid.e2 * sin_lat * sin_lat)
    height = p / np.cos(lat) - prime_vertical

    lat_deg = np.rad2deg(lat)
    lon_deg = np.rad2deg(lon)
    if np.ndim(lat_deg) == 0:
        return float(lat_deg), float(lon_deg), float(height)
    return lat_deg, lon_deg, height


def xyz_to_lon_lat(xyz, ellipsoid: Ellipsoid = WGS84):
    return ecef_to_llh(xyz[0], xyz[1], xyz[2], ellipsoid=ellipsoid)


def lon_lat_to_xyz(llh, ellipsoid: Ellipsoid = WGS84):
    lat, lon, height = llh[0], llh[1], llh[2]
    return llh_to_ecef(lat, lon, height, ellipsoid=ellipsoid)
