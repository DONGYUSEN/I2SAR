from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import h5py
import pyproj
from osgeo import gdal
gdal.UseExceptions()

from i2sar.dem.interpolator import DEMInterpolator
from i2sar.geometry.radar_grid import RadarGrid
from i2sar.geometry.accelerated_geometry import rdr2geo_fast as rdr2geo
from i2sar.geometry.geo2rdr import compute_geo2rdr_mapping
from i2sar.orbit.interpolate import OrbitInterpolator
from i2sar.project import Project


@dataclass(frozen=True)
class RTCResult:
    output_path: Path
    hdf5_file: Path
    gamma0_file: Path
    sigma0_file: Path
    beta0_file: Path
    incidence_angle_file: Path
    intensity_file: Path
    intensity_png_file: Path
    resolution: float
    epsg: int
    utm_zone: int
    output_rows: int
    output_cols: int


@dataclass(frozen=True)
class UTMGeocodePlan:
    pixel_y: np.ndarray
    pixel_x: np.ndarray
    source_indices: np.ndarray
    valid_shape: Tuple[int, int]
    geotransform: Tuple[float, float, float, float, float, float]
    min_x: float
    max_y: float


def _calculate_output_resolution(
    range_spacing: float,
    azimuth_spacing: float,
) -> float:
    max_spacing = max(range_spacing, azimuth_spacing)
    base_res = max_spacing * 2.0
    rounded = np.ceil(base_res * 2) / 2
    return float(rounded)


def _estimate_utm_zone(longitude: float) -> int:
    return int(np.floor((longitude + 180) / 6) + 1)


def _get_utm_epsg(latitude: float, longitude: float) -> Tuple[int, int]:
    zone = _estimate_utm_zone(longitude)
    epsg = 32600 + zone if latitude >= 0 else 32700 + zone
    return epsg, zone


def _radiometric_calibration(
    slc_data: np.ndarray,
    incidence_angle: np.ndarray,
    range_spacing: float,
    azimuth_spacing: float,
    prf: float,
    wavelength: float,
    beta_nought: float = 0.0,
) -> Dict[str, np.ndarray]:
    if slc_data.dtype.names is not None:
        real = np.asarray(slc_data["real"], dtype=np.float32)
        imag = np.asarray(slc_data["imag"], dtype=np.float32)
        power = real * real + imag * imag
    elif np.iscomplexobj(slc_data):
        real = slc_data.real.astype(np.float32, copy=False)
        imag = slc_data.imag.astype(np.float32, copy=False)
        power = real * real + imag * imag
    else:
        power = np.square(np.asarray(slc_data, dtype=np.float32))

    range_pixel_spacing_m = range_spacing
    azimuth_pixel_spacing_m = azimuth_spacing

    sigma0 = power * (range_pixel_spacing_m * azimuth_pixel_spacing_m) * (wavelength ** 2) / (
        (4 * np.pi ** 3) * (beta_nought + 1e-10)
    )

    cos_theta = np.cos(incidence_angle)
    cos_theta[cos_theta < 1e-10] = 1e-10
    gamma0 = sigma0 * cos_theta

    beta0 = sigma0 / cos_theta

    return {
        "sigma0": sigma0.astype(np.float32),
        "gamma0": gamma0.astype(np.float32),
        "beta0": beta0.astype(np.float32),
        "intensity": np.sqrt(power, dtype=np.float32),
    }


def _geocode_to_utm(
    data: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    dem_interpolator: DEMInterpolator,
    output_resolution: float,
    epsg: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    plan = _build_utm_geocode_plan(lat, lon, output_resolution, epsg)
    geocoded, valid_mask = _geocode_with_plan(data, plan)
    return geocoded, valid_mask, plan.geotransform, plan.min_x, plan.max_y


def _build_utm_geocode_plan(
    lat: np.ndarray,
    lon: np.ndarray,
    output_resolution: float,
    epsg: int,
) -> UTMGeocodePlan:
    valid_mask = np.isfinite(lat) & np.isfinite(lon)
    
    if not np.any(valid_mask):
        raise ValueError("No valid lat/lon data for geocoding")
    
    valid_lat = lat[valid_mask]
    valid_lon = lon[valid_mask]
    
    min_lon, max_lon = np.min(valid_lon), np.max(valid_lon)
    min_lat, max_lat = np.min(valid_lat), np.max(valid_lat)

    transformer = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    min_x, min_y = transformer.transform(min_lon, min_lat)
    max_x, max_y = transformer.transform(max_lon, max_lat)
    
    if np.isnan(min_x) or np.isnan(min_y) or np.isnan(max_x) or np.isnan(max_y):
        print(f"Warning: UTM conversion produced NaN values")
        print(f"  epsg: {epsg}")
        print(f"  min_lon={min_lon}, min_lat={min_lat}")
        print(f"  max_lon={max_lon}, max_lat={max_lat}")
        print(f"  min_x={min_x}, min_y={min_y}")
        print(f"  max_x={max_x}, max_y={max_y}")
        raise ValueError(f"UTM conversion failed for EPSG:{epsg}")

    min_x, max_x = min(min_x, max_x), max(min_x, max_x)
    min_y, max_y = min(min_y, max_y), max(min_y, max_y)

    padding = output_resolution * 5
    min_x -= padding
    max_x += padding
    min_y -= padding
    max_y += padding

    width = int(np.ceil((max_x - min_x) / output_resolution))
    height = int(np.ceil((max_y - min_y) / output_resolution))

    geotransform = (min_x, output_resolution, 0.0, max_y, 0.0, -output_resolution)

    lat_flat = lat.flatten()
    lon_flat = lon.flatten()

    valid_indices = np.isfinite(lat_flat) & np.isfinite(lon_flat)
    lat_valid = lat_flat[valid_indices]
    lon_valid = lon_flat[valid_indices]

    x_coords, y_coords = transformer.transform(lon_valid, lat_valid)

    pixel_x = ((x_coords - min_x) / output_resolution).astype(np.int32)
    pixel_y = ((max_y - y_coords) / output_resolution).astype(np.int32)

    valid_pixel_mask = (pixel_x >= 0) & (pixel_x < width) & (pixel_y >= 0) & (pixel_y < height)

    pixel_y = pixel_y[valid_pixel_mask]
    pixel_x = pixel_x[valid_pixel_mask]
    source_indices = np.flatnonzero(valid_indices)[valid_pixel_mask]

    output_indices = pixel_y * width + pixel_x
    _, reverse_unique = np.unique(output_indices[::-1], return_index=True)
    keep = len(output_indices) - 1 - reverse_unique
    keep.sort()

    return UTMGeocodePlan(
        pixel_y=pixel_y[keep],
        pixel_x=pixel_x[keep],
        source_indices=source_indices[keep],
        valid_shape=(height, width),
        geotransform=geotransform,
        min_x=min_x,
        max_y=max_y,
    )


def _geocode_with_plan(
    data: np.ndarray,
    plan: UTMGeocodePlan,
) -> Tuple[np.ndarray, np.ndarray]:
    geocoded = np.zeros(plan.valid_shape, dtype=np.float32)
    valid_mask = np.zeros(plan.valid_shape, dtype=bool)
    data_flat = np.asarray(data).ravel()

    geocoded[plan.pixel_y, plan.pixel_x] = data_flat[plan.source_indices]
    valid_mask[plan.pixel_y, plan.pixel_x] = True
    geocoded[~valid_mask] = np.nan

    return geocoded, valid_mask


def _apply_2percent_stretch(data: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    
    if valid_mask is None:
        valid_data = data[np.isfinite(data)]
    else:
        valid_data = data[valid_mask & np.isfinite(data)]
    
    if len(valid_data) == 0:
        return np.zeros_like(data, dtype=np.uint8)
    
    p2 = np.percentile(valid_data, 2)
    p98 = np.percentile(valid_data, 98)
    
    stretched = np.clip((data - p2) / (p98 - p2 + 1e-10), 0.0, 1.0)
    stretched = (stretched * 255).astype(np.uint8)
    
    nan_mask = ~np.isfinite(data)
    stretched[nan_mask] = 0
    
    if valid_mask is not None:
        stretched[~valid_mask] = 0
    
    return stretched


def _write_png(data: np.ndarray, valid_mask: np.ndarray, output_path: Path):
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Writing RTC PNG previews requires opencv-python or system cv2 bindings") from exc

    stretched = _apply_2percent_stretch(data, valid_mask)
    rgb = cv2.cvtColor(stretched, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(str(output_path), rgb)


class RTCProcessor:
    def __init__(
        self,
        scene_h5_path: Path,
        dem_h5_path: Path,
        output_resolution: Optional[float] = None,
        output_dir: Optional[Path] = None,
        full_resolution: bool = True,
    ):
        self.scene_h5_path = Path(scene_h5_path)
        self.dem_h5_path = Path(dem_h5_path)
        
        if output_dir is None:
            self.output_dir = Path.cwd()
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.output_resolution = output_resolution
        self.full_resolution = full_resolution
        self._scene_info = None
        self._radar_grid = None
        self._orbit = None
        self._dem = None
        self._scene_center_lat = 0.0
        self._scene_center_lon = 0.0

    def _load_scene_info(self):
        with h5py.File(self.scene_h5_path, "r") as h5:
            self._scene_info = {
                "sensor": h5.attrs.get("sensor", ""),
                "acquisition_mode": h5.attrs.get("acquisition_mode", ""),
            }
            
            import json
            if "metadata/acquisition" in h5:
                acq_json = h5["metadata/acquisition"].attrs.get("json", "{}")
                self._scene_info["acquisition"] = json.loads(acq_json)
            
            if "radar_grid" in h5:
                grid_json = h5["radar_grid"].attrs.get("json", "{}")
                self._radar_grid = json.loads(grid_json)
            
            if "derived" in h5:
                derived_json = h5["derived"].attrs.get("json", "{}")
                derived = json.loads(derived_json)
                if "sceneCenterCoord" in derived:
                    center = derived["sceneCenterCoord"]
                    self._scene_center_lat = center.get("lat", 0.0)
                    self._scene_center_lon = center.get("lon", 0.0)
            
            if self._scene_center_lat == 0.0 and self._scene_center_lon == 0.0:
                acq = self._scene_info.get("acquisition", {})
                self._scene_center_lat = acq.get("centerLat", 0.0)
                self._scene_center_lon = acq.get("centerLon", 0.0)
            
            if "orbit" in h5:
                time = h5["orbit/time"][:]
                position = h5["orbit/position"][:]
                velocity = h5["orbit/velocity"][:]
                self._orbit = OrbitInterpolator(time, position, velocity)

    def _load_dem(self):
        with h5py.File(self.dem_h5_path, "r") as h5:
            elevation = h5["dem/elevation"][:]
            geotransform = h5["dem"].attrs.get("geotransform", [0.0, 1.0, 0.0, 0.0, 0.0, -1.0])
            self._dem = DEMInterpolator(elevation, geotransform)

    def _get_step_size(self):
        num_rows = self._radar_grid.get("numberOfRows", 1000)
        num_cols = self._radar_grid.get("numberOfColumns", 1000)
        
        if self.full_resolution:
            return 1, 1, num_rows, num_cols
        
        step_row = max(50, num_rows // 50)
        step_col = max(50, num_cols // 50)
        
        out_rows = (num_rows + step_row - 1) // step_row
        out_cols = (num_cols + step_col - 1) // step_col
        
        return step_row, step_col, out_rows, out_cols

    def _compute_geometry(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._radar_grid is None or self._orbit is None or self._dem is None:
            raise ValueError("Scene info, radar grid, orbit, and DEM must be loaded first")
        
        radar_grid = RadarGrid.from_mapping(self._radar_grid)
        
        num_rows = self._radar_grid.get("numberOfRows", 1000)
        num_cols = self._radar_grid.get("numberOfColumns", 1000)
        
        step_row, step_col, _, _ = self._get_step_size()
        
        rows = np.arange(0, num_rows, step_row)
        cols = np.arange(0, num_cols, step_col)
        
        row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")
        
        flat_rows = row_grid.flatten()
        flat_cols = col_grid.flatten()
        
        lats, lons, heights = rdr2geo(
            line=flat_rows,
            pixel=flat_cols,
            radar_grid=radar_grid,
            satellite_position=self._orbit,
            velocity=None,
            doppler=0.0,
            dem=self._dem,
        )
        
        lats = lats.reshape(row_grid.shape)
        lons = lons.reshape(row_grid.shape)
        heights = heights.reshape(row_grid.shape)
        
        return lats, lons, heights

    def _compute_incidence_angle(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        height: np.ndarray,
    ) -> np.ndarray:
        if self._orbit is None or self._radar_grid is None:
            raise ValueError("Orbit and radar grid must be loaded")
        
        radar_grid = RadarGrid.from_mapping(self._radar_grid)
        
        step_row, step_col, out_rows, out_cols = self._get_step_size()
        rows = np.arange(0, self._radar_grid.get("numberOfRows", 1000), step_row)
        cols = np.arange(0, self._radar_grid.get("numberOfColumns", 1000), step_col)
        
        row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")
        az_times = radar_grid.line_to_azimuth_time(row_grid.flatten())
        
        lat_flat = lat.flatten()
        lon_flat = lon.flatten()
        height_flat = height.flatten()
        
        n_points = len(az_times)
        
        try:
            from i2sar.geometry.incidence_numba import compute_incidence_angle_numba
            
            orbit_times = self._orbit.time
            orbit_positions = self._orbit.position
            
            inc_angle = compute_incidence_angle_numba(
                lat_flat,
                lon_flat,
                height_flat,
                az_times,
                orbit_times,
                orbit_positions
            )
        except ImportError:
            from i2sar.geometry.ellipsoid import llh_to_ecef
            
            x, y, z = llh_to_ecef(lat_flat, lon_flat, height_flat)
            target_ecef = np.stack([x, y, z], axis=1)
            
            inc_angle = np.zeros(n_points, dtype=np.float64)
            
            for i in range(n_points):
                time = az_times[i]
                orbit_state = self._orbit.state_at(time, allow_extrapolation=True)
                sat_pos = orbit_state.position
                
                look_vec = target_ecef[i] - sat_pos
                look_vec_norm = np.linalg.norm(look_vec)
                look_dir = look_vec / look_vec_norm
                
                nadir_vec = -sat_pos / np.linalg.norm(sat_pos)
                
                cos_inc = np.dot(look_dir, nadir_vec)
                inc_angle[i] = np.arccos(np.clip(cos_inc, -1.0, 1.0))
        
        return inc_angle.reshape(lat.shape)

    def _write_geotiff(
        self,
        data: np.ndarray,
        geotransform: Tuple[float, float, float, float, float, float],
        epsg: int,
        output_path: Path,
    ):
        driver = gdal.GetDriverByName("GTiff")
        height, width = data.shape
        
        dst_ds = driver.Create(
            str(output_path),
            width,
            height,
            1,
            gdal.GDT_Float32,
            ["COMPRESS=LZW", "TILED=YES"],
        )
        
        dst_ds.SetGeoTransform(geotransform)
        dst_ds.SetProjection(f"EPSG:{epsg}")
        
        dst_ds.GetRasterBand(1).WriteArray(data)
        dst_ds.GetRasterBand(1).SetNoDataValue(np.nan)
        dst_ds.FlushCache()
        dst_ds = None

    def _write_hdf5(
        self,
        gamma0: np.ndarray,
        sigma0: np.ndarray,
        beta0: np.ndarray,
        intensity: np.ndarray,
        incidence_angle: np.ndarray,
        geotransform: Tuple[float, float, float, float, float, float],
        epsg: int,
        resolution: float,
        output_path: Path,
    ):
        with h5py.File(output_path, "w") as h5:
            h5.create_dataset("gamma0", data=gamma0)
            h5.create_dataset("sigma0", data=sigma0)
            h5.create_dataset("beta0", data=beta0)
            h5.create_dataset("intensity", data=intensity)
            h5.create_dataset("incidence_angle", data=incidence_angle)
            
            h5.attrs["epsg"] = epsg
            h5.attrs["resolution"] = resolution
            h5.attrs["geotransform"] = geotransform
            h5.attrs["width"] = gamma0.shape[1]
            h5.attrs["height"] = gamma0.shape[0]
            h5.attrs["full_resolution"] = self.full_resolution

    def process(self) -> RTCResult:
        self._load_scene_info()
        self._load_dem()
        
        if self.output_resolution is None:
            range_spacing = self._radar_grid.get("columnSpacing", 10.0)
            azimuth_spacing = self._radar_grid.get("rowSpacing", 10.0)
            self.output_resolution = _calculate_output_resolution(range_spacing, azimuth_spacing)
        
        step_row, step_col, out_rows, out_cols = self._get_step_size()
        
        print(f"RTC Processing: full_resolution={self.full_resolution}, step_row={step_row}, step_col={step_col}")
        print(f"Output dimensions: {out_rows} x {out_cols}")
        
        lat, lon, height = self._compute_geometry()
        
        valid_mask = np.isfinite(lat) & np.isfinite(lon)
        if np.any(valid_mask):
            center_lat = np.mean(lat[valid_mask])
            center_lon = np.mean(lon[valid_mask])
        else:
            center_lat = self._scene_center_lat
            center_lon = self._scene_center_lon
        
        epsg, utm_zone = _get_utm_epsg(center_lat, center_lon)
        print(f"Scene center: lat={center_lat:.4f}, lon={center_lon:.4f}, EPSG={epsg}, UTM zone={utm_zone}")
        
        inc_angle = self._compute_incidence_angle(lat, lon, height)
        
        range_spacing = self._radar_grid.get("columnSpacing", 10.0)
        azimuth_spacing = self._radar_grid.get("rowSpacing", 10.0)
        prf = self._radar_grid.get("prf", 1000.0)
        
        acq = self._scene_info.get("acquisition", {})
        wavelength = acq.get("centerFrequency", 9.6e9)
        wavelength = 299792458.0 / wavelength
        
        with h5py.File(self.scene_h5_path, "r") as h5:
            slc_data = h5["slc/data"][:]
        
        slc_subset = slc_data[::step_row, ::step_col]
        
        calib_result = _radiometric_calibration(
            slc_subset,
            inc_angle,
            range_spacing,
            azimuth_spacing,
            prf,
            wavelength,
        )
        
        geocode_plan = _build_utm_geocode_plan(
            lat,
            lon,
            self.output_resolution,
            epsg,
        )
        geotransform = geocode_plan.geotransform

        gamma0_geocoded, _ = _geocode_with_plan(calib_result["gamma0"], geocode_plan)
        sigma0_geocoded, _ = _geocode_with_plan(calib_result["sigma0"], geocode_plan)
        beta0_geocoded, _ = _geocode_with_plan(calib_result["beta0"], geocode_plan)
        intensity_geocoded, intensity_valid_mask = _geocode_with_plan(calib_result["intensity"], geocode_plan)
        inc_angle_geocoded, _ = _geocode_with_plan(inc_angle, geocode_plan)
        
        scene_id = self.scene_h5_path.stem.replace("scene_", "")
        
        suffix = "_full" if self.full_resolution else ""
        hdf5_file = self.output_dir / f"{scene_id}_rtc{suffix}.h5"
        gamma0_file = self.output_dir / f"{scene_id}_gamma0{suffix}.tif"
        sigma0_file = self.output_dir / f"{scene_id}_sigma0{suffix}.tif"
        beta0_file = self.output_dir / f"{scene_id}_beta0{suffix}.tif"
        incidence_angle_file = self.output_dir / f"{scene_id}_incidence_angle{suffix}.tif"
        intensity_file = self.output_dir / f"{scene_id}_intensity{suffix}.tif"
        intensity_png_file = self.output_dir / f"{scene_id}_intensity{suffix}.png"
        
        self._write_hdf5(
            gamma0_geocoded,
            sigma0_geocoded,
            beta0_geocoded,
            intensity_geocoded,
            inc_angle_geocoded,
            geotransform,
            epsg,
            self.output_resolution,
            hdf5_file,
        )
        
        self._write_geotiff(gamma0_geocoded, geotransform, epsg, gamma0_file)
        self._write_geotiff(sigma0_geocoded, geotransform, epsg, sigma0_file)
        self._write_geotiff(beta0_geocoded, geotransform, epsg, beta0_file)
        self._write_geotiff(intensity_geocoded, geotransform, epsg, intensity_file)
        self._write_geotiff(inc_angle_geocoded, geotransform, epsg, incidence_angle_file)
        
        _write_png(intensity_geocoded, intensity_valid_mask, intensity_png_file)
        
        return RTCResult(
            output_path=self.output_dir,
            hdf5_file=hdf5_file,
            gamma0_file=gamma0_file,
            sigma0_file=sigma0_file,
            beta0_file=beta0_file,
            incidence_angle_file=incidence_angle_file,
            intensity_file=intensity_file,
            intensity_png_file=intensity_png_file,
            resolution=self.output_resolution,
            epsg=epsg,
            utm_zone=utm_zone,
            output_rows=out_rows,
            output_cols=out_cols,
        )


def process_rtc(
    scene_h5_path: str | Path,
    dem_h5_path: str | Path,
    output_resolution: Optional[float] = None,
    output_dir: Optional[str | Path] = None,
    full_resolution: bool = True,
) -> Dict[str, Any]:
    processor = RTCProcessor(
        scene_h5_path=scene_h5_path,
        dem_h5_path=dem_h5_path,
        output_resolution=output_resolution,
        output_dir=output_dir,
        full_resolution=full_resolution,
    )
    
    result = processor.process()
    
    return {
        "output_path": str(result.output_path),
        "hdf5_file": str(result.hdf5_file),
        "gamma0_file": str(result.gamma0_file),
        "sigma0_file": str(result.sigma0_file),
        "beta0_file": str(result.beta0_file),
        "incidence_angle_file": str(result.incidence_angle_file),
        "intensity_file": str(result.intensity_file),
        "intensity_png_file": str(result.intensity_png_file),
        "resolution": result.resolution,
        "epsg": result.epsg,
        "utm_zone": result.utm_zone,
        "output_rows": result.output_rows,
        "output_cols": result.output_cols,
        "full_resolution": full_resolution,
    }
