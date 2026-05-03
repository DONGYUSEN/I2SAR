from __future__ import annotations

import numpy as np
from typing import Union, Optional

from i2sar.core.enums import LookSide
from i2sar.geometry.radar_grid import RadarGrid
from i2sar.orbit import OrbitInterpolator
from i2sar.geometry.rdr2geo import rdr2geo, rdr2geo_parallel, rdr2geo_arrayfire
from i2sar.geometry.geo2rdr import geo2rdr, geo2rdr_parallel, geo2rdr_arrayfire
from i2sar.geometry.rdr2geo_numba import rdr2geo_numba_parallel
from i2sar.geometry.geo2rdr_numba import geo2rdr_numba_parallel
from i2sar.geometry.geometry_arrayfire import rdr2geo_arrayfire_fast, rdr2geo_arrayfire_chunked


class AcceleratedRdr2Geo:
    def __init__(self):
        self._numba_compiled = False
        self._last_performance = {}
        self._af_available = self._check_arrayfire()
        self._gpu_memory_mb = _detect_gpu_memory()
        self._max_rows_per_chunk = _get_max_rows_by_memory(self._gpu_memory_mb)

    def _check_arrayfire(self) -> bool:
        try:
            from i2sar.accel.arrayfire_backend import arrayfire_available
            return arrayfire_available()
        except ImportError:
            return False

    def __call__(
        self,
        line: Union[int, float, np.ndarray],
        pixel: Union[int, float, np.ndarray],
        radar_grid: RadarGrid,
        satellite_position: Union[np.ndarray, OrbitInterpolator],
        velocity: Optional[np.ndarray] = None,
        doppler: float = 0.0,
        dem: Union[float, np.ndarray] = 0.0,
        method: str = "auto",
        **kwargs
    ) -> np.ndarray:
        line_arr = np.atleast_1d(np.asarray(line, dtype=np.float64))
        pixel_arr = np.atleast_1d(np.asarray(pixel, dtype=np.float64))
        n_points = len(line_arr)

        can_use_numba = isinstance(doppler, (int, float))
        can_use_af = (
            self._af_available 
            and not isinstance(satellite_position, OrbitInterpolator)
            and velocity is not None
            and isinstance(doppler, (int, float))
        )

        if method == "auto":
            if can_use_numba and n_points >= 100:
                method = "numba"
            elif n_points >= 100:
                method = "numba"
            else:
                method = "original"

        if method == "original":
            return rdr2geo(
                line=line,
                pixel=pixel,
                radar_grid=radar_grid,
                satellite_position=satellite_position,
                velocity=velocity,
                doppler=doppler,
                dem=dem,
                **kwargs
            )

        elif method == "multiprocessing":
            return rdr2geo_parallel(
                line=line,
                pixel=pixel,
                radar_grid=radar_grid,
                satellite_position=satellite_position,
                velocity=velocity,
                doppler=doppler,
                dem=dem,
                **kwargs
            )

        elif method == "numba":
            return self._numba_rdr2geo_with_chunking(line_arr, pixel_arr, radar_grid, satellite_position, velocity, doppler, dem)

        elif method == "arrayfire":
            return self._arrayfire_rdr2geo_optimized(line_arr, pixel_arr, radar_grid, satellite_position, velocity, doppler, dem)

        raise ValueError(f"unknown rdr2geo acceleration method: {method}")

    def _arrayfire_rdr2geo_optimized(self, line_arr, pixel_arr, radar_grid, satellite_position, velocity, doppler, dem):
        n_points = len(line_arr)
        num_cols = radar_grid.width if hasattr(radar_grid, 'width') else int(np.sqrt(n_points))
        num_rows = n_points // num_cols
        
        if num_rows == 0:
            num_rows = 1
            num_cols = n_points
        
        max_rows = self._max_rows_per_chunk
        
        print(f"[GPU] rdr2geo: {num_rows} rows × {num_cols} cols = {n_points:,} points, "
              f"GPU: {self._gpu_memory_mb} MB, chunk: {max_rows} rows")

        if isinstance(satellite_position, OrbitInterpolator):
            mid_time = satellite_position.reference_epoch + satellite_position.number_of_seconds / 2.0
            orbit_state = satellite_position.state_at(mid_time)
            sat_pos = orbit_state.position
            sat_vel = orbit_state.velocity
        else:
            sat_pos = satellite_position
            sat_vel = velocity

        if isinstance(dem, (int, float)):
            dem_arr = np.full(n_points, float(dem), dtype=np.float64)
        elif dem.ndim == 1:
            dem_arr = np.asarray(dem, dtype=np.float64)
        elif dem.ndim == 2:
            dem_arr = dem[line_arr.astype(np.int64), pixel_arr.astype(np.int64)].astype(np.float64)
        else:
            dem_arr = np.full(n_points, float(np.mean(dem)), dtype=np.float64)

        chunk_size = max_rows * num_cols
        if n_points <= chunk_size:
            try:
                return rdr2geo_arrayfire_fast(
                    line=line_arr,
                    pixel=pixel_arr,
                    radar_grid=radar_grid,
                    satellite_position=sat_pos,
                    velocity=sat_vel,
                    doppler=doppler,
                    dem=dem_arr,
                )
            except Exception as e:
                print(f"[ArrayFire] rdr2geo failed: {e}, falling back to Numba")
                return self._numba_rdr2geo_with_chunking(line_arr, pixel_arr, radar_grid, satellite_position, velocity, doppler, dem)

        total_chunks = (n_points + chunk_size - 1) // chunk_size
        
        result_lat = np.zeros(n_points, dtype=np.float64)
        result_lon = np.zeros(n_points, dtype=np.float64)
        result_hgt = np.zeros(n_points, dtype=np.float64)
        
        for chunk_idx in range(total_chunks):
            idx_start = chunk_idx * chunk_size
            idx_end = min(idx_start + chunk_size, n_points)
            
            print(f"\r[GPU] rdr2geo: {chunk_idx + 1}/{total_chunks}", end="", flush=True)
            
            try:
                chunk_result = rdr2geo_arrayfire_fast(
                    line=line_arr[idx_start:idx_end],
                    pixel=pixel_arr[idx_start:idx_end],
                    radar_grid=radar_grid,
                    satellite_position=sat_pos,
                    velocity=sat_vel,
                    doppler=doppler,
                    dem=dem_arr[idx_start:idx_end],
                )
            except Exception as e:
                print(f"\n[ArrayFire] chunk {chunk_idx} failed: {e}, falling back to Numba")
                chunk_result = self._numba_rdr2geo(
                    line_arr[idx_start:idx_end],
                    pixel_arr[idx_start:idx_end],
                    radar_grid,
                    satellite_position,
                    velocity,
                    doppler,
                    dem
                )
            
            result_lat[idx_start:idx_end] = chunk_result[0]
            result_lon[idx_start:idx_end] = chunk_result[1]
            result_hgt[idx_start:idx_end] = chunk_result[2]
        
        print()
        
        return np.stack([result_lat, result_lon, result_hgt], axis=0)

    def _arrayfire_rdr2geo(self, line_arr, pixel_arr, radar_grid, satellite_position, velocity, doppler, dem):
        n_points = len(line_arr)
        num_cols = radar_grid.width if hasattr(radar_grid, 'width') else int(np.sqrt(n_points))
        num_rows = n_points // num_cols
        
        if num_rows == 0:
            num_rows = 1
            num_cols = n_points
        
        max_rows = self._max_rows_per_chunk
        
        print(f"[GPU] rdr2geo: {num_rows} rows × {num_cols} cols = {n_points:,} points, "
              f"GPU: {self._gpu_memory_mb} MB, chunk: {max_rows} rows")

        if num_rows <= max_rows:
            return self._arrayfire_rdr2geo_single_chunk(line_arr, pixel_arr, radar_grid, 
                                                       satellite_position, velocity, doppler, dem)
        
        total_chunks = (num_rows + max_rows - 1) // max_rows
        num_points_per_chunk = max_rows * num_cols
        
        result_lat = np.zeros(n_points, dtype=np.float64)
        result_lon = np.zeros(n_points, dtype=np.float64)
        result_hgt = np.zeros(n_points, dtype=np.float64)
        
        for chunk_idx in range(total_chunks):
            idx_start = chunk_idx * num_points_per_chunk
            idx_end = min(idx_start + num_points_per_chunk, n_points)
            
            print(f"\r[GPU] rdr2geo: {chunk_idx + 1}/{total_chunks}", end="", flush=True)
            
            chunk_result = self._arrayfire_rdr2geo_single_chunk(
                line_arr[idx_start:idx_end], pixel_arr[idx_start:idx_end], radar_grid,
                satellite_position, velocity, doppler, dem
            )
            
            result_lat[idx_start:idx_end] = chunk_result[0]
            result_lon[idx_start:idx_end] = chunk_result[1]
            result_hgt[idx_start:idx_end] = chunk_result[2]
        
        print()
        
        return np.stack([result_lat, result_lon, result_hgt], axis=0)

    def _arrayfire_rdr2geo_single_chunk(self, line_arr, pixel_arr, radar_grid, satellite_position, velocity, doppler, dem):
        try:
            if isinstance(satellite_position, OrbitInterpolator):
                mid_time = satellite_position.reference_epoch + satellite_position.number_of_seconds / 2.0
                orbit_state = satellite_position.state_at(mid_time)
                sat_pos = orbit_state.position
                sat_vel = orbit_state.velocity
            else:
                sat_pos = satellite_position
                sat_vel = velocity

            if isinstance(dem, (int, float)):
                dem_arr = float(dem)
            elif hasattr(dem, 'interpolate'):
                from i2sar.geometry.rdr2geo import rdr2geo
                temp_result = rdr2geo(
                    line=line_arr,
                    pixel=pixel_arr,
                    radar_grid=radar_grid,
                    satellite_position=satellite_position,
                    velocity=None,
                    doppler=doppler,
                    dem=0.0
                )
                temp_lats, temp_lons = temp_result[0], temp_result[1]
                dem_arr = np.array([float(dem.interpolate(np.rad2deg(temp_lats[i]), np.rad2deg(temp_lons[i]))) 
                                   for i in range(len(line_arr))], dtype=np.float64)
            else:
                dem_arr = np.asarray(dem, dtype=np.float64)
                if dem_arr.ndim == 0:
                    dem_arr = float(dem_arr)
                elif dem_arr.ndim == 2:
                    dem_arr = dem_arr[line_arr.astype(np.int64), pixel_arr.astype(np.int64)]

            result = rdr2geo_arrayfire(
                line=line_arr,
                pixel=pixel_arr,
                radar_grid=radar_grid,
                satellite_position=sat_pos,
                velocity=sat_vel,
                doppler=doppler,
                dem=dem_arr,
            )

            if len(line_arr) == 1:
                return np.array([result[0], result[1], result[2]], dtype=np.float64)

            return result
        except Exception as e:
            print(f"[ArrayFire] rdr2geo GPU failed: {e}, falling back to Numba")
            return self._numba_rdr2geo(line_arr, pixel_arr, radar_grid, satellite_position, velocity, doppler, dem)

    def _numba_rdr2geo_with_chunking(self, line_arr, pixel_arr, radar_grid, satellite_position, velocity, doppler, dem):
        n_points = len(line_arr)
        num_cols = radar_grid.width if hasattr(radar_grid, 'width') else int(np.sqrt(n_points))
        num_rows = n_points // num_cols
        
        if num_rows == 0:
            num_rows = 1
            num_cols = n_points
        
        max_rows = self._max_rows_per_chunk
        chunk_size = max_rows * num_cols
        
        print(f"[Numba] rdr2geo: {num_rows} rows × {num_cols} cols = {n_points:,} points, chunk: {max_rows} rows")
        
        if n_points <= chunk_size:
            return self._numba_rdr2geo(line_arr, pixel_arr, radar_grid, satellite_position, velocity, doppler, dem)
        
        total_chunks = (n_points + chunk_size - 1) // chunk_size
        
        result_lat = np.zeros(n_points, dtype=np.float64)
        result_lon = np.zeros(n_points, dtype=np.float64)
        result_hgt = np.zeros(n_points, dtype=np.float64)
        
        for chunk_idx in range(total_chunks):
            idx_start = chunk_idx * chunk_size
            idx_end = min(idx_start + chunk_size, n_points)
            
            print(f"\r[Numba] rdr2geo: {chunk_idx + 1}/{total_chunks}", end="", flush=True)
            
            chunk_result = self._numba_rdr2geo(
                line_arr[idx_start:idx_end], 
                pixel_arr[idx_start:idx_end], 
                radar_grid,
                satellite_position, 
                velocity, 
                doppler, 
                dem
            )
            
            result_lat[idx_start:idx_end] = chunk_result[0]
            result_lon[idx_start:idx_end] = chunk_result[1]
            result_hgt[idx_start:idx_end] = chunk_result[2]
        
        print()
        
        return np.stack([result_lat, result_lon, result_hgt], axis=0)

    def _numba_rdr2geo(self, line_arr, pixel_arr, radar_grid, satellite_position, velocity, doppler, dem):
        a = 6378137.0
        b = 6356752.314245

        orbit_start_time = 0.0
        orbit_duration = 100.0

        if isinstance(satellite_position, OrbitInterpolator):
            times = np.linspace(
                satellite_position.reference_epoch,
                satellite_position.reference_epoch + satellite_position.number_of_seconds,
                1000
            )
            sat_positions = np.array([satellite_position.state_at(t).position for t in times], dtype=np.float64)
            sat_velocities = np.array([satellite_position.state_at(t).velocity for t in times], dtype=np.float64)
            orbit_start_time = float(satellite_position.reference_epoch)
            orbit_duration = float(satellite_position.number_of_seconds)
        else:
            sat_positions = np.array([satellite_position], dtype=np.float64)
            sat_velocities = np.array([velocity], dtype=np.float64)

        if isinstance(dem, (int, float)):
            dem_heights = np.ones(len(line_arr), dtype=np.float64) * float(dem)
        elif hasattr(dem, 'interpolate'):
            from i2sar.geometry.rdr2geo import rdr2geo
            
            temp_result = rdr2geo(
                line=line_arr,
                pixel=pixel_arr,
                radar_grid=radar_grid,
                satellite_position=satellite_position,
                velocity=None,
                doppler=doppler,
                dem=0.0
            )
            temp_lats, temp_lons = temp_result[0], temp_result[1]
            
            dem_heights = np.zeros(len(line_arr), dtype=np.float64)
            for i in range(len(line_arr)):
                dem_heights[i] = float(dem.interpolate(np.rad2deg(temp_lats[i]), np.rad2deg(temp_lons[i])))
        else:
            dem_arr = np.asarray(dem, dtype=np.float64)
            if dem_arr.ndim == 0:
                dem_heights = np.ones(len(line_arr), dtype=np.float64) * dem_arr.item()
            elif dem_arr.ndim == 1:
                dem_heights = dem_arr.astype(np.float64)
            else:
                dem_heights = np.ones(len(line_arr), dtype=np.float64) * float(np.mean(dem_arr))

        length = int(radar_grid.length) if hasattr(radar_grid, 'length') else 10000
        width = int(radar_grid.width) if hasattr(radar_grid, 'width') else 10000
        
        lats, lons, heights = rdr2geo_numba_parallel(
            line_arr=line_arr.astype(np.float64),
            pixel_arr=pixel_arr.astype(np.float64),
            sensing_start_s=float(radar_grid.sensing_start_s),
            prf_hz=float(radar_grid.prf_hz),
            starting_range_m=float(radar_grid.starting_range_m),
            range_pixel_spacing_m=float(radar_grid.range_pixel_spacing_m),
            sat_positions=sat_positions,
            sat_velocities=sat_velocities,
            doppler=float(doppler),
            wavelength=0.0565642,
            look_side_sign=-1.0,  # RIGHT look side (matches ISCE3 convention)
            dem_heights=dem_heights,
            a=a,
            b=b,
            max_iterations=25,
            extra_iterations=15,
            threshold=1e-8,
            orbit_start_time=orbit_start_time,
            orbit_duration=orbit_duration,
            length=length,
            width=width
        )

        if len(line_arr) == 1:
            return np.array([lats[0], lons[0], heights[0]], dtype=np.float64)

        return np.stack([lats, lons, heights], axis=0).astype(np.float64)


def _detect_gpu_memory() -> int:
    try:
        import arrayfire as af
        device_info = af.device_info()
        return int(device_info["global_mem_size"] / (1024 ** 2))
    except:
        return 6000


def _get_max_rows_by_memory(mem_total_mb: int) -> int:
    if mem_total_mb < 6000:
        return 128
    elif mem_total_mb < 12000:
        return 256
    elif mem_total_mb < 18000:
        return 512
    elif mem_total_mb < 24000:
        return 1024
    else:
        return 2048


class AcceleratedGeo2Rdr:
    def __init__(self):
        self._af_available = self._check_arrayfire()
        self._gpu_memory_mb = _detect_gpu_memory()
        self._max_rows_per_chunk = _get_max_rows_by_memory(self._gpu_memory_mb)

    def _check_arrayfire(self) -> bool:
        try:
            from i2sar.accel.arrayfire_backend import arrayfire_available
            return arrayfire_available()
        except ImportError:
            return False

    def __call__(
        self,
        lat: Union[int, float, np.ndarray],
        lon: Union[int, float, np.ndarray],
        height: Union[int, float, np.ndarray],
        radar_grid: RadarGrid,
        satellite_position: Union[np.ndarray, OrbitInterpolator],
        velocity: Optional[np.ndarray] = None,
        doppler: Union[float] = 0.0,
        method: str = "auto",
        **kwargs
    ) -> np.ndarray:
        lat_arr = np.atleast_1d(np.asarray(lat, dtype=np.float64))
        lon_arr = np.atleast_1d(np.asarray(lon, dtype=np.float64))
        h_arr = np.atleast_1d(np.asarray(height, dtype=np.float64))
        n_points = len(lat_arr)

        can_use_numba = (
            not isinstance(satellite_position, OrbitInterpolator)
            and isinstance(doppler, (int, float))
        )
        can_use_af = (
            self._af_available
            and not isinstance(satellite_position, OrbitInterpolator)
            and velocity is not None
            and isinstance(doppler, (int, float))
        )

        if method == "auto":
            if can_use_numba and n_points >= 100:
                method = "numba"
            elif n_points >= 100:
                method = "numba"
            else:
                method = "original"

        if method == "original":
            return geo2rdr(
                lat=lat,
                lon=lon,
                height=height,
                radar_grid=radar_grid,
                satellite_position=satellite_position,
                velocity=velocity,
                doppler=doppler,
                **kwargs
            )

        elif method == "multiprocessing":
            return geo2rdr_parallel(
                lat=lat,
                lon=lon,
                height=height,
                radar_grid=radar_grid,
                satellite_position=satellite_position,
                velocity=velocity,
                doppler=doppler,
                **kwargs
            )

        elif method == "numba":
            look_side = kwargs.get('look_side', LookSide.RIGHT)
            return self._numba_geo2rdr(lat_arr, lon_arr, h_arr, radar_grid, satellite_position, velocity, doppler, look_side)

        elif method == "arrayfire":
            look_side = kwargs.get('look_side', LookSide.RIGHT)
            return self._arrayfire_geo2rdr(lat_arr, lon_arr, h_arr, radar_grid, satellite_position, velocity, doppler, look_side)

        raise ValueError(f"unknown geo2rdr acceleration method: {method}")

    def _arrayfire_geo2rdr(self, lat_arr, lon_arr, h_arr, radar_grid, satellite_position, velocity, doppler, look_side):
        n_points = len(lat_arr)
        num_cols = radar_grid.width if hasattr(radar_grid, 'width') else int(np.sqrt(n_points))
        num_rows = n_points // num_cols
        
        if num_rows == 0:
            num_rows = 1
            num_cols = n_points
        
        max_rows = self._max_rows_per_chunk
        
        print(f"[GPU] geo2rdr: {num_rows} rows × {num_cols} cols = {n_points:,} points, "
              f"GPU: {self._gpu_memory_mb} MB, chunk: {max_rows} rows")

        if num_rows <= max_rows:
            return self._arrayfire_geo2rdr_single_chunk(lat_arr, lon_arr, h_arr, radar_grid,
                                                       satellite_position, velocity, doppler, look_side)
        
        total_chunks = (num_rows + max_rows - 1) // max_rows
        num_points_per_chunk = max_rows * num_cols
        
        result_az = np.zeros(n_points, dtype=np.float64)
        result_range = np.zeros(n_points, dtype=np.float64)
        
        for chunk_idx in range(total_chunks):
            idx_start = chunk_idx * num_points_per_chunk
            idx_end = min(idx_start + num_points_per_chunk, n_points)
            
            print(f"\r[GPU] geo2rdr: {chunk_idx + 1}/{total_chunks}", end="", flush=True)
            
            chunk_result = self._arrayfire_geo2rdr_single_chunk(
                lat_arr[idx_start:idx_end], lon_arr[idx_start:idx_end], h_arr[idx_start:idx_end], 
                radar_grid, satellite_position, velocity, doppler, look_side
            )
            
            result_az[idx_start:idx_end] = chunk_result[0]
            result_range[idx_start:idx_end] = chunk_result[1]
        
        print()
        
        return np.stack([result_az, result_range], axis=0)

    def _arrayfire_geo2rdr_single_chunk(self, lat_arr, lon_arr, h_arr, radar_grid, satellite_position, velocity, doppler, look_side):
        try:
            if isinstance(satellite_position, OrbitInterpolator):
                mid_time = satellite_position.reference_epoch + satellite_position.number_of_seconds / 2.0
                orbit_state = satellite_position.state_at(mid_time)
                sat_pos = orbit_state.position
                sat_vel = orbit_state.velocity
            else:
                sat_pos = satellite_position
                sat_vel = velocity

            result = geo2rdr_arrayfire(
                lat=lat_arr,
                lon=lon_arr,
                height=h_arr,
                radar_grid=radar_grid,
                satellite_position=sat_pos,
                velocity=sat_vel,
                doppler=doppler,
                look_side=look_side,
            )

            if len(lat_arr) == 1:
                return np.array([result[0], result[1]], dtype=np.float64)

            return result
        except Exception as e:
            print(f"[ArrayFire] geo2rdr GPU failed: {e}, falling back to Numba")
            return self._numba_geo2rdr(lat_arr, lon_arr, h_arr, radar_grid, satellite_position, velocity, doppler, look_side)

    def _numba_geo2rdr(self, lat_arr, lon_arr, h_arr, radar_grid, satellite_position, velocity, doppler, look_side):
        n_points = len(lat_arr)
        num_cols = radar_grid.width if hasattr(radar_grid, 'width') else int(np.sqrt(n_points))
        num_rows = n_points // num_cols
        
        if num_rows == 0:
            num_rows = 1
            num_cols = n_points
        
        max_rows = self._max_rows_per_chunk
        
        print(f"[Numba] geo2rdr: {num_rows} rows × {num_cols} cols = {n_points:,} points, chunk: {max_rows} rows")
        
        if num_rows <= max_rows:
            return self._numba_geo2rdr_single(lat_arr, lon_arr, h_arr, radar_grid, satellite_position, velocity, doppler, 0.0565642, look_side)
        
        total_chunks = (num_rows + max_rows - 1) // max_rows
        num_points_per_chunk = max_rows * num_cols
        
        result_az = np.zeros(n_points, dtype=np.float64)
        result_range = np.zeros(n_points, dtype=np.float64)
        
        for chunk_idx in range(total_chunks):
            idx_start = chunk_idx * num_points_per_chunk
            idx_end = min(idx_start + num_points_per_chunk, n_points)
            
            print(f"\r[Numba] geo2rdr: {chunk_idx + 1}/{total_chunks}", end="", flush=True)
            
            chunk_result = self._numba_geo2rdr_single(
                lat_arr[idx_start:idx_end], lon_arr[idx_start:idx_end], h_arr[idx_start:idx_end],
                radar_grid, satellite_position, velocity, doppler, 0.0565642, look_side
            )
            
            result_az[idx_start:idx_end] = chunk_result[0]
            result_range[idx_start:idx_end] = chunk_result[1]
        
        print()
        return np.stack([result_az, result_range], axis=0)
    
    def _numba_geo2rdr_single(self, lat_arr, lon_arr, h_arr, radar_grid, satellite_position, velocity, doppler, wavelength_m, look_side):
        a = 6378137.0
        b = 6356752.314245

        if isinstance(satellite_position, OrbitInterpolator):
            times = np.linspace(
                satellite_position.reference_epoch,
                satellite_position.reference_epoch + satellite_position.number_of_seconds,
                1000
            )
            sat_positions = np.array([satellite_position.state_at(t).position for t in times], dtype=np.float64)
            sat_velocities = np.array([satellite_position.state_at(t).velocity for t in times], dtype=np.float64)
        else:
            from i2sar.geometry.geo2rdr_numba import static_geo2rdr_numba

            aztimes, ranges = static_geo2rdr_numba(
                lat_arr=lat_arr.astype(np.float64),
                lon_arr=lon_arr.astype(np.float64),
                h_arr=h_arr.astype(np.float64),
                sat_pos=np.asarray(satellite_position, dtype=np.float64),
                vel=np.asarray(velocity, dtype=np.float64),
                sensing_start_s=float(radar_grid.sensing_start_s),
                prf_hz=float(radar_grid.prf_hz),
                length=int(radar_grid.length),
                a=a,
                b=b,
                doppler=float(doppler),
                wavelength=float(wavelength_m),
                look_side_right=(look_side == LookSide.RIGHT),
            )

            if len(lat_arr) == 1:
                return np.array([aztimes[0], ranges[0]], dtype=np.float64)

            return np.stack([aztimes, ranges], axis=0).astype(np.float64)

        aztimes, ranges = geo2rdr_numba_parallel(
            lat_arr=lat_arr.astype(np.float64),
            lon_arr=lon_arr.astype(np.float64),
            h_arr=h_arr.astype(np.float64),
            sat_positions=sat_positions,
            sat_velocities=sat_velocities,
            doppler=float(doppler),
            wavelength=0.0565642,
            a=a,
            b=b,
            max_iterations=50,
            threshold=1e-8,
            sensing_start_s=float(radar_grid.sensing_start_s),
            prf_hz=float(radar_grid.prf_hz),
            length=int(radar_grid.length),
            look_side_right=True
        )

        if len(lat_arr) == 1:
            return np.array([aztimes[0], ranges[0]], dtype=np.float64)

        return np.stack([aztimes, ranges], axis=0).astype(np.float64)


rdr2geo_fast = AcceleratedRdr2Geo()
geo2rdr_fast = AcceleratedGeo2Rdr()