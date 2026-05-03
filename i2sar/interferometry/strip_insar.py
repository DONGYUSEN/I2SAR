from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple
import numpy as np
import h5py

try:
    import arrayfire as af
    AF_AVAILABLE = True
    print("[ArrayFire] 已加载，使用GPU加速")
except ImportError:
    AF_AVAILABLE = False
    print("[ArrayFire] 未加载，使用CPU模式")

from i2sar.project import Project
from i2sar.io import import_scene
from i2sar.geometry.radar_grid import RadarGrid
from i2sar.geometry.accelerated_geometry import rdr2geo_fast, geo2rdr_fast
from i2sar.dem.interpolator import DEMInterpolator
from i2sar.dem.sampling import sample_dem_at_latlons
from i2sar.orbit.interpolate import read_orbit_interpolator
from i2sar.interferometry.interferogram import InterferogramGenerator
from i2sar.interferometry.filtering import goldstein_filter_vectorized
from i2sar.interferometry.phases import filter_and_remove_phase
from i2sar.interferometry.unwrapping import SnaphuUnwrapper, CostMode, InitMethod
from i2sar.interferometry.los import los_conversion
from i2sar.accel.arrayfire_filtering import (
    goldstein_filter_gpu,
    compute_coherence_gpu,
    phase_unwrap_gpu,
    arrayfire_available as af_filter_available,
)


def import_source_to_h5(source_path: str, sensor: str, work_dir: str, scene_name: str = "scene") -> str:
    """将原始数据导入为HDF5格式，返回场景HDF5文件路径"""
    project_dir = Path(work_dir) / f"project_{scene_name}"
    
    if project_dir.exists():
        import shutil
        shutil.rmtree(project_dir, ignore_errors=True)
    
    project = Project.create(project_dir, name=f"insar_temp_project_{scene_name}")
    
    try:
        import_result = import_scene(project, source_path, sensor=sensor)
        return str(import_result.scene_path)
    except Exception as e:
        if project_dir.exists():
            import shutil
            shutil.rmtree(project_dir, ignore_errors=True)
        raise e


@dataclass(frozen=True)
class PairContext:
    master_scene_path: Path
    slave_scene_path: Path
    output_root: Path
    pair_name: str
    wavelength: float


@dataclass(frozen=True)
class StageResult:
    success: bool
    outputs: Dict[str, Any]
    metadata: Dict[str, Any]


class StripInSARProcessor:
    STAGE_SEQUENCE = ("prep", "topo", "geo2rdr", "coarse_resample", 
                      "refine_offset", "refined_resample", "crossmul", 
                      "unwrap", "geocode", "product")
    STAGE_DIR_NAMES = {
        "prep": "prep",
        "topo": "p0_topo",
        "geo2rdr": "p1_geo2rdr",
        "coarse_resample": "p2_coarse_resample",
        "refine_offset": "p3_refine_offset",
        "refined_resample": "p4_refined_resample",
        "crossmul": "p5_crossmul",
        "unwrap": "p6_unwrap",
        "geocode": "p7_geocode",
        "product": "p8_product",
    }
    
    def __init__(self, context: PairContext, use_gpu: bool = True):
        self.context = context
        self._stage_records = {}
        self.use_gpu = use_gpu and AF_AVAILABLE
        
        if self.use_gpu:
            try:
                af.info()
                current_backend = af.get_backend()
                if current_backend == 'cuda':
                    print(f"[GPU] 使用CUDA后端")
                elif current_backend == 'opencl':
                    print(f"[GPU] 使用OpenCL后端")
                elif current_backend == 'unified':
                    print(f"[GPU] 使用Unified后端 (自动选择)")
                else:
                    print(f"[GPU] 使用{current_backend}后端")
            except Exception as e:
                print(f"[GPU] 初始化失败: {e}")
                self.use_gpu = False
                print(f"[GPU] 回退到CPU")
        
    def _stage_dir(self, stage: str) -> Path:
        return self.context.output_root / self.STAGE_DIR_NAMES[stage]
    
    def _write_stage_record(self, stage: str, record: Dict[str, Any]):
        stage_dir = self._stage_dir(stage)
        stage_dir.mkdir(parents=True, exist_ok=True)
        record_path = stage_dir / "stage.json"
        with open(record_path, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        self._stage_records[stage] = record
    
    def _load_stage_record(self, stage: str) -> Optional[Dict[str, Any]]:
        record_path = self._stage_dir(stage) / "stage.json"
        if record_path.exists():
            with open(record_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def _mark_stage_success(self, stage: str):
        success_path = self._stage_dir(stage) / "SUCCESS"
        success_path.write_text("success\n", encoding='utf-8')
    
    def stage_prep(self) -> StageResult:
        print("[prep] 加载主从影像数据...")
        
        with h5py.File(self.context.master_scene_path, 'r') as h5:
            master_grid_raw = h5["radar_grid"].attrs.get("json", "{}")
            master_grid = RadarGrid.from_mapping(json.loads(master_grid_raw))
            master_orbit = read_orbit_interpolator(self.context.master_scene_path)
            
            if "slc/data" in h5:
                master_slc = h5["slc/data"][:]
            else:
                master_slc = None
        
        with h5py.File(self.context.slave_scene_path, 'r') as h5:
            slave_grid_raw = h5["radar_grid"].attrs.get("json", "{}")
            slave_grid = RadarGrid.from_mapping(json.loads(slave_grid_raw))
            slave_orbit = read_orbit_interpolator(self.context.slave_scene_path)
            
            if "slc/data" in h5:
                slave_slc = h5["slc/data"][:]
            else:
                slave_slc = None
        
        self._stage_dir("prep").mkdir(parents=True, exist_ok=True)
        
        record = {
            "master_grid": asdict(master_grid),
            "slave_grid": asdict(slave_grid),
            "wavelength": self.context.wavelength,
            "use_gpu": self.use_gpu,
        }
        self._write_stage_record("prep", record)
        self._mark_stage_success("prep")
        
        print("[prep] 完成")
        return StageResult(
            success=True,
            outputs={
                "master_grid": master_grid,
                "slave_grid": slave_grid,
                "master_orbit": master_orbit,
                "slave_orbit": slave_orbit,
                "master_slc": master_slc,
                "slave_slc": slave_slc,
            },
            metadata=record
        )
    
    def stage_topo(self, dem_path: str, prep_result: StageResult) -> StageResult:
        print("[topo] 计算地形几何（全分辨率）...")
        
        master_grid = prep_result.outputs["master_grid"]
        master_orbit = prep_result.outputs["master_orbit"]
        
        dem = DEMInterpolator.from_hdf5(dem_path)
        
        num_rows = master_grid.length
        num_cols = master_grid.width
        
        rows = np.arange(num_rows)
        cols = np.arange(num_cols)
        row_grid, col_grid = np.meshgrid(rows, cols, indexing='ij')
        
        lats, lons, heights = rdr2geo_fast(
            line=row_grid.flatten(),
            pixel=col_grid.flatten(),
            radar_grid=master_grid,
            satellite_position=master_orbit,
            velocity=None,
            doppler=0.0,
            dem=0.0,
            method="auto"
        )
        
        dem_heights = sample_dem_at_latlons(
            lats, lons, dem.elevation, dem.geotransform, use_numba=True
        )
        
        incidence_angle = self._compute_incidence_angle(
            lats, lons, dem_heights, master_grid, master_orbit
        )
        
        stage_dir = self._stage_dir("topo")
        stage_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(stage_dir / "topo.h5", 'w') as h5:
            h5.create_dataset("lat", data=lats.reshape(row_grid.shape))
            h5.create_dataset("lon", data=lons.reshape(row_grid.shape))
            h5.create_dataset("height", data=dem_heights.reshape(row_grid.shape))
            h5.create_dataset("incidence_angle", data=incidence_angle.reshape(row_grid.shape))
        
        record = {
            "shape": (num_rows, num_cols),
        }
        self._write_stage_record("topo", record)
        self._mark_stage_success("topo")
        
        print("[topo] 完成")
        return StageResult(
            success=True,
            outputs={
                "lat": lats.reshape(row_grid.shape),
                "lon": lons.reshape(row_grid.shape),
                "height": dem_heights.reshape(row_grid.shape),
                "incidence_angle": incidence_angle.reshape(row_grid.shape),
            },
            metadata=record
        )
    
    def _compute_incidence_angle(self, lat, lon, height, radar_grid, orbit):
        from i2sar.geometry.ellipsoid import llh_to_ecef
        
        x, y, z = llh_to_ecef(lat, lon, height)
        target_ecef = np.stack([x, y, z], axis=1)
        
        num_cols = radar_grid.width
        num_rows = radar_grid.length
        
        times = radar_grid.line_to_azimuth_time(np.arange(num_rows))
        
        sat_positions = np.zeros((num_rows, 3), dtype=np.float64)
        for i in range(num_rows):
            orbit_state = orbit.state_at(times[i], allow_extrapolation=True)
            sat_positions[i] = orbit_state.position
        
        sat_positions_expanded = np.repeat(sat_positions, num_cols, axis=0)
        
        look_vec = target_ecef - sat_positions_expanded
        look_norm = np.linalg.norm(look_vec, axis=1)
        look_dir = look_vec / look_norm[:, np.newaxis]
        
        nadir_vec = -sat_positions_expanded / np.linalg.norm(sat_positions_expanded, axis=1)[:, np.newaxis]
        
        cos_inc = np.sum(look_dir * nadir_vec, axis=1)
        inc_angle = np.arccos(np.clip(cos_inc, -1.0, 1.0))
        
        return inc_angle
    
    def stage_geo2rdr(self, prep_result: StageResult, topo_result: StageResult, dem_path: str) -> StageResult:
        print("[geo2rdr] 执行地理坐标到雷达坐标转换（粗配准）...")
        
        master_grid = prep_result.outputs["master_grid"]
        slave_grid = prep_result.outputs["slave_grid"]
        slave_orbit = prep_result.outputs["slave_orbit"]
        
        lats = topo_result.outputs["lat"]
        lons = topo_result.outputs["lon"]
        heights = topo_result.outputs["height"]
        
        aztimes, slant_ranges = geo2rdr_fast(
            lat=lats.flatten(),
            lon=lons.flatten(),
            height=heights.flatten(),
            radar_grid=slave_grid,
            satellite_position=slave_orbit,
            velocity=None,
            doppler=0.0,
            method="auto"
        )
        
        lines = slave_grid.azimuth_time_to_line(aztimes)
        pixels = slave_grid.slant_range_to_pixel(slant_ranges)
        
        lines = lines.reshape(lats.shape)
        pixels = pixels.reshape(lons.shape)
        
        valid_mask = ~(np.isnan(lines) | np.isnan(pixels))
        
        lines = np.nan_to_num(lines, nan=0.0)
        pixels = np.nan_to_num(pixels, nan=0.0)
        
        az_offset = lines - np.arange(master_grid.length).reshape(-1, 1)
        rg_offset = pixels - np.arange(master_grid.width).reshape(1, -1)
        
        az_offset[~valid_mask] = np.nan
        rg_offset[~valid_mask] = np.nan
        
        stage_dir = self._stage_dir("geo2rdr")
        stage_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(stage_dir / "geo2rdr.h5", 'w') as h5:
            h5.create_dataset("lines", data=lines)
            h5.create_dataset("pixels", data=pixels)
            h5.create_dataset("az_offset", data=az_offset)
            h5.create_dataset("rg_offset", data=rg_offset)
            h5.create_dataset("valid_mask", data=valid_mask)
        
        self._save_offset_images(stage_dir, az_offset, rg_offset, valid_mask)
        
        valid_az_offset = az_offset[valid_mask]
        valid_rg_offset = rg_offset[valid_mask]
        
        record = {
            "shape": lats.shape,
            "mean_az_offset": float(np.mean(valid_az_offset)) if valid_az_offset.size > 0 else np.nan,
            "mean_rg_offset": float(np.mean(valid_rg_offset)) if valid_rg_offset.size > 0 else np.nan,
            "valid_points": int(np.sum(valid_mask)),
            "total_points": int(np.prod(lats.shape)),
        }
        self._write_stage_record("geo2rdr", record)
        self._mark_stage_success("geo2rdr")
        
        print(f"[geo2rdr] 完成，有效点: {record['valid_points']}/{record['total_points']}")
        return StageResult(
            success=True,
            outputs={
                "lines": lines,
                "pixels": pixels,
                "az_offset": az_offset,
                "rg_offset": rg_offset,
                "valid_mask": valid_mask,
            },
            metadata=record
        )
    
    def _save_offset_images(self, stage_dir, az_offset, rg_offset, valid_mask):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        rg_offset_display = np.copy(rg_offset)
        rg_offset_display[~valid_mask] = np.nan
        
        plt.figure(figsize=(10, 6))
        plt.imshow(rg_offset_display, cmap='jet', interpolation='nearest')
        plt.colorbar(label='Range Offset (pixels)')
        plt.title('Range Offset (Invalid pixels masked)')
        plt.xlabel('Range')
        plt.ylabel('Azimuth')
        plt.savefig(stage_dir / "rg_offset.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        az_offset_display = np.copy(az_offset)
        az_offset_display[~valid_mask] = np.nan
        
        plt.figure(figsize=(10, 6))
        plt.imshow(az_offset_display, cmap='jet', interpolation='nearest')
        plt.colorbar(label='Azimuth Offset (pixels)')
        plt.title('Azimuth Offset (Invalid pixels masked)')
        plt.xlabel('Range')
        plt.ylabel('Azimuth')
        plt.savefig(stage_dir / "az_offset.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[geo2rdr] 已生成偏移量图片: {stage_dir / 'rg_offset.png'}")
    
    def stage_coarse_resample(self, prep_result: StageResult, geo2rdr_result: StageResult) -> StageResult:
        print("[coarse_resample] 使用geo2rdr结果进行粗重采样...")
        
        master_slc = prep_result.outputs["master_slc"]
        slave_slc = prep_result.outputs["slave_slc"]
        master_grid = prep_result.outputs["master_grid"]
        
        if master_slc is None or slave_slc is None:
            print("[coarse_resample] 警告：缺少SLC数据")
            self._mark_stage_success("coarse_resample")
            return StageResult(
                success=True,
                outputs={"resampled_slave": None},
                metadata={"status": "skipped"}
            )
        
        az_offset = geo2rdr_result.outputs["az_offset"]
        rg_offset = geo2rdr_result.outputs["rg_offset"]
        
        master_slc = self._to_complex(master_slc)
        slave_slc = self._to_complex(slave_slc)
        
        num_rows = master_grid.length
        num_cols = master_grid.width
        
        resampled_slave = self._resample_slc(slave_slc, az_offset, rg_offset, 1, num_rows, num_cols)
        
        stage_dir = self._stage_dir("coarse_resample")
        stage_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(stage_dir / "resampled_slave.h5", 'w') as h5:
            h5.create_dataset("data", data=resampled_slave)
        
        record = {
            "shape": resampled_slave.shape,
        }
        self._write_stage_record("coarse_resample", record)
        self._mark_stage_success("coarse_resample")
        
        print("[coarse_resample] 完成")
        return StageResult(
            success=True,
            outputs={
                "resampled_slave": resampled_slave,
            },
            metadata=record
        )
    
    def _to_complex(self, arr: np.ndarray) -> np.ndarray:
        if arr.dtype.names is not None:
            return arr['real'] + 1j * arr['imag']
        return arr
    
    def _resample_slc(self, slave_slc, az_offset, rg_offset, step, num_rows, num_cols):
        from scipy.interpolate import RegularGridInterpolator
        
        rows = np.arange(0, num_rows, step)
        cols = np.arange(0, num_cols, step)
        
        interp_az = RegularGridInterpolator(
            (rows, cols), az_offset, method='linear', bounds_error=False, fill_value=0
        )
        interp_rg = RegularGridInterpolator(
            (rows, cols), rg_offset, method='linear', bounds_error=False, fill_value=0
        )
        
        full_rows = np.arange(num_rows)
        full_cols = np.arange(num_cols)
        grid_rows, grid_cols = np.meshgrid(full_rows, full_cols, indexing='ij')
        
        fine_az_offset = interp_az((grid_rows, grid_cols))
        fine_rg_offset = interp_rg((grid_rows, grid_cols))
        
        sample_rows = grid_rows + fine_az_offset
        sample_cols = grid_cols + fine_rg_offset
        
        valid_mask = (sample_rows >= 0) & (sample_rows < slave_slc.shape[0] - 1) & \
                     (sample_cols >= 0) & (sample_cols < slave_slc.shape[1] - 1) & \
                     ~np.isnan(sample_rows) & ~np.isnan(sample_cols)
        
        sample_rows_clipped = np.clip(sample_rows, 0, slave_slc.shape[0] - 2)
        sample_cols_clipped = np.clip(sample_cols, 0, slave_slc.shape[1] - 2)
        
        sample_rows_clipped = np.nan_to_num(sample_rows_clipped, nan=0.0)
        sample_cols_clipped = np.nan_to_num(sample_cols_clipped, nan=0.0)
        
        row_floor = np.floor(sample_rows_clipped).astype(np.int64)
        col_floor = np.floor(sample_cols_clipped).astype(np.int64)
        
        row_floor = np.clip(row_floor, 0, slave_slc.shape[0] - 2)
        col_floor = np.clip(col_floor, 0, slave_slc.shape[1] - 2)
        
        row_frac = sample_rows_clipped - row_floor
        col_frac = sample_cols_clipped - col_floor
        
        resampled = (1 - row_frac) * (1 - col_frac) * slave_slc[row_floor, col_floor] + \
                    (1 - row_frac) * col_frac * slave_slc[row_floor, col_floor + 1] + \
                    row_frac * (1 - col_frac) * slave_slc[row_floor + 1, col_floor] + \
                    row_frac * col_frac * slave_slc[row_floor + 1, col_floor + 1]
        
        resampled[~valid_mask] = 0
        
        return resampled
    
    def stage_refine_offset(self, prep_result: StageResult, coarse_result: StageResult) -> StageResult:
        print("[refine_offset] 使用互相关精化偏移估计...")
        
        master_slc = prep_result.outputs["master_slc"]
        resampled_slave = coarse_result.outputs["resampled_slave"]
        
        if master_slc is None or resampled_slave is None:
            print("[refine_offset] 警告：缺少SLC数据，跳过精化")
            self._mark_stage_success("refine_offset")
            return StageResult(
                success=True,
                outputs={"az_offset": None, "rg_offset": None},
                metadata={"status": "skipped"}
            )
        
        master_slc = self._to_complex(master_slc)
        
        if self.use_gpu:
            offsets_az, offsets_rg = self._estimate_offsets_gpu(master_slc, resampled_slave)
        else:
            offsets_az, offsets_rg = self._estimate_offsets_cpu(master_slc, resampled_slave)
        
        stage_dir = self._stage_dir("refine_offset")
        stage_dir.mkdir(parents=True, exist_ok=True)
        np.save(stage_dir / "offsets_az.npy", offsets_az)
        np.save(stage_dir / "offsets_rg.npy", offsets_rg)
        
        record = {
            "shape": offsets_az.shape,
            "mean_az_offset": float(np.mean(offsets_az)),
            "mean_rg_offset": float(np.mean(offsets_rg)),
            "use_gpu": self.use_gpu,
        }
        self._write_stage_record("refine_offset", record)
        self._mark_stage_success("refine_offset")
        
        print("[refine_offset] 完成")
        return StageResult(
            success=True,
            outputs={
                "az_offset": offsets_az,
                "rg_offset": offsets_rg,
            },
            metadata=record
        )
    
    def _estimate_offsets_cpu(self, master, slave):
        from scipy.fft import fft2, ifft2, fftshift
        
        h, w = master.shape
        window = 256
        search_radius = 16
        
        num_tiles_row = max(1, h // (window * 2))
        num_tiles_col = max(1, w // (window * 2))
        
        offsets_az = np.zeros((h, w), dtype=np.float64)
        offsets_rg = np.zeros((h, w), dtype=np.float64)
        
        for i in range(num_tiles_row + 1):
            for j in range(num_tiles_col + 1):
                row = min(i * window, h - window)
                col = min(j * window, w - window)
                
                master_crop = master[row:row+window, col:col+window]
                slave_crop = slave[row:row+window, col:col+window]
                
                master_fft = fft2(master_crop)
                slave_fft = fft2(slave_crop)
                corr = ifft2(master_fft * np.conj(slave_fft))
                corr = np.abs(fftshift(corr))
                
                center = window // 2
                y, x = np.unravel_index(
                    np.argmax(corr[center-search_radius:center+search_radius, 
                                   center-search_radius:center+search_radius]),
                    (search_radius * 2, search_radius * 2)
                )
                offset_y = y - search_radius
                offset_x = x - search_radius
                
                offsets_az[row:row+window, col:col+window] = offset_y
                offsets_rg[row:row+window, col:col+window] = offset_x
        
        return offsets_az, offsets_rg
    
    def _estimate_offsets_gpu(self, master, slave):
        h, w = master.shape
        window = 256
        search_radius = 16
        
        num_tiles_row = max(1, h // (window * 2))
        num_tiles_col = max(1, w // (window * 2))
        
        total_tiles = (num_tiles_row + 1) * (num_tiles_col + 1)
        current_tile = 0
        
        offsets_az = np.zeros((h, w), dtype=np.float64)
        offsets_rg = np.zeros((h, w), dtype=np.float64)
        
        try:
            for i in range(num_tiles_row + 1):
                for j in range(num_tiles_col + 1):
                    current_tile += 1
                    print(f"\r[GPU] offset: {current_tile}/{total_tiles}", end="", flush=True)
                    
                    row = min(i * window, h - window)
                    col = min(j * window, w - window)
                    
                    master_crop = master[row:row+window, col:col+window]
                    slave_crop = slave[row:row+window, col:col+window]
                    
                    master_crop = np.ascontiguousarray(master_crop, dtype=np.complex64)
                    slave_crop = np.ascontiguousarray(slave_crop, dtype=np.complex64)
                    
                    master_af = af.Array(master_crop)
                    slave_af = af.Array(slave_crop)
                    
                    master_fft = af.fft2(master_af)
                    slave_fft = af.fft2(slave_af)
                    corr = af.ifft2(master_fft * af.conjg(slave_fft))
                    corr = af.abs(af.fft_shift(corr))
                    
                    corr_np = corr.to_array()
                    
                    center = window // 2
                    y, x = np.unravel_index(
                        np.argmax(corr_np[center-search_radius:center+search_radius, 
                                          center-search_radius:center+search_radius]),
                        (search_radius * 2, search_radius * 2)
                    )
                    offset_y = y - search_radius
                    offset_x = x - search_radius
                    
                    offsets_az[row:row+window, col:col+window] = offset_y
                    offsets_rg[row:row+window, col:col+window] = offset_x
            
            print()
            return offsets_az, offsets_rg
        except Exception as e:
            print(f"\n[GPU] offset estimation failed: {e}, falling back to CPU")
            return self._estimate_offsets_cpu(master, slave)
    
    def stage_refined_resample(self, prep_result: StageResult, coarse_result: StageResult, 
                               refine_result: StageResult) -> StageResult:
        print("[refined_resample] 使用精化偏移进行精确重采样...")
        
        slave_slc = prep_result.outputs["slave_slc"]
        master_grid = prep_result.outputs["master_grid"]
        resampled_slave = coarse_result.outputs["resampled_slave"]
        az_offset = refine_result.outputs["az_offset"]
        rg_offset = refine_result.outputs["rg_offset"]
        
        if slave_slc is None or resampled_slave is None or az_offset is None:
            print("[refined_resample] 警告：使用粗重采样结果")
            self._mark_stage_success("refined_resample")
            return StageResult(
                success=True,
                outputs={"refined_slave": resampled_slave},
                metadata={"status": "used_coarse"}
            )
        
        slave_slc = self._to_complex(slave_slc)
        
        num_rows = master_grid.length
        num_cols = master_grid.width
        
        if self.use_gpu:
            refined_slave = self._refined_resample_gpu(slave_slc, az_offset, rg_offset, num_rows, num_cols)
        else:
            refined_slave = self._refined_resample_cpu(slave_slc, az_offset, rg_offset, num_rows, num_cols)
        
        stage_dir = self._stage_dir("refined_resample")
        stage_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(stage_dir / "refined_slave.h5", 'w') as h5:
            h5.create_dataset("data", data=refined_slave)
        
        record = {
            "shape": refined_slave.shape,
            "use_gpu": self.use_gpu,
        }
        self._write_stage_record("refined_resample", record)
        self._mark_stage_success("refined_resample")
        
        print("[refined_resample] 完成")
        return StageResult(
            success=True,
            outputs={
                "refined_slave": refined_slave,
            },
            metadata=record
        )
    
    def _refined_resample_cpu(self, slave_slc, az_offset, rg_offset, num_rows, num_cols):
        grid_rows, grid_cols = np.meshgrid(np.arange(num_rows), np.arange(num_cols), indexing='ij')
        
        sample_rows = grid_rows + az_offset
        sample_cols = grid_cols + rg_offset
        
        valid_mask = (sample_rows >= 0) & (sample_rows < slave_slc.shape[0] - 1) & \
                     (sample_cols >= 0) & (sample_cols < slave_slc.shape[1] - 1)
        
        sample_rows_clipped = np.clip(sample_rows, 0, slave_slc.shape[0] - 2)
        sample_cols_clipped = np.clip(sample_cols, 0, slave_slc.shape[1] - 2)
        
        row_floor = np.floor(sample_rows_clipped).astype(int)
        col_floor = np.floor(sample_cols_clipped).astype(int)
        row_frac = sample_rows_clipped - row_floor
        col_frac = sample_cols_clipped - col_floor
        
        refined_slave = (1 - row_frac) * (1 - col_frac) * slave_slc[row_floor, col_floor] + \
                        (1 - row_frac) * col_frac * slave_slc[row_floor, col_floor + 1] + \
                        row_frac * (1 - col_frac) * slave_slc[row_floor + 1, col_floor] + \
                        row_frac * col_frac * slave_slc[row_floor + 1, col_floor + 1]
        
        refined_slave[~valid_mask] = 0
        
        return refined_slave
    
    def _refined_resample_gpu(self, slave_slc, az_offset, rg_offset, num_rows, num_cols):
        try:
            slave_slc = np.ascontiguousarray(slave_slc, dtype=np.complex64)
            az_offset = np.ascontiguousarray(az_offset, dtype=np.float64)
            rg_offset = np.ascontiguousarray(rg_offset, dtype=np.float64)
            
            slave_af = af.Array(slave_slc)
            
            rows_af = af.range(num_rows, 1, dtype=af.Dtype.f64)
            cols_af = af.range(num_cols, 1, dtype=af.Dtype.f64)
            
            grid_rows_af = af.tile(rows_af, 1, num_cols)
            grid_cols_af = af.tile(af.transpose(cols_af), num_rows, 1)
            
            az_offset_af = af.Array(az_offset)
            rg_offset_af = af.Array(rg_offset)
            
            sample_rows = grid_rows_af + az_offset_af
            sample_cols = grid_cols_af + rg_offset_af
            
            sample_rows = af.clamp(sample_rows, 0, slave_slc.shape[0] - 2)
            sample_cols = af.clamp(sample_cols, 0, slave_slc.shape[1] - 2)
            
            row_floor = af.floor(sample_rows).as_type(af.Dtype.s32)
            col_floor = af.floor(sample_cols).as_type(af.Dtype.s32)
            row_frac = sample_rows - af.floor(sample_rows)
            col_frac = sample_cols - af.floor(sample_cols)
            
            val00 = slave_af(row_floor, col_floor)
            val01 = slave_af(row_floor, col_floor + 1)
            val10 = slave_af(row_floor + 1, col_floor)
            val11 = slave_af(row_floor + 1, col_floor + 1)
        
            refined_slave = (1 - row_frac) * (1 - col_frac) * val00 + \
                            (1 - row_frac) * col_frac * val01 + \
                            row_frac * (1 - col_frac) * val10 + \
                            row_frac * col_frac * val11
            
            valid_mask = (sample_rows >= 0) & (sample_cols >= 0) & \
                         (sample_rows < slave_slc.shape[0] - 1) & \
                         (sample_cols < slave_slc.shape[1] - 1)
            
            refined_slave = af.select(valid_mask, refined_slave, 0)
            
            return refined_slave.to_array()
        except Exception as e:
            print(f"\n[GPU] refined resample failed: {e}, falling back to CPU")
            return self._refined_resample_cpu(slave_slc, az_offset, rg_offset, num_rows, num_cols)
    
    def stage_crossmul(self, prep_result: StageResult, refined_result: StageResult, 
                       topo_result: Optional[StageResult] = None) -> StageResult:
        print("[crossmul] 生成干涉图...")
        
        master_slc = prep_result.outputs["master_slc"]
        refined_slave = refined_result.outputs["refined_slave"]
        
        if master_slc is None or refined_slave is None:
            print("[crossmul] 错误：缺少SLC数据")
            return StageResult(
                success=False,
                outputs={},
                metadata={"error": "缺少SLC数据"}
            )
        
        master_slc = self._to_complex(master_slc)
        
        if self.use_gpu:
            interferogram, coherence = self._compute_interferogram_gpu(master_slc, refined_slave)
        else:
            generator = InterferogramGenerator(use_fft=True)
            interferogram = generator._compute_interferogram(master_slc, refined_slave)
            coherence = generator._compute_coherence(master_slc, refined_slave, window_size=5)
        
        if topo_result is not None and topo_result.outputs.get("height") is not None:
            dem_height = topo_result.outputs["height"]
            incidence_angle = topo_result.outputs.get("incidence_angle")
            
            if dem_height is not None and incidence_angle is not None:
                print("[crossmul] 去除地形相位...")
                phase_result = filter_and_remove_phase(
                    interferogram, coherence,
                    dem_height, incidence_angle, self.context.wavelength,
                    alpha=1.0, window_size=32
                )
                filtered = phase_result["corrected"]
                topo_phase = phase_result["topo_phase"]
            else:
                if self.use_gpu and af_filter_available():
                    print("[crossmul] 应用Goldstein滤波（GPU加速）...")
                    filt_result = goldstein_filter_gpu(
                        interferogram, coherence, alpha=1.0, window_size=32
                    )
                else:
                    print("[crossmul] 应用Goldstein滤波...")
                    filt_result = goldstein_filter_vectorized(
                        interferogram, coherence, alpha=1.0, window_size=32
                    )
                filtered = filt_result["filtered"]
                topo_phase = None
        else:
            if self.use_gpu and af_filter_available():
                print("[crossmul] 应用Goldstein滤波（GPU加速）...")
                filt_result = goldstein_filter_gpu(
                    interferogram, coherence, alpha=1.0, window_size=32
                )
            else:
                print("[crossmul] 应用Goldstein滤波...")
                filt_result = goldstein_filter_vectorized(
                    interferogram, coherence, alpha=1.0, window_size=32
                )
            filtered = filt_result["filtered"]
            topo_phase = None
        
        stage_dir = self._stage_dir("crossmul")
        stage_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(stage_dir / "interferogram.h5", 'w') as h5:
            h5.create_dataset("interferogram", data=interferogram)
            h5.create_dataset("coherence", data=coherence)
            h5.create_dataset("filtered", data=filtered)
            if topo_phase is not None:
                h5.create_dataset("topo_phase", data=topo_phase)
        
        record = {
            "shape": interferogram.shape,
            "mean_coherence": float(np.mean(coherence)),
            "topo_phase_removed": topo_phase is not None,
            "use_gpu": self.use_gpu,
        }
        self._write_stage_record("crossmul", record)
        self._mark_stage_success("crossmul")
        
        print("[crossmul] 完成")
        outputs = {
            "interferogram": interferogram,
            "coherence": coherence,
            "filtered": filtered,
        }
        if topo_phase is not None:
            outputs["topo_phase"] = topo_phase
        
        return StageResult(
            success=True,
            outputs=outputs,
            metadata=record
        )
    
    def _compute_interferogram_cpu(self, master, slave, use_numba: bool = True):
        if use_numba:
            try:
                return self._compute_interferogram_numba(master, slave)
            except Exception as e:
                print(f"[Numba] interferogram computation failed: {e}, falling back to scipy")
        
        from scipy.signal import fftconvolve
        
        master = master.astype(np.complex128)
        slave = slave.astype(np.complex128)
        
        interferogram = master * np.conj(slave)
        
        window_size = 5
        ones = np.ones((window_size, window_size))
        
        master_mag_sq = np.abs(master) ** 2
        slave_mag_sq = np.abs(slave) ** 2
        
        master_sum = fftconvolve(master_mag_sq, ones, mode="same") / (window_size ** 2)
        slave_sum = fftconvolve(slave_mag_sq, ones, mode="same") / (window_size ** 2)
        
        cross = master * np.conj(slave)
        cross_sum = fftconvolve(np.abs(cross), ones, mode="same") / (window_size ** 2)
        
        coherence = cross_sum / np.sqrt(master_sum * slave_sum + 1e-10)
        coherence = np.clip(coherence, 0, 1)
        
        return interferogram, coherence
    
    def _compute_interferogram_numba(self, master, slave):
        from numba import njit, prange
        
        master = np.ascontiguousarray(master, dtype=np.complex128)
        slave = np.ascontiguousarray(slave, dtype=np.complex128)
        
        interferogram = master * np.conj(slave)
        
        window_size = 5
        half_win = window_size // 2
        h, w = master.shape
        
        master_mag_sq = np.abs(master) ** 2
        slave_mag_sq = np.abs(slave) ** 2
        cross_abs = np.abs(master * np.conj(slave))
        
        @njit(parallel=True, fastmath=True)
        def compute_coherence(master_sq, slave_sq, cross_abs_arr, h, w, window_size, half_win):
            coherence = np.zeros((h, w), dtype=np.float64)
            for i in prange(h):
                for j in range(w):
                    row_start = max(0, i - half_win)
                    row_end = min(h, i + half_win + 1)
                    col_start = max(0, j - half_win)
                    col_end = min(w, j + half_win + 1)
                    
                    master_sum = 0.0
                    slave_sum = 0.0
                    cross_sum = 0.0
                    count = 0
                    
                    for ii in range(row_start, row_end):
                        for jj in range(col_start, col_end):
                            master_sum += master_sq[ii, jj]
                            slave_sum += slave_sq[ii, jj]
                            cross_sum += cross_abs_arr[ii, jj]
                            count += 1
                    
                    if count > 0:
                        master_sum /= count
                        slave_sum /= count
                        cross_sum /= count
                        coherence[i, j] = cross_sum / np.sqrt(master_sum * slave_sum + 1e-10)
            
            return coherence
        
        coherence = compute_coherence(master_mag_sq, slave_mag_sq, cross_abs, h, w, window_size, half_win)
        coherence = np.clip(coherence, 0.0, 1.0)
        
        return interferogram, coherence
    
    def _compute_interferogram_gpu(self, master, slave):
        try:
            master = np.ascontiguousarray(master, dtype=np.complex64)
            slave = np.ascontiguousarray(slave, dtype=np.complex64)
            
            master_af = af.Array(master)
            slave_af = af.Array(slave)
            
            interferogram_af = master_af * af.conjg(slave_af)
            interferogram = interferogram_af.to_array()
            
            window_size = 5
            half_win = window_size // 2
            
            master_mag_sq = af.abs(master_af) ** 2
            slave_mag_sq = af.abs(slave_af) ** 2
            
            master_sum = af.convolve(master_mag_sq, af.constant(1.0 / (window_size ** 2), window_size, window_size))
            slave_sum = af.convolve(slave_mag_sq, af.constant(1.0 / (window_size ** 2), window_size, window_size))
            
            cross_af = master_af * af.conjg(slave_af)
            cross_sum = af.convolve(af.abs(cross_af), af.constant(1.0 / (window_size ** 2), window_size, window_size))
            
            coherence_af = cross_sum / af.sqrt(master_sum * slave_sum + 1e-10)
            coherence = coherence_af.to_array()
            
            coherence = np.clip(coherence, 0, 1)
            
            return interferogram, coherence
        except Exception as e:
            print(f"\n[GPU] interferogram computation failed: {e}, falling back to CPU")
            return self._compute_interferogram_cpu(master, slave)
    
    def stage_unwrap(self, crossmul_result: StageResult) -> StageResult:
        print("[unwrap] 相位解缠...")
        
        interferogram = crossmul_result.outputs["filtered"]
        coherence = crossmul_result.outputs["coherence"]
        
        wrapped_phase = np.angle(interferogram)
        
        unwrapper = SnaphuUnwrapper(
            cost_mode=CostMode.DEFO,
            init_method=InitMethod.MCF
        )
        
        try:
            result = unwrapper.unwrap(wrapped_phase, coherence)
            unwrapped_phase = result.unwrapped_phase
        except Exception as e:
            print(f"[unwrap] Snaphu解缠失败，使用简单解缠: {e}")
            if self.use_gpu:
                unwrapped_phase = self._simple_unwrap_gpu(wrapped_phase)
            else:
                unwrapped_phase = self._simple_unwrap_cpu(wrapped_phase)
        
        stage_dir = self._stage_dir("unwrap")
        stage_dir.mkdir(parents=True, exist_ok=True)
        np.save(stage_dir / "unwrapped_phase.npy", unwrapped_phase)
        
        record = {
            "shape": unwrapped_phase.shape,
            "min_phase": float(np.min(unwrapped_phase)),
            "max_phase": float(np.max(unwrapped_phase)),
        }
        self._write_stage_record("unwrap", record)
        self._mark_stage_success("unwrap")
        
        print("[unwrap] 完成")
        return StageResult(
            success=True,
            outputs={
                "unwrapped_phase": unwrapped_phase,
            },
            metadata=record
        )
    
    def _simple_unwrap_cpu(self, wrapped_phase):
        unwrapped = np.zeros_like(wrapped_phase, dtype=np.float64)
        
        for i in range(1, wrapped_phase.shape[0]):
            diff = wrapped_phase[i, :] - wrapped_phase[i-1, :]
            diff = np.mod(diff + np.pi, 2 * np.pi) - np.pi
            unwrapped[i, :] = unwrapped[i-1, :] + diff
        
        for j in range(1, wrapped_phase.shape[1]):
            diff = wrapped_phase[:, j] - wrapped_phase[:, j-1]
            diff = np.mod(diff + np.pi, 2 * np.pi) - np.pi
            unwrapped[:, j] = unwrapped[:, j-1] + diff
        
        return unwrapped
    
    def _simple_unwrap_gpu(self, wrapped_phase):
        wrapped_af = af.Array(wrapped_phase.astype(np.float64))
        
        unwrapped_af = af.copy(wrapped_af)
        
        for i in range(1, wrapped_phase.shape[0]):
            diff = wrapped_af[i, :] - wrapped_af[i-1, :]
            diff = af.mod(diff + np.pi, 2 * np.pi) - np.pi
            unwrapped_af[i, :] = unwrapped_af[i-1, :] + diff
        
        for j in range(1, wrapped_phase.shape[1]):
            diff = wrapped_af[:, j] - wrapped_af[:, j-1]
            diff = af.mod(diff + np.pi, 2 * np.pi) - np.pi
            unwrapped_af[:, j] = unwrapped_af[:, j-1] + diff
        
        return unwrapped_af.to_array()
    
    def stage_geocode(self, topo_result: StageResult, unwrap_result: StageResult) -> StageResult:
        print("[geocode] LOS转换和地理编码...")
        
        unwrapped_phase = unwrap_result.outputs["unwrapped_phase"]
        incidence_angle = topo_result.outputs["incidence_angle"]
        lat = topo_result.outputs["lat"]
        lon = topo_result.outputs["lon"]
        
        los_result = los_conversion(
            unwrapped_phase,
            self.context.wavelength,
            incidence_angle,
            None
        )
        
        stage_dir = self._stage_dir("geocode")
        stage_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(stage_dir / "los.h5", 'w') as h5:
            h5.create_dataset("los_deformation_m", data=los_result["los_deformation_m"])
            h5.create_dataset("los_deformation_mm", data=los_result["los_deformation_mm"])
            h5.create_dataset("incidence_angle", data=los_result["incidence_angle"])
            h5.create_dataset("lat", data=lat)
            h5.create_dataset("lon", data=lon)
        
        record = {
            "mean_deformation_mm": float(np.mean(los_result["los_deformation_mm"])),
        }
        self._write_stage_record("geocode", record)
        self._mark_stage_success("geocode")
        
        print("[geocode] 完成")
        return StageResult(
            success=True,
            outputs=los_result,
            metadata=record
        )
    
    def stage_product(self, geocode_result: StageResult, crossmul_result: StageResult) -> StageResult:
        print("[product] 生成最终产品...")
        
        stage_dir = self._stage_dir("product")
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        product_path = stage_dir / f"{self.context.pair_name}.h5"
        with h5py.File(product_path, 'w') as h5:
            h5.create_dataset("los_deformation_m", data=geocode_result.outputs["los_deformation_m"])
            h5.create_dataset("los_deformation_mm", data=geocode_result.outputs["los_deformation_mm"])
            h5.create_dataset("incidence_angle", data=geocode_result.outputs["incidence_angle"])
            h5.create_dataset("coherence", data=crossmul_result.outputs["coherence"])
            
            h5.attrs["pair_name"] = self.context.pair_name
            h5.attrs["wavelength"] = self.context.wavelength
            h5.attrs["master_scene"] = str(self.context.master_scene_path)
            h5.attrs["slave_scene"] = str(self.context.slave_scene_path)
            h5.attrs["use_gpu"] = self.use_gpu
        
        record = {
            "product_path": str(product_path),
            "wavelength": self.context.wavelength,
            "use_gpu": self.use_gpu,
        }
        self._write_stage_record("product", record)
        self._mark_stage_success("product")
        
        print(f"[product] 完成，产品路径: {product_path}")
        return StageResult(
            success=True,
            outputs={
                "product_path": product_path,
            },
            metadata=record
        )
    
    def run(self, dem_path: str, stages: Optional[Tuple[str, ...]] = None) -> Dict[str, StageResult]:
        results = {}
        
        if stages is None:
            stages = self.STAGE_SEQUENCE
        
        if "prep" in stages:
            results["prep"] = self.stage_prep()
        
        if "topo" in stages and "prep" in results and results["prep"].success:
            results["topo"] = self.stage_topo(dem_path, results["prep"])
        
        if "geo2rdr" in stages and "prep" in results and results["prep"].success and \
           "topo" in results and results["topo"].success:
            results["geo2rdr"] = self.stage_geo2rdr(results["prep"], results["topo"], dem_path)
        
        if "coarse_resample" in stages and "prep" in results and results["prep"].success and \
           "geo2rdr" in results and results["geo2rdr"].success:
            results["coarse_resample"] = self.stage_coarse_resample(results["prep"], results["geo2rdr"])
        
        if "refine_offset" in stages and "prep" in results and results["prep"].success and \
           "coarse_resample" in results and results["coarse_resample"].success:
            results["refine_offset"] = self.stage_refine_offset(results["prep"], results["coarse_resample"])
        
        if "refined_resample" in stages and "prep" in results and results["prep"].success and \
           "coarse_resample" in results and results["coarse_resample"].success and \
           "refine_offset" in results and results["refine_offset"].success:
            results["refined_resample"] = self.stage_refined_resample(
                results["prep"], results["coarse_resample"], results["refine_offset"]
            )
        
        if "crossmul" in stages and "prep" in results and results["prep"].success and \
           "refined_resample" in results and results["refined_resample"].success:
            topo_result = results.get("topo")
            results["crossmul"] = self.stage_crossmul(results["prep"], results["refined_resample"], topo_result)
        
        if "unwrap" in stages and "crossmul" in results and results["crossmul"].success:
            results["unwrap"] = self.stage_unwrap(results["crossmul"])
        
        if "geocode" in stages and "topo" in results and results["topo"].success and \
           "unwrap" in results and results["unwrap"].success:
            results["geocode"] = self.stage_geocode(results["topo"], results["unwrap"])
        
        if "product" in stages and "geocode" in results and results["geocode"].success and \
           "crossmul" in results and results["crossmul"].success:
            results["product"] = self.stage_product(results["geocode"], results["crossmul"])
        
        return results


def process_strip_insar(
    master_scene_path: str,
    slave_scene_path: str,
    dem_path: str,
    output_dir: str,
    wavelength: float = 0.056,
    stages: Optional[Tuple[str, ...]] = None,
    use_gpu: bool = True
) -> Dict[str, StageResult]:
    master_path = Path(master_scene_path)
    slave_path = Path(slave_scene_path)
    output_root = Path(output_dir)
    
    master_name = master_path.stem
    slave_name = slave_path.stem
    pair_name = f"{master_name}_{slave_name}"
    
    context = PairContext(
        master_scene_path=master_path,
        slave_scene_path=slave_path,
        output_root=output_root,
        pair_name=pair_name,
        wavelength=wavelength
    )
    
    processor = StripInSARProcessor(context, use_gpu=use_gpu)
    return processor.run(dem_path, stages)


def _detect_file_type(path: str) -> str:
    """检测文件类型"""
    path_lower = path.lower()
    if path_lower.endswith('.h5') or path_lower.endswith('.hdf5'):
        return 'h5'
    elif path_lower.endswith('.tar.gz') or path_lower.endswith('.tgz'):
        return 'tar.gz'
    elif path_lower.endswith('.zip'):
        return 'zip'
    elif Path(path).is_dir():
        return 'directory'
    else:
        return 'unknown'


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        prog='strip_insar',
        description='SAR Strip-mode Interferometric Processing Tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '-m', '--master',
        required=True,
        help='Path to master scene (HDF5, tar.gz, zip, or directory)'
    )
    
    parser.add_argument(
        '-s', '--slave',
        required=True,
        help='Path to slave scene (HDF5, tar.gz, zip, or directory)'
    )
    
    parser.add_argument(
        '-d', '--dem',
        required=True,
        help='Path to DEM HDF5 file'
    )
    
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output directory for processing results'
    )
    
    parser.add_argument(
        '-w', '--wavelength',
        type=float,
        default=0.056,
        help='Radar wavelength in meters (default: 0.056 for C-band)'
    )
    
    parser.add_argument(
        '--sensor',
        choices=['tianyi', 'sentinel1', 'lutan'],
        default='tianyi',
        help='Satellite sensor type: tianyi, sentinel1, or lutan'
    )
    
    parser.add_argument(
        '--stages',
        nargs='+',
        choices=StripInSARProcessor.STAGE_SEQUENCE,
        default=None,
        help='Specific stages to run. If not specified, runs all stages.'
    )
    
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        default=True,
        help='Enable GPU acceleration using ArrayFire'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU acceleration, use CPU only'
    )
    
    parser.add_argument(
        '--list-stages',
        action='store_true',
        help='List all available processing stages and exit'
    )
    
    parser.add_argument(
        '--list-sensors',
        action='store_true',
        help='List all supported satellite sensors and exit'
    )
    
    args = parser.parse_args()
    
    if args.list_stages:
        print("Available processing stages:")
        for stage in StripInSARProcessor.STAGE_SEQUENCE:
            print(f"  {stage}")
        return
    
    if args.list_sensors:
        print("Supported satellite sensors:")
        print("  tianyi    - Tianyi SAR satellite")
        print("  sentinel1 - Sentinel-1 SAR satellite")
        print("  lutan     - Lutan SAR satellite")
        return
    
    use_gpu = args.use_gpu and not args.no_gpu
    sensor = args.sensor
    
    master_type = _detect_file_type(args.master)
    slave_type = _detect_file_type(args.slave)
    
    print(f"{'='*60}")
    print("Strip InSAR Processor")
    print(f"{'='*60}")
    print(f"Master scene:   {args.master}")
    print(f"  Type:         {master_type}")
    print(f"Slave scene:    {args.slave}")
    print(f"  Type:         {slave_type}")
    print(f"Sensor:         {sensor}")
    print(f"DEM:            {args.dem}")
    print(f"Output dir:     {args.output}")
    print(f"Wavelength:     {args.wavelength} m")
    print(f"GPU enabled:    {use_gpu}")
    if args.stages:
        print(f"Stages to run:  {', '.join(args.stages)}")
    else:
        print(f"Stages to run:  all")
    print(f"{'='*60}")
    
    temp_dir = None
    
    try:
        master_h5 = args.master
        slave_h5 = args.slave
        
        if master_type != 'h5':
            print(f"\n[Import] 正在导入 master 数据: {args.master}")
            temp_dir = tempfile.mkdtemp(prefix="insar_")
            master_h5 = str(import_source_to_h5(args.master, sensor, temp_dir, "master"))
            print(f"[Import] Master 导入完成: {master_h5}")
        
        if slave_type != 'h5':
            print(f"\n[Import] 正在导入 slave 数据: {args.slave}")
            if temp_dir is None:
                temp_dir = tempfile.mkdtemp(prefix="insar_")
            slave_h5 = str(import_source_to_h5(args.slave, sensor, temp_dir, "slave"))
            print(f"[Import] Slave 导入完成: {slave_h5}")
        
        results = process_strip_insar(
            master_scene_path=master_h5,
            slave_scene_path=slave_h5,
            dem_path=args.dem,
            output_dir=args.output,
            wavelength=args.wavelength,
            stages=tuple(args.stages) if args.stages else None,
            use_gpu=use_gpu
        )
        
        print(f"\n{'='*60}")
        print("Processing Summary")
        print(f"{'='*60}")
        
        success_count = 0
        fail_count = 0
        
        for stage, result in results.items():
            status = "✓ SUCCESS" if result.success else "✗ FAILED"
            if result.success:
                success_count += 1
            else:
                fail_count += 1
            print(f"  {stage:15s} : {status}")
        
        print(f"{'='*60}")
        print(f"Total: {success_count} succeeded, {fail_count} failed")
        
        if fail_count > 0:
            exit(1)
    
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    finally:
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
