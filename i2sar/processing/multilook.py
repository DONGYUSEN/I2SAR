import json
from pathlib import Path
from typing import Dict, Tuple, Any
import numpy as np
from osgeo import gdal


def multilook(data: np.ndarray, nalks: int, nrlks: int, boundary: str = 'crop') -> np.ndarray:
    """
    多视处理（通用版本）

    Args:
        data: 输入复数数组 (rows, cols), dtype=complex64
        nalks: 方位向视数
        nrlks: 距离向视数
        boundary: 'crop' 裁剪不能整除部分, 'pad' 填充到可整除

    Returns:
        多视后复数数组, dtype=complex64
    """
    rows, cols = data.shape
    out_rows = rows // nalks
    out_cols = cols // nrlks

    if boundary == 'crop':
        data = data[:out_rows * nalks, :out_cols * nrlks]
    elif boundary == 'pad':
        pad_rows = (nalks - rows % nalks) % nalks
        pad_cols = (nrlks - cols % nrlks) % nrlks
        if pad_rows > 0 or pad_cols > 0:
            data = np.pad(data, ((0, pad_rows), (0, pad_cols)), mode='edge')
        out_rows = data.shape[0] // nalks
        out_cols = data.shape[1] // nrlks

    amplitude = np.abs(data)
    amp = amplitude.reshape(out_rows, nalks, out_cols, nrlks).sum(axis=(1, 3)) / (nalks * nrlks)

    phase_complex = np.exp(1j * np.angle(data))
    phase_complex = phase_complex.reshape(out_rows, nalks, out_cols, nrlks).sum(axis=(1, 3))
    phase_ml = np.angle(phase_complex)

    result = amp * np.exp(1j * phase_ml)
    return result.astype(np.complex64)


def multilook_phase(phase_data: np.ndarray, nalks: int, nrlks: int, boundary: str = 'crop') -> np.ndarray:
    """
    对相位数据进行多视处理（矢量平均法）

    Args:
        phase_data: 相位数据数组，形状为 (长度, 宽度)
        nalks: 方位向视数
        nrlks: 距离向视数
        boundary: 'crop' 裁剪, 'pad' 填充

    Returns:
        多视处理后的相位数据数组
    """
    rows, cols = phase_data.shape
    out_rows = rows // nalks
    out_cols = cols // nrlks

    if boundary == 'crop':
        phase_data = phase_data[:out_rows * nalks, :out_cols * nrlks]
    elif boundary == 'pad':
        pad_rows = (nalks - rows % nalks) % nalks
        pad_cols = (nrlks - cols % nrlks) % nrlks
        if pad_rows > 0 or pad_cols > 0:
            phase_data = np.pad(phase_data, ((0, pad_rows), (0, pad_cols)), mode='edge')
        out_rows = phase_data.shape[0] // nalks
        out_cols = phase_data.shape[1] // nrlks

    complex_repr = np.exp(1j * phase_data)
    azimuth_sum = complex_repr.reshape(out_rows, nalks, out_cols * nrlks).sum(axis=1)
    total_sum = azimuth_sum.reshape(out_rows, out_cols, nrlks).sum(axis=2)
    mean_phase = np.angle(total_sum)

    return mean_phase


def multilook_preserve_phase(data: np.ndarray, nalks: int, nrlks: int, boundary: str = 'crop') -> np.ndarray:
    """
    相位保持多视处理

    对于SLC数据和干涉数据，分别处理振幅和相位，以更好地保留相位信息。

    Args:
        data: 复数数据数组，形状为 (长度, 宽度)
        nalks: 方位向视数
        nrlks: 距离向视数
        boundary: 'crop' 裁剪, 'pad' 填充

    Returns:
        多视处理后的复数数据数组
    """
    amplitude = np.abs(data)
    phase = np.angle(data)

    amp_ml = multilook(amplitude.astype(np.float32), nalks, nrlks, boundary=boundary)
    phase_ml = multilook_phase(phase, nalks, nrlks, boundary=boundary)

    result = amp_ml * np.exp(1j * phase_ml)
    return result.astype(np.complex64)


def multilook_with_stats(
    data: np.ndarray,
    nalks: int,
    nrlks: int,
    preserve_phase: bool = False,
    boundary: str = 'crop'
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    多视处理并返回统计信息

    Args:
        data: 输入数据
        nalks: 方位向视数
        nrlks: 距离向视数
        preserve_phase: 是否使用相位保持多视处理
        boundary: 边界处理方式

    Returns:
        (result, stats): 多视结果和统计信息字典
    """
    amplitude_orig = np.abs(data)
    mean_amp_orig = np.mean(amplitude_orig)
    std_amp_orig = np.std(amplitude_orig)
    snr_orig = (mean_amp_orig / std_amp_orig) ** 2 if std_amp_orig > 0 else float('inf')

    original_stats = {
        'mean_amplitude': float(mean_amp_orig),
        'std_amplitude': float(std_amp_orig),
        'enl_estimate': float(snr_orig),
        'shape': data.shape
    }

    if preserve_phase:
        result = multilook_preserve_phase(data, nalks, nrlks, boundary=boundary)
    else:
        result = multilook(data, nalks, nrlks, boundary=boundary)

    amplitude_ml = np.abs(result)
    mean_amp_ml = np.mean(amplitude_ml)
    std_amp_ml = np.std(amplitude_ml)
    snr_ml = (mean_amp_ml / std_amp_ml) ** 2 if std_amp_ml > 0 else float('inf')

    multilook_stats = {
        'mean_amplitude': float(mean_amp_ml),
        'std_amplitude': float(std_amp_ml),
        'enl_estimate': float(snr_ml),
        'shape': result.shape
    }

    theoretical_enl_gain = nalks * nrlks
    actual_enl_gain = snr_ml / snr_orig if snr_orig > 0 and snr_orig != float('inf') else 1.0

    stats = {
        'original': original_stats,
        'multilook': multilook_stats,
        'improvement': {
            'enl_gain': float(actual_enl_gain),
            'enl_gain_db': float(10 * np.log10(actual_enl_gain)) if actual_enl_gain > 0 else 0.0,
            'theoretical_enl_gain': int(theoretical_enl_gain),
            'theoretical_enl_gain_db': float(10 * np.log10(theoretical_enl_gain)),
            'noise_reduction_percent': float((std_amp_orig - std_amp_ml) / std_amp_orig * 100) if std_amp_orig > 0 else 0.0,
            'efficiency': float(actual_enl_gain / theoretical_enl_gain * 100)
        },
        'parameters': {
            'nalks': nalks,
            'nrlks': nrlks,
            'total_looks': nalks * nrlks,
            'preserve_phase': preserve_phase
        }
    }

    return result, stats


def multilook_chunked(
    input_path: str | Path,
    output_path: str | Path,
    nalks: int,
    nrlks: int,
    chunk_lines: int = 1000,
    preserve_phase: bool = True
) -> bool:
    """
    分块多视处理（文件到文件）

    Args:
        input_path: 输入TIFF文件路径
        output_path: 输出TIFF文件路径
        nalks: 方位向视数
        nrlks: 距离向视数
        chunk_lines: 分块行数（输出行数）
        preserve_phase: 是否使用相位保持多视处理

    Returns:
        处理是否成功
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    in_ds = gdal.Open(str(input_path))
    if in_ds is None:
        raise ValueError(f"无法打开文件: {input_path}")

    in_rows = in_ds.RasterYSize
    in_cols = in_ds.RasterXSize
    geotransform = in_ds.GetGeoTransform()
    projection = in_ds.GetProjection()

    out_rows = in_rows // nalks
    out_cols = in_cols // nrlks

    new_geotransform = list(geotransform)
    new_geotransform[1] *= nrlks
    new_geotransform[5] *= nalks

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(str(output_path), out_cols, out_rows, 1, gdal.GDT_CFloat32,
                          options=['COMPRESS=LZW', 'TILED=YES'])
    out_ds.SetGeoTransform(tuple(new_geotransform))
    out_ds.SetProjection(projection)
    out_band = out_ds.GetRasterBand(1)

    for out_row in range(0, out_rows, chunk_lines):
        chunk_size = min(chunk_lines, out_rows - out_row)
        read_rows = chunk_size * nalks

        data = in_ds.GetRasterBand(1).ReadAsArray(0, out_row * nalks, in_cols, read_rows)
        if data is None:
            raise ValueError(f"读取分块失败 at row {out_row * nalks}")

        if preserve_phase:
            ml_data = multilook_preserve_phase(data.astype(np.complex64), nalks, nrlks)
        else:
            ml_data = multilook(data.astype(np.complex64), nalks, nrlks)
        out_band.WriteArray(ml_data, 0, out_row)

    in_ds = None
    out_ds = None
    return True


def update_hdf5_radar_grid(
    h5_path: str | Path,
    nalks: int,
    nrlks: int,
    update_prf: bool = True
) -> bool:
    """
    更新HDF5文件中radar_grid的json参数以反映多视处理后的变化

    Args:
        h5_path: HDF5文件路径
        nalks: 方位向视数
        nrlks: 距离向视数
        update_prf: 是否更新prf为有效PRF (prf/nalks)

    Returns:
        更新是否成功
    """
    import h5py

    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"文件不存在: {h5_path}")

    with h5py.File(h5_path, 'r+') as h5:
        if 'radar_grid' not in h5:
            raise ValueError("HDF5文件中没有radar_grid组")

        radar_grid = h5['radar_grid']
        json_attr = radar_grid.attrs.get('json')
        if json_attr is None:
            raise ValueError("radar_grid没有json属性")

        params = json.loads(json_attr)

        original_rows = params.get('numberOfRows')
        original_cols = params.get('numberOfColumns')

        params['numberOfRows'] = original_rows // nalks
        params['numberOfColumns'] = original_cols // nrlks
        params['azimuthLooks'] = nalks
        params['rangeLooks'] = nrlks

        if 'azimuthResolution' in params:
            params['azimuthResolution'] = params['azimuthResolution'] * nalks
        if 'rowSpacing' in params:
            params['rowSpacing'] = params['rowSpacing'] * nalks
        if 'columnSpacing' in params:
            params['columnSpacing'] = params['columnSpacing'] * nrlks
        if 'groundRangeResolution' in params:
            params['groundRangeResolution'] = params['groundRangeResolution'] * nrlks

        if update_prf and 'prf' in params:
            params['prf'] = params['prf'] / nalks

        original_col_spacing = float(params.get('columnSpacing', 0))
        if original_col_spacing > 0:
            out_cols = original_cols // nrlks
            new_column_spacing = original_col_spacing * nrlks
            first_pixel_time = float(params.get('rangeTimeFirstPixel', 0))
            params['columnSpacing'] = new_column_spacing
            params['rangeTimeLastPixel'] = first_pixel_time + (out_cols - 1) * new_column_spacing

        radar_grid.attrs['json'] = json.dumps(params)

    return True


def multilook_hdf5(
    input_path: str | Path,
    output_path: str | Path,
    nalks: int,
    nrlks: int,
    chunk_lines: int = 1000,
    preserve_phase: bool = True
) -> bool:
    """
    对HDF5格式的SLC数据进行多视处理并更新参数

    Args:
        input_path: 输入HDF5文件路径
        output_path: 输出HDF5文件路径
        nalks: 方位向视数
        nrlks: 距离向视数
        chunk_lines: 分块行数（输出行数）
        preserve_phase: 是否使用相位保持多视处理

    Returns:
        处理是否成功
    """
    import h5py

    input_path = Path(input_path)
    output_path = Path(output_path)

    import shutil
    shutil.copy2(input_path, output_path)

    def compound_to_complex(arr):
        """将compound类型转换为complex64"""
        if arr.dtype.names and 'real' in arr.dtype.names and 'imag' in arr.dtype.names:
            return arr['real'] + 1j * arr['imag']
        return arr

    def complex_to_compound(arr, original_dtype):
        """将complex64转换回compound类型"""
        if hasattr(original_dtype, 'names') and original_dtype.names and 'real' in original_dtype.names and 'imag' in original_dtype.names:
            result = np.empty(arr.shape, dtype=original_dtype)
            result['real'] = arr.real
            result['imag'] = arr.imag
            return result
        return arr

    with h5py.File(input_path, 'r') as src_h5:
        if 'slc/data' not in src_h5:
            raise ValueError("HDF5文件中没有slc/data数据集")
        slc_ds = src_h5['slc/data']
        original_dtype = slc_ds.dtype
        src_data = slc_ds[:]
        in_rows, in_cols = src_data.shape

    src_complex = compound_to_complex(src_data)

    out_rows = in_rows // nalks
    out_cols = in_cols // nrlks

    temp_tiff = output_path.parent / f"temp_ml_{output_path.stem}.tiff"

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(str(temp_tiff), in_cols, in_rows, 1, gdal.GDT_CFloat32,
                          options=['COMPRESS=LZW', 'TILED=YES'])
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(src_complex)
    out_ds = None

    multilook_chunked(temp_tiff, temp_tiff, nalks, nrlks, chunk_lines, preserve_phase)

    ds = gdal.Open(str(temp_tiff))
    ml_data = ds.GetRasterBand(1).ReadAsArray()
    ds = None

    ml_compound = complex_to_compound(ml_data, original_dtype)

    with h5py.File(output_path, 'r+') as h5:
        if 'slc/data' in h5:
            del h5['slc/data']
        h5.create_dataset('slc/data', data=ml_compound)

    temp_tiff.unlink()

    update_hdf5_radar_grid(output_path, nalks, nrlks)

    return True