from __future__ import annotations

import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

try:
    from osgeo import gdal
    gdal.UseExceptions()
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False
    import tifffile

from i2sar.io.base import SourceRef


def _compound_complex(real: np.ndarray, imag: np.ndarray) -> np.ndarray:
    if real.shape != imag.shape:
        raise ValueError(f"real and imag shapes differ: {real.shape} != {imag.shape}")
    if real.dtype != imag.dtype:
        raise ValueError(f"real and imag dtypes differ: {real.dtype} != {imag.dtype}")
    out = np.empty(real.shape, dtype=np.dtype([("real", real.dtype), ("imag", imag.dtype)]))
    out["real"] = real
    out["imag"] = imag
    return out


def _read_tiff_gdal(filepath: str) -> np.ndarray:
    ds = gdal.Open(filepath)
    if ds is None:
        raise RuntimeError(f"GDAL failed to open {filepath}")
    
    num_bands = ds.RasterCount
    if num_bands == 1:
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray()
    else:
        band = ds.GetRasterBand(1)
        dtype = band.DataType
        if dtype == gdal.GDT_Int16:
            np_dtype = np.int16
        elif dtype == gdal.GDT_UInt16:
            np_dtype = np.uint16
        elif dtype == gdal.GDT_Float32:
            np_dtype = np.float32
        elif dtype == gdal.GDT_Float64:
            np_dtype = np.float64
        else:
            np_dtype = np.float32
        
        data = np.zeros((ds.RasterYSize, ds.RasterXSize, num_bands), dtype=np_dtype)
        for i in range(num_bands):
            band = ds.GetRasterBand(i + 1)
            data[..., i] = band.ReadAsArray()
    
    ds = None
    return data


def _read_tiff(ref: SourceRef) -> np.ndarray:
    if ref.storage == "zip":
        if ref.member is None:
            raise ValueError("zip SLC SourceRef requires member")
        with zipfile.ZipFile(ref.path) as zf:
            with zf.open(ref.member) as fh:
                with tempfile.NamedTemporaryFile(suffix=Path(ref.member).suffix, delete=False) as tmp:
                    tmp.write(fh.read())
                    tmp_path = tmp.name
            try:
                if GDAL_AVAILABLE:
                    return _read_tiff_gdal(tmp_path)
                else:
                    return tifffile.imread(tmp_path)
            finally:
                import os
                os.unlink(tmp_path)
    if ref.storage == "tar":
        if ref.member is None:
            raise ValueError("tar SLC SourceRef requires member")
        with tarfile.open(ref.path) as tf:
            fh = tf.extractfile(ref.member)
            if fh is None:
                raise FileNotFoundError(ref.member)
            with tempfile.NamedTemporaryFile(suffix=Path(ref.member).suffix, delete=False) as tmp:
                tmp.write(fh.read())
                tmp_path = tmp.name
            try:
                if GDAL_AVAILABLE:
                    return _read_tiff_gdal(tmp_path)
                else:
                    return tifffile.imread(tmp_path)
            finally:
                import os
                os.unlink(tmp_path)
    
    filepath = str(Path(ref.path))
    if GDAL_AVAILABLE:
        return _read_tiff_gdal(filepath)
    else:
        return tifffile.imread(filepath)


def load_slc_as_compound_complex(ref: SourceRef, attrs: dict[str, Any]) -> np.ndarray:
    image = _read_tiff(ref)
    layout = attrs.get("storage_layout")
    if layout == "two_band_iq":
        if image.ndim != 3 or image.shape[-1] != 2:
            raise ValueError(f"two_band_iq SLC must have shape (rows, columns, 2), got {image.shape}")
        return _compound_complex(image[..., 0], image[..., 1])
    if layout == "single_band_complex":
        if np.iscomplexobj(image):
            return _compound_complex(image.real.astype(image.real.dtype, copy=False), image.imag.astype(image.imag.dtype, copy=False))
        if image.ndim == 3 and image.shape[-1] == 2:
            return _compound_complex(image[..., 0], image[..., 1])
        raise ValueError(f"single_band_complex SLC requires complex dtype or trailing IQ dimension, got shape={image.shape} dtype={image.dtype}")
    raise ValueError(f"unsupported SLC storage_layout: {layout}")
