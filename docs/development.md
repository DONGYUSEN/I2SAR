# I2SAR 技术开发文档

## 开发进展总结

### 1. Strip InSAR 干涉处理

#### 1.1 处理流程

| 阶段 | 名称 | 功能 | 加速方案 |
|------|------|------|----------|
| **prep** | 数据准备 | 加载主从影像数据 | - |
| **topo** | 地形几何 | 计算经纬度、高程、入射角 | ArrayFire GPU |
| **geo2rdr** | 粗配准 | 地理坐标→雷达坐标转换 | ArrayFire GPU |
| **coarse_resample** | 粗重采样 | 使用粗偏移量重采样 | NumPy |
| **refine_offset** | 精化偏移 | FFT互相关精化偏移估计 | ArrayFire GPU |
| **refined_resample** | 精确重采样 | 双线性插值重采样 | ArrayFire GPU |
| **crossmul** | 干涉图生成 | 生成干涉图、相干性、Goldstein滤波 | ArrayFire GPU |
| **unwrap** | 相位解缠 | Snaphu相位解缠 | NumPy |
| **geocode** | LOS转换 | 视线方向形变转换 | NumPy |
| **product** | 产品生成 | 生成最终HDF5产品 | - |

#### 1.2 ArrayFire GPU 加速

**显存管理策略**：
```python
def _get_max_rows_by_memory(mem_total_mb: int) -> int:
    if mem_total_mb < 12000:    # < 12GB
        return 128
    elif mem_total_mb < 18000:   # 12-18GB
        return 256
    elif mem_total_mb < 24000:   # 18-24GB
        return 384
    elif mem_total_mb < 32000:   # 24-32GB
        return 512
    else:                      # >= 32GB
        return 1024
```

**加速模块**：
- `i2sar/geometry/accelerated_geometry.py` - rdr2geo/geo2rdr GPU加速
- `i2sar/accel/arrayfire_filtering.py` - 滤波GPU加速

#### 1.3 CLI 接口

```bash
python -m i2sar.interferometry.strip_insar \
    -m master.zip \
    -s slave.zip \
    -d dem.h5 \
    -o output \
    --sensor tianyi
```

参数：
- `-m/--master`: 主影像路径
- `-s/--slave`: 从影像路径
- `-d/--dem`: DEM文件
- `-o/--output`: 输出目录
- `--sensor`: 卫星传感器类型
- `--stages`: 指定处理阶段
- `--use-gpu/--no-gpu`: GPU开关

### 2. RTC 辐射定标处理

#### 2.1 处理流程

```python
from i2sar.rtc.strip_rtc import run_strip_rtc

result = run_strip_rtc(
    source="/path/to/data.tar.gz",
    output_dir="/path/to/output",
    sensor="lutan",
    use_gpu=True,        # 启用GPU加速
    full_resolution=True,
)
```

#### 2.2 GPU 加速

- 使用 ArrayFire 加速 rdr2geo 几何计算
- 与 InSAR 共用 `accelerated_geometry.py` 模块

### 3. DEM 下载

#### 3.1 CLI 接口

```bash
python -m i2sar download-dem \
    --bounds 28.0 30.0 94.0 96.5 \
    --output ~/Temp/dem/dem.h5
```

参数：
- `--bounds S N W E`: 边界框（南 北 西 东）
- `--output`: 输出DEM文件

### 4. 核心算法改进

#### 4.1 双阶段配准

1. **粗配准 (geo2rdr)**: 使用轨道参数和DEM计算几何偏移
2. **精化配准 (FFT互相关)**: 分块FFT精化偏移量

#### 4.2 GPU 分块处理

```python
total_chunks = (num_rows + max_rows_per_chunk - 1) // max_rows_per_chunk
for chunk_idx, row_start in enumerate(range(0, num_rows, max_rows_per_chunk)):
    print(f"\r[GPU] rdr2geo: {chunk_idx + 1}/{total_chunks}", end="")
    chunk_result = process_chunk(...)
    results.append(chunk_result)
print()
```

#### 4.3 内存管理

- 预分配GPU数组，避免频繁分配/释放
- 根据GPU显存自动调整分块大小
- 自动降级：GPU失败→CPU

### 5. 文件结构

```
i2sar/
├── interferometry/
│   ├── strip_insar.py      # 主处理流程
│   ├── phases.py           # 干涉相位处理
│   ├── filtering.py       # 滤波算法
│   └── interferogram.py  # 干涉图生成
├── geometry/
│   ├── accelerated_geometry.py  # GPU加速几何
│   ├── rdr2geo.py              # 雷达→地理
│   ├── geo2rdr.py              # 地理→雷达
│   └── geometry_arrayfire.py     # ArrayFire核心
├── rtc/
│   └── strip_rtc.py       # RTC处理
├── accel/
│   ├── arrayfire_filtering.py  # GPU滤波
│   └── arrayfire_backend.py   # 后端管理
├── cli/
│   └── main.py            # CLI入口
└── dem/
    ├── tiles.py           # SRTM瓦片
    ├── hdf.py           # HDF读写
    └── sampling.py       # DEM采样
```

### 6. 测试数据

- **Tianyi SAR数据**:
  - 主影像: `BC3-SM-SLC-1SVV-20231110T043948-002131-000033-000853.zip`
  - 从影像: `BC3-SM-SLC-1SVV-20231121T043936-002414-000034-00096E.zip`
  - DEM: `dem_S1B_S1_VV_2023-11-10T04_39_48.881889.h5`

- **数据规模**:
  - 行数: 14580
  - 列数: 12544
  - 总点数: 182,891,520 (1.82亿)

### 7. 待完成工作

- [ ] InSAR全流程测试完成
- [ ] GPU性能优化（显存管理）
- [ ] 相位解缠算法优化
- [ ] 产品格式验证

### 8. 依赖环境

- Python 3.10
- NumPy, SciPy
- h5py, GDAL
- ArrayFire (GPU加速)
- Numba (CPU加速)

---

*更新日期: 2026-04-29*