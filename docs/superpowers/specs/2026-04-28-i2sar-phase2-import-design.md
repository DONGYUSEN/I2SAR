# I2SAR 第二阶段数据导入设计

## 1. 目标

第二阶段把 D2SAR 的数据导入和数据组织能力迁移到 I2SAR。交付目标不是跑通几何、配准、InSAR 或 RTC，而是让外部 SAR 产品稳定转换成标准 `scene.h5`，并注册到 `project.h5`。

本阶段重点支持：

- LuTan Stripmap SLC 导入，包括轨道平滑。
- Tianyi / SAFE-like Stripmap SLC 导入，包括单波段复数 TIFF 格式语义。
- Sentinel-1 TOPS/SLC 的 schema 表达和 source reference 解析框架，为后续完整 TOPS burst 解析预留接口。

## 2. 设计原则

- 导入器只负责外部产品到 I2SAR `Scene` / `scene.h5`，不做几何求解、配准、干涉或 RTC。
- 传感器类型和成像模式分开表达：`sensor` 不替代 `acquisition_mode`。
- HDF5 是导入后的内部真源。外部 manifest、XML、TIFF、ZIP/TAR 成员只作为 source reference 和 provenance。
- D2SAR 中经过验证的导入经验要迁移为明确测试，而不是隐式脚本行为。

## 3. 包结构

新增：

```text
i2sar/io/
  __init__.py
  base.py              # Importer 协议、ImportResult、SourceRef
  source.py            # 目录/ZIP/TAR/VSI 路径引用和发现工具
  scene_writer.py      # 标准 scene.h5 metadata/orbit/radar_grid/doppler/slc 写入
  lutan.py             # LuTan parser/importer
  safe_like.py         # Tianyi/Sentinel SAFE-like parser/importer

i2sar/orbit/
  __init__.py
  smooth.py            # 从 D2SAR orbit_smooth 迁移的纯 numpy/scipy 可选轨道平滑
```

现有 HDF5 schema 需要扩展：

```text
scene.h5
  /metadata/source
  /metadata/acquisition
  /metadata/quality
  /radar_grid
  /orbit
    time[]
    position[]
    velocity[]
    attrs: smoothed, smoothing_json
  /orbit_raw
    time[]
    position[]
    velocity[]
  /doppler
  /slc
    attrs:
      path
      storage
      member
      format
      sample_format
      storage_layout
      complex_band_count
      processing_format
```

## 4. LuTan 导入要求

LuTan importer 需要迁移 D2SAR 经验：

- 支持目录、ZIP、TAR/TGZ 输入。
- 发现 `*.meta.xml`、SLC TIFF、incidence XML、RPC。
- 解析 acquisition、scene、orbit、radar_grid、doppler。
- 校验/修正 look direction 的接口保留为可插拔策略；第二阶段可先记录 XML 值和 verified 值相同。
- 必须执行 LuTan orbit smoothing：
  - 少于 8 个 state vector 时跳过。
  - 默认参数：`degree=5`、`sigma=4.0`、`max_iter=3`、`ignore_start=-1`、`ignore_end=-1`。
  - 写入 raw orbit 和 smoothed orbit。
  - 写入 smoothing provenance。
- SLC 格式语义：
  - `sample_format=iq_int16`
  - `storage_layout=two_band_iq`
  - `complex_band_count=1`

## 5. Tianyi / SAFE-like / Sentinel 导入要求

SAFE-like importer 需要支持：

- 目录、ZIP、TAR/TGZ 输入。
- annotation XML、calibration XML、manifest.safe、measurement TIFF 发现。
- GDAL VSI source reference：`/vsizip/...`、`/vsitar/...`。
- 解析 acquisition、scene corners、orbit、radar_grid、doppler。
- 单波段复数 TIFF 格式语义：
  - `sample_format=cint16`
  - `storage_layout=single_band_complex`
  - `complex_band_count=1`
  - `processing_format=single_band_cfloat32`

Sentinel-1 TOPS 在第二阶段的最低要求：

- 能表达 `acquisition_mode=tops`。
- `scene.h5:/tops/swaths`、`/tops/bursts`、`/tops/burst_grid`、`/tops/azimuth_steering` 存在。
- source reference 能记录 SAFE ZIP/目录中的 annotation、calibration、manifest.safe、measurement 成员。
- 完整 burst 几何解析可作为第二阶段后续子任务，不阻塞 LuTan/Tianyi 的 scene.h5 落盘。

## 6. API

统一入口：

```python
from i2sar.io import import_scene

scene_path = import_scene(
    project,
    source="/path/to/product.zip",
    sensor="auto",
    acquisition_mode="auto",
    scene_id=None,
)
```

导入结果：

```python
ImportResult(
    scene_id: str,
    scene_path: Path,
    sensor: str,
    acquisition_mode: AcquisitionMode,
    slc: SourceRef,
)
```

传感器 importer：

```python
class SceneImporter:
    def detect(self, source: Path) -> bool: ...
    def parse(self) -> ParsedScene: ...
    def import_to_project(self, project: Project) -> ImportResult: ...
```

## 7. 测试策略

用小型合成 XML/目录测试，不依赖真实大影像。

必须覆盖：

- LuTan 文件发现：目录、ZIP、TAR 至少覆盖一种归档输入。
- LuTan orbit smoothing：
  - state vector >= 8 时调用平滑。
  - raw orbit 和 smoothed orbit 都写入 HDF5。
  - smoothing provenance 写入。
- LuTan SLC format attrs：`iq_int16`、`two_band_iq`。
- Tianyi SAFE-like 文件发现。
- Tianyi SLC format attrs：`cint16`、`single_band_complex`、`single_band_cfloat32`。
- Sentinel/TOPS schema 表达：TOPS groups 存在，source refs 能记录 SAFE 成员。
- `Project` index 中 scene 条目与 `scene.h5` 一致。

## 8. 非目标

第二阶段不实现：

- `rdr2geo` / `geo2rdr`
- 配准
- 干涉
- RTC 计算
- Sentinel TOPS burst 几何完整求解
- GDAL 真实大数组读取和重写

