# I2SAR 发现记录

## 2026-04-28 初步代码摸底

- `/home/ysdong/Software/I2SAR` 当前目录基本为空，且不是 git 仓库。
- `/home/ysdong/Software/D2SAR` 含有较多脚本和测试，重点能力包括：
  - `scripts/lutan_importer.py`：LuTan-1 产品发现、元数据解析、VSI 路径支持、DEM 选择。
  - `scripts/tianyi_importer.py`：天仪/Sentinel SAFE 风格产品发现、元数据解析、VSI 路径支持、DEM 选择。
  - `scripts/insar_preprocess.py`：主辅影像采样、PRF、多普勒一致性处理策略。
  - `scripts/dem_manager.py`、`insar_registration.py`、`insar_filtering.py` 等可作为迁移来源。
- `/home/ysdong/Software/C2SAR2` 已有模块化雏形：
  - `dinsar.py` 是当前全流程总控脚本。
  - `registration/` 覆盖粗配准、精配准、偏移拟合、质量评估、重采样和 ESD。
  - `interferometry/` 覆盖干涉图、相干性、滤波、地形/平地相位处理和 snaphu 封装。
  - `utils/` 覆盖 geosar、baseline、ll2sar、sar2ll、DEM、多视、传感器导入等能力。
  - `workflow/` 和 `core/` 有通用基础设施雏形，但还没有真正成为主流程骨架。

## 初步架构判断

- I2SAR 不应简单合并 D2SAR 和 C2SAR2 脚本；需要先建立统一数据模型、包结构和工作流契约。
- 数据导入与数据组织应尽早统一为标准 `Scene`、`RadarGrid`、`Orbit`、`Doppler`、`ProductManifest` 等对象，否则后续几何、配准和时序接口会继续被 YAML/路径细节耦合。
- `rdr2geo` / `geo2rdr` 是风险最高的核心模块，应单独成为 `geometry` 子包，并保留与 ISCE3 或已有 C2SAR2/D2SAR 结果的数值对照测试。
- 用户选择首个 MVP 采用 D：先做统一数据模型和空流程，不绑定具体传感器。
- 用户明确要求用 HDF5 方式组织管理所有数据。I2SAR 应把 HDF5 作为内部主容器，而不是仅作为某个中间模块的输出格式。
- 用户选择 HDF5 粒度方案 C：项目级 HDF5 轻量索引 + scene/pair/product 独立 HDF5 数据文件。该方案适合并行、断点续跑、长期时序扩展和 Web 查询。
- 用户选择几何模块方案 A：完全纯 Python/Numpy 实现，不依赖 ISCE3；ISCE3 只用于测试对照。该约束会提高前期几何验证成本，但能减少部署依赖并增强可控性。
- 用户选择接口方案 D：Python API 为核心，CLI 和配置作为薄封装，同时为未来 Web API 预留。该决策要求工作流节点、数据访问和算法模块都必须可直接由 Python 调用。
- 用户补充需求：数据导入需要支持更多 Stripmap 模式和 TOPS 模式数据。设计上应把传感器类型与成像模式拆开，避免把 `sensor` 当成处理逻辑唯一分支。
- 用户补充需求：RTC 处理要从干涉处理中拆出来，作为独立处理链。D2SAR 的 `scripts/strip_rtc.py` 已包含可迁移经验，包括 RTC factor、topo lon/lat/height、UTM transform/rasterize、preview 和 HDF 输出等阶段。
- D2SAR 的 `strip_insar.py` / `strip_insar2.py` 已有 Stripmap InSAR 分阶段经验，但其中仍混有 ISCE3/NISAR/GPU 路径；I2SAR 迁移时要抽取处理契约，避免继承运行时依赖。

## 2026-04-28 第二阶段导入迁移发现

- LuTan 导入必须迁移轨道平滑逻辑：
  - D2SAR `lutan_importer.py` 中 `_smooth_orbit_for_import()` 调用 `orbit_smooth.orbit_smooth()`。
  - 平滑参数为 `degree=5`、`sigma=4.0`、`max_iter=3`、`ignore_start=-1`、`ignore_end=-1`。
  - 少于 8 个 state vector 时跳过平滑，并记录 `status=skipped-insufficient-state-vectors`。
  - 平滑后要同时保留 raw orbit 与 smoothed orbit，并写入 smoothing provenance，包括 algorithm、status、n_outliers、ignore_start、ignore_end、methods、used_spline。
- D2SAR `tests/test_lutan_importer_orbit_smooth.py` 明确验证：
  - importer 会调用 orbit smoothing。
  - 输出 smoothed orbit 和 raw orbit。
  - LuTan SLC manifest 使用 `sample_format=iq_int16`、`storage_layout=two_band_iq`。
- 天仪/Sentinel SAFE-like 导入需要迁移：
  - annotation、calibration、manifest.safe、measurement TIFF 发现逻辑。
  - ZIP/TAR/目录三类输入，并支持 GDAL VSI 路径 `/vsizip/`、`/vsitar/`。
  - `extract_acquisition()`、`extract_scene_info()`、`extract_orbit()`、`extract_radar_grid()`、`extract_doppler()` 的 SAFE-like XML 解析。
- D2SAR `tests/test_tianyi_importer_slc_format.py` 明确验证：
  - 天仪 measurement TIFF 是单波段复数。
  - SLC 格式为 `sample_format=cint16`、`storage_layout=single_band_complex`、`complex_band_count=1`、`processing_format=single_band_cfloat32`。
- D2SAR `common_processing.py` 还有 manifest 路径解析和 ancillary fallback：
  - 能通过 `metadata/scene.json` 或 ancillary annotation/meta XML 恢复 scene corners。
  - 第二阶段 I2SAR 不应复制所有 ISCE3 构造逻辑，但应迁移 manifest/source reference 和 VSI 解析契约。
- D2SAR `strip_insar2.py` 对 LuTan pair 强制选择 Legendre orbit interpolation；I2SAR 第二阶段只记录 orbit interpolation policy provenance，真正几何插值在第三阶段实现。

## 2026-04-28 第二阶段实施发现

- `scene.h5` 需要明确包含 `/orbit_raw`，否则 LuTan 导入无法同时保留原始轨道和平滑轨道；已将其纳入 scene skeleton 与 schema 校验。
- `import_scene(sensor="tianyi")` 在未显式传入 `acquisition_mode` 时应默认 `stripmap`；`sensor="sentinel1"` 默认 `tops` 更符合当前 SAFE-like/TOPS 迁移目标。
- 第二阶段实施计划中的 LuTan 代码片段有 markdown 转义残留，实际实现已按设计意图重写，未照抄损坏片段。
- `scene_writer` 写空轨道时需要保持 `position`/`velocity` 为 `(0, 3)`，否则后续几何模块读取会遇到一维空数组；已补回归测试。
- ZIP/TAR/目录导入应统一通过 `SourceRef` 记录 `path`、`storage`、`member`，HDF5 的 `/slc` attrs 不直接写死 VSI 字符串，以便 Web/API 层可结构化理解源数据位置。
- SLC 数据导入不能用 NumPy complex dtype 表示整数 IQ，因为 complex dtype 会把整数提升为浮点；应使用 HDF5 compound dtype `{real: 原始dtype, imag: 原始dtype}`，既统一单波段复数据契约，又保持原始数值类型。
- 压缩包内 TIFF 通过 `tifffile` 读取时需要带后缀的临时文件名；匿名 file-like 对象会触发文件名处理错误。

## 2026-04-28 第三阶段几何摸底

- C2SAR2 `utils/geosar.py` 主要围绕 DEM->SAR 映射表、SAR lat/lon 网格和 geocoding 产物组织，依赖 `scipy`、`gdal`、`numba` 等运行时；其中 `write_sar_latlon_grids_from_dem_mapping()`、`rasterize_to_sar_grid_last_win()`、`interpolate_sar_grid()` 可作为 I2SAR 后续 raster/grid 工具参考。
- C2SAR2 `utils/ll2sar.py` 和 `utils/sar2ll.py` 的核心价值是已有 geosar HDF 映射表上的双线性采样、DEM geotransform 反解、经纬度/DEM/SAR 像素转换；这部分适合先迁移为纯 numpy 的 grid sampling 工具。
- D2SAR `strip_insar.py` 的 `geo2rdr` 主要调用 ISCE3/CUDA `Geo2Rdr` 或 `isce3.geometry.geo2rdr`，不能直接作为 I2SAR 运行实现，但可作为参数、输出命名、offset 解释和验证对照来源。
- 阶段 3 第一版建议先实现几何基础件：WGS84/LLH/ECEF、轨道插值、雷达网格时间/距离转换、DEM grid helper、rdr2geo/geo2rdr API 框架和小型解析几何测试；真实大场景与 ISCE3 数值对照后续逐步补。
- 用户明确 DEM bbox 应相对 scene corners 四向外扩 `0.2` 度；该值已作为 `DEFAULT_SCENE_MARGIN_DEG`，并写入 DEM HDF5 provenance。
- 当前环境没有 `osgeo`/`rasterio`，因此 D2SAR 的 DEM 下载、GDAL mosaic/clip、geoid correction 暂不作为硬依赖；第一版先实现本地 HGT/array -> DEM HDF5 和 coverage/tile 基础能力。
- DEM 作为 project/scene 可共享产品更合适，因此新增 `ProductType.DEM`，并以 product HDF5 存放，而不是塞进单个 scene.h5。
- 几何基础件已从 WGS84 LLH/ECEF 转换开始实现；当前 API 支持标量和 numpy 向量输入，后续 orbit interpolation、radar grid 时间/距离转换、DEM sampling 和 `geo2rdr`/`rdr2geo` 都应复用该基础坐标层。
- 轨道插值第一版采用线性插值，目标是先固定可替换的调用接口和 HDF5 读取契约；真实 SAR 几何求解阶段应基于验证数据再升级为 Hermite、Lagrange 或传感器指定插值策略。
