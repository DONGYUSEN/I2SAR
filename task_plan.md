# I2SAR 重写规划

## 目标

重写一套 SAR/InSAR 数据处理软件 I2SAR，覆盖数据导入、数据组织、rdr2geo、geo2rdr、配准、干涉、SNAP/snaphu 解缠、LOS 转换、产品输出，并为后续时序 InSAR 和 Web 处理预留稳定接口。

## 当前阶段

阶段 5b：几何与配准模块调试（进行中），重点修复 geo2rdr NaN 问题和 ArrayFire 导入错误。

## 阶段计划

| 阶段 | 状态 | 内容 |
| --- | --- | --- |
| 0 | complete | 梳理 D2SAR/C2SAR2 现状、明确 I2SAR 架构边界 |
| 1 | complete | 定义包结构、核心数据模型、项目目录和配置规范 |
| 2 | complete | 迁移 D2SAR 数据导入与数据组织能力，支持 Stripmap 与 TOPS 数据模式 |
| 3 | complete | 实现纯 Python 几何模块：DEM HDF5、rdr2geo、geo2rdr、轨道/多普勒/DEM sampling 接口 |
| 4 | complete | 迁移并重构配准模块（参考 ISCE2 流程 + C2SAR2 精确配准算法） |
| 5 | complete | 实现干涉、滤波、解缠、LOS 转换模块 |
| 5b | testing | RTC 处理链（辐射校正、地形校正、地理编码）；当前发现 geo2rdr 模块返回 NaN 的核心 bug |
| 6 | pending | 建立全流程工作流、缓存、断点续跑和产品 manifest |
| 7 | pending | 为时序 InSAR 与 Web API 预留接口 |
| 8 | pending | 建立测试、基准数据、验证报告和文档 |

## 关键架构决策

- I2SAR 第一阶段采用“统一数据模型和空流程优先”，不绑定具体传感器；传感器导入器在第二阶段逐步接入。
- 几何模块要求完全摆脱 ISCE3 运行时依赖，`rdr2geo` / `geo2rdr` 采用纯 Python/Numpy 实现；ISCE3 只可作为测试对照来源，不作为运行后端。
- 产品格式主干采用 HDF5：用 HDF5 组织管理原始导入、中间结果、几何、配准、干涉、解缠、LOS 和产品元数据；必要时再导出 GeoTIFF/COG 等交换格式。
- 接口形态采用 Python API 为核心，CLI、配置驱动工作流和未来 Web API 都作为薄封装调用同一套核心 API。

## 已定架构约束

- 第一阶段先建立统一数据模型和空流程，不绑定具体传感器。
- HDF5 是 I2SAR 的主数据容器和内部产品格式。
- HDF5 文件粒度采用混合模式：项目级 `project.h5` 作为轻量索引，每个 scene、pair、product 使用独立 HDF5 文件承载数据。
- 几何核心不依赖 ISCE3 运行时；纯 Python/Numpy 是目标实现路径。
- Python API 是第一主接口；CLI、YAML/JSON 配置和 Web API 不应复制业务逻辑。
- 数据模型必须显式区分成像模式 `acquisition_mode`，至少支持 `stripmap` 和 `tops`。
- 处理链必须显式区分 workflow 类型，至少包括 `insar` 和 `rtc`；RTC 不是 InSAR 的附属导出步骤，而是独立的一景/多景处理链。

## 阶段 2 当前结果

- 已完成导入基础骨架：`SourceRef`、目录/ZIP/TAR 成员发现、GDAL VSI 路径表达、统一 `scene_writer`、`import_scene` 分派入口。
- 已迁移 LuTan 关键导入契约：轨道平滑入口、少于 8 个 state vector 跳过、raw/smoothed orbit 同时写入 HDF5、smoothing provenance、`iq_int16`/`two_band_iq` SLC 语义。
- 已迁移 SAFE-like 关键导入契约：annotation/calibration/manifest/measurement 发现、天仪/Sentinel 单波段复数 TIFF 语义、Sentinel TOPS HDF5 schema 组织。
- 已补充 SLC 数据落盘契约：导入时将源 SLC 写入 `/slc/data`，统一为单波段复数据，HDF5 dtype 使用 compound `real`/`imag`，字段 dtype 保持源整数或浮点类型，数据大小不改变。
- 阶段 2 已完成第一版：LuTan 与 SAFE-like 可从目录/ZIP/TAR 成员发现和最小真实 XML 解析生成标准 `scene.h5`。后续若遇到真实产品字段差异，在阶段 3/后续数据验证中按传感器样例补兼容解析。

## 阶段 3 当前结果

- 已前置迁移 DEM 基础能力：scene corners 默认外扩 `0.2` 度生成 bbox，bbox 生成 SRTM tile 列表，本地 HGT 读取为 float32 elevation。
- 已新增 DEM HDF5 product：`/dem/elevation`、`/dem/mask`、`/dem` attrs (`crs`、`vertical_datum`、`geotransform`、`bbox`、`nodata`) 和 `/provenance` source/margin。
- 已支持从 scene.h5 的 `derived.sceneCorners` 生成 DEM product，并在 project.h5 中注册 `product_type=dem`。
- 已实现纯 Python 几何基础件：`WGS84` 椭球、`llh_to_ecef`/`ecef_to_llh` 坐标转换、`RadarGrid` 时间和距离双向转换。
- 已实现 `OrbitInterpolator`：线性位置/速度插值，支持时间排序、向量化查询和越界保护。
- 已实现 `rdr2geo`/`compute_rdr2geo_mapping`：雷达坐标→地理坐标，支持标量/向量/网格输入、零多普勒收敛、DEM 高程和 `OrbitInterpolator` 集成。
- 已实现 `geo2rdr`/`compute_geo2rdr_mapping`：地理坐标→雷达坐标，支持 `OrbitInterpolator` 集成和 `rdr2geo` 往返测试。
- 已实现 `Doppler` 类：常数和多普勒模型，HDF5 JSON 读写。
- 已实现 `DEM sampling`：从 DEM HDF5 读取，双线性插值采样，边界保护。
- 已增强几何模块：`rdr2geo` 和 `geo2rdr` 添加 bracket 模式（Brent 根查找），支持 HERMITE/LEGENDRE 轨道插值，DEM 支持 BIQUINTIC/SINC 插值。

## 阶段 4 当前结果

- 已实现粗配准模块 (`coarse.py`)：FFT 交叉相关算法，支持 center/grid/random 窗口策略，自动搜索偏移。
- 已实现精配准模块 (`fine.py`)：全图均匀分布配准点，多尺度窗口支持，16x 亚像素插值，多线程并行计算。
- 已实现偏移拟合模块 (`offset_fit.py`)：鲁棒多项式拟合（1-2 阶），迭代离群点剔除，支持 normalization 标准化。
- 已实现重采样模块 (`resampler.py`)：支持双线性/双三次/最近邻插值，边界保护和有效掩码。
- 配准流程参考 ISCE2，算法采用 C2SAR2 的精确配准实现，全部使用 Python/Numpy 实现，不依赖 ISCE3。
