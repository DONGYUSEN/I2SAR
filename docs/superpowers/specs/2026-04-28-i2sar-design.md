# I2SAR 第一版总体设计

## 1. 目标和范围

I2SAR 是一套重新设计的 SAR/InSAR 数据处理软件，目标是覆盖数据导入、数据组织、纯 Python 几何求解、配准、干涉处理、解缠、LOS 转换、RTC 处理、产品输出，并为后续时序 InSAR 和 Web 处理预留稳定接口。

第一阶段不优先绑定具体传感器，也不迁移完整算法链。第一阶段先建立统一数据模型、HDF5 数据组织、空工作流、缓存/断点续跑、Python API 和 CLI 薄入口。

核心约束：

- 内部主数据格式采用 HDF5。
- HDF5 采用混合粒度：项目级 `project.h5` 作为轻量索引，`scene`、`pair`、`product` 分别使用独立 HDF5 文件。
- `rdr2geo` / `geo2rdr` 采用纯 Python/Numpy 实现，不依赖 ISCE3 运行时；ISCE3 只可作为测试参考结果来源。
- Python API 是核心接口；CLI、配置驱动工作流和未来 Web API 都是薄封装。
- 数据模型显式区分成像模式，至少支持 `stripmap` 和 `tops`。
- 处理链显式区分 workflow 类型，至少包括 `insar` 和 `rtc`。RTC 是独立处理链，不是 InSAR 的附属导出步骤。

## 2. 包结构

```text
i2sar/
  core/            # 配置、日志、异常、ID、路径、版本、运行上下文
  model/           # Project、Scene、Pair、RadarGrid、Orbit、Doppler、DEM、Product
  hdf/             # HDF5 schema、读写器、索引、校验、迁移
  io/              # 外部产品导入器：LuTan-1、天仪、Sentinel-1 等
  geometry/        # 纯 Python/Numpy rdr2geo、geo2rdr、DEM/轨道/多普勒计算
  registration/    # 配准、偏移估计、质量评估、重采样
  interferometry/  # 干涉图、相干性、滤波、地形/平地相位
  unwrapping/      # SNAP/snaphu 封装和解缠产品管理
  displacement/    # LOS 转换、形变产品、投影/地理编码输出
  rtc/             # 辐射校正、地形校正、RTC factor、RTC 产品
  workflow/        # DAG、阶段状态、缓存、断点续跑、任务报告
  timeseries/      # 时序 InSAR 预留接口
  web/             # Web/API schema 预留，不放核心算法
  cli/             # 命令行入口，薄封装
tests/
docs/
```

核心对象分为三层：

- 数据实体：`Project`、`Scene`、`Pair`、`Product`
- 几何/雷达模型：`RadarGrid`、`Orbit`、`DopplerModel`、`DEMGrid`、`GeoGrid`
- 处理结果：`GeometryProduct`、`RegistrationProduct`、`InterferogramProduct`、`UnwrappedProduct`、`LOSProduct`、`RTCProduct`、`ExportProduct`

## 3. HDF5 组织

项目目录采用：

```text
project/
  project.h5
  scenes/
    <scene_id>.h5
  pairs/
    <pair_id>.h5
  products/
    <product_id>.h5
  exports/
    ...
```

`project.h5` 只存轻量索引、工作流状态和 provenance，不存大数组：

```text
/
  attrs:
    i2sar_schema_version
    i2sar_version
    project_id
    created_at
    updated_at

  /project
  /scenes      # scene_id、file_path、sensor、acquisition_mode、time、status
  /pairs       # pair_id、master、slave、file_path、baseline、status
  /products    # product_id、product_type、owner、file_path、status
  /workflow
  /provenance
```

`scenes/<scene_id>.h5` 存单景数据：

```text
/
  attrs:
    schema_version
    entity_type = "scene"
    scene_id
    sensor
    acquisition_mode
    acquisition_time

  /metadata
    /source
    /acquisition
    /quality

  /radar_grid
  /orbit
  /doppler
  /slc
    data[azimuth, range]

  /stripmap
    attrs:
      continuous_azimuth = true

  /tops
    /swaths
    /bursts
    /burst_grid
    /azimuth_steering

  /derived
  /provenance
```

`pairs/<pair_id>.h5` 存干涉对数据：

```text
/
  attrs:
    schema_version
    entity_type = "pair"
    pair_id
    master_scene_id
    slave_scene_id

  /metadata
  /registration
  /geometry
  /interferometry
  /unwrapping
  /displacement
  /provenance
```

`products/<product_id>.h5` 存最终产品和派生产品。RTC 产品结构：

```text
/
  attrs:
    entity_type = "product"
    product_type = "rtc"
    source_scene_id
    grid_type = "geo"

  /grid
  /rtc
    sigma0[]
    gamma0[]
    beta0_optional[]
    rtc_factor[]
    incidence_angle[]
    mask[]
  /provenance
```

所有大数组必须 chunked，并设置 compression。每个 HDF5 文件必须有 `schema_version`。数组必须写入 `units`、`description`、`grid_ref` 和 `nodata` 等属性。

## 4. 数据流和模块职责

主数据流：

```text
外部 SAR 产品
  -> io.import_scene()
  -> scenes/<scene_id>.h5

Scene 选择 / Pair 创建
  -> workflow.create_pair()
  -> pairs/<pair_id>.h5

Pair 几何求解
  -> geometry.rdr2geo()
  -> geometry.geo2rdr()
  -> pair.h5:/geometry

配准
  -> registration.estimate_offsets()
  -> registration.resample_slave()
  -> pair.h5:/registration

干涉
  -> interferometry.form_interferogram()
  -> interferometry.estimate_coherence()
  -> interferometry.remove_topo_flat_phase()
  -> pair.h5:/interferometry

解缠
  -> unwrapping.unwrap()
  -> pair.h5:/unwrapping

LOS 转换
  -> displacement.phase_to_los()
  -> pair.h5:/displacement

RTC
  -> rtc.compute_factor()
  -> rtc.terrain_correct()
  -> rtc.radiometric_normalize()
  -> products/<rtc_product_id>.h5
```

`io/` 只负责外部产品到 I2SAR 内部 `Scene`。导入器需要同时表达 `sensor` 和 `acquisition_mode`，不能把传感器类型作为唯一处理分支。

`geometry/` 只依赖标准 `RadarGrid`、`Orbit`、`DopplerModel`、`DEMGrid`，不读取传感器原始目录结构。

`workflow/` 只编排，不实现算法。所有任务调用 Python API。

## 5. 工作流、缓存和断点续跑

任务模型：

```text
TaskSpec:
  task_id
  owner_type      # project / scene / pair / product
  owner_id
  inputs
  outputs
  params
```

状态机：

```text
pending -> running -> completed
pending -> running -> failed
pending -> skipped
completed -> invalidated
```

缓存判断使用 `task_hash`：

```text
task_hash = hash(
  task_name
  + i2sar_version
  + schema_version
  + input_dataset_refs
  + input_content_hash_or_mtime
  + params
)
```

断点续跑规则：

- output 存在、schema 校验通过、`task_hash` 一致：跳过。
- output 存在但 `task_hash` 不一致：标记 invalidated 并重算。
- output 不完整：清理临时 group 后重算。
- 上游失败：下游任务保持 pending 或 blocked。

任务写入先进入临时 group，校验完整后再更新最终 group 和 `project.h5` 状态。

## 6. Workflow 类型

`SceneWorkflow`：

- `import_scene`
- `validate_scene`
- `multilook`
- `quicklook`

`InSARWorkflow`：

- `create_pair`
- `prepare_geometry`
- `register`
- `form_interferogram`
- `unwrap`
- `phase_to_los`
- `export`

`RTCWorkflow`：

- `prepare_geometry`
- `compute_rtc_factor`
- `terrain_correct`
- `radiometric_normalize`
- `geocode`
- `make_rtc_product`
- `export`

`ProjectWorkflow`：

- `build_pairs`
- `batch_run_pairs`
- `batch_run_rtc`
- `build_timeseries_ready_stack`
- `summarize_products`

## 7. 测试与验证

测试分四层：schema/model、导入、算法、工作流。

第一阶段 MVP 必须覆盖：

- `Project` / `Scene` / `Pair` / `Product` HDF5 创建、读取和校验。
- `project.h5` 索引到实体文件的路径有效性。
- 空 workflow 的执行、跳过、失败记录和 provenance 写入。
- `stripmap` / `tops` 成像模式字段可表达。
- `insar` / `rtc` workflow 类型可表达。
- CLI 调用同一套 Python API。

导入测试覆盖：

- `sensor`: `lutan`、`tianyi`、`sentinel1`、未来传感器扩展。
- `acquisition_mode`: `stripmap`、`tops`。
- TOPS 必须验证 swath、burst、burst grid 和 azimuth steering。
- Stripmap 必须验证连续方位维和无 burst 时的正确处理。

纯 Python 几何验证分三级：

- 数学单元测试：轨道插值、多普勒模型、椭球转换、DEM 采样、斜距方程、零多普勒方程、迭代收敛。
- 合成场景 roundtrip：`rdr2geo -> geo2rdr`，方位/距离误差目标小于 `1e-3` 像素。
- 真实数据对照：使用 D2SAR/C2SAR2/ISCE3 生成的 reference 数据，统计 median、p95、max residual。

InSAR 验证覆盖：

- 配准 offset 和重采样结果。
- 干涉图、相干性、滤波、解缠、LOS。
- Stripmap pair 和 TOPS pair。

RTC 验证覆盖：

- 单景 RTC workflow。
- RTC factor、terrain correction、sigma0/gamma0、地理编码产品。
- RTC product HDF5 schema。

## 8. 第一阶段交付

第一阶段交付内容：

- 标准 Python 包骨架。
- `Project`、`Scene`、`Pair`、`Product` 数据模型。
- HDF5 schema 创建、读取、校验。
- `project.h5` 索引和实体文件管理。
- 空 `SceneWorkflow`、`InSARWorkflow`、`RTCWorkflow`。
- `task_hash`、任务状态、provenance、resume/skipped 逻辑。
- Python API 和 CLI 薄入口。
- 最小测试集。

