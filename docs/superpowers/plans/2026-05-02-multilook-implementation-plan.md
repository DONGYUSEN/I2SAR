# 多视处理模块 (Multilook) 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现相位保持多视处理模块，支持分块处理大型SAR图像和CLI工具

**Architecture:** 使用NumPy向量化操作实现相位保持多视算法：分别对振幅和相位进行多视处理后重建。分块处理使用GDAL读写大型TIFF文件。

**Tech Stack:** NumPy, GDAL, Numba (可选加速)

---

## 文件结构

### 新建文件
- `i2sar/processing/multilook.py` - 多视处理核心模块
- `i2sar/processing/__init__.py` - 模块导出
- `tests/test_multilook.py` - 单元测试

### CLI扩展
- `i2sar/cli/main.py` - 添加multilook子命令

---

## 实现任务

### Task 1: 创建多视处理核心模块

**Files:**
- Create: `i2sar/processing/multilook.py`
- Test: `tests/test_multilook.py`

#### 步骤 1: 编写基础多视测试

```python
import numpy as np
import pytest

def test_multilook_basic():
    """测试基本多视处理"""
    # 创建简单测试数据: 4x4复数数组，全部相同值
    data = np.ones((4, 4), dtype=np.complex64) * (1 + 1j)
    nalks, nrlks = 2, 2

    from i2sar.processing.multilook import multilook
    result = multilook(data, nalks, nrlks)

    # 期望输出: 2x2 数组，值相同
    assert result.shape == (2, 2)
    assert np.allclose(result, (1 + 1j))
```

#### 步骤 2: 运行测试验证失败

Run: `cd /home/ysdong/Software/I2SAR && uv run pytest tests/test_multilook.py::test_multilook_basic -v`
Expected: FAIL - "module 'i2sar.processing.multilook' has no attribute 'multilook'"

#### 步骤 3: 实现基础multilook函数

```python
import numpy as np

def multilook(data: np.ndarray, nalks: int, nrlks: int, boundary: str = 'crop') -> np.ndarray:
    """
    相位保持多视处理

    Args:
        data: 输入复数数组 (rows, cols), dtype=complex64
        nalks: 方位向视数
        nrlks: 距离向视数
        boundary: 'crop' 裁剪不能整除部分

    Returns:
        多视后复数数组, dtype=complex64
    """
    rows, cols = data.shape
    out_rows = rows // nalks
    out_cols = cols // nrlks

    if boundary == 'crop':
        data = data[:out_rows * nalks, :out_cols * nrlks]

    # 振幅和相位分离
    amplitude = np.abs(data)
    phase = np.angle(data)

    # 振幅多视 - 矢量平均
    amp = amplitude.reshape(out_rows, nalks, out_cols, nrlks).sum(axis=(1, 3)) / (nalks * nrlks)

    # 相位多视 - 矢量平均
    phase_complex = np.exp(1j * phase)
    phase_complex = phase_complex.reshape(out_rows, nalks, out_cols, nrlks).sum(axis=(1, 3))
    phase_ml = np.angle(phase_complex)

    # 重建复数
    result = amp * np.exp(1j * phase_ml)
    return result.astype(np.complex64)
```

#### 步骤 4: 运行测试验证通过

Run: `cd /home/ysdong/Software/I2SAR && uv run pytest tests/test_multilook.py::test_multilook_basic -v`
Expected: PASS

#### 步骤 5: 提交代码

```bash
git add i2sar/processing/multilook.py tests/test_multilook.py
git commit -m "feat: 添加多视处理核心模块"
```

---

### Task 2: 实现分块文件处理

**Files:**
- Modify: `i2sar/processing/multilook.py:120-180`

#### 步骤 1: 添加分块测试

```python
def test_multilook_chunked(tmp_path):
    """测试分块多视处理"""
    from osgeo import gdal
    import numpy as np

    # 创建测试输入文件 (1000x1000)
    input_path = tmp_path / "input.tiff"
    data = np.random.rand(1000, 1000).astype(np.float32) + 1j * np.random.rand(1000, 1000).astype(np.float32)
    ds = gdal.GetDriverByName('GTiff').Create(str(input_path), 1000, 1000, 1, gdal.GDT_CFloat32)
    ds.GetRasterBand(1).WriteArray(data)
    ds = None

    output_path = tmp_path / "output.tiff"

    from i2sar.processing.multilook import multilook_chunked
    result = multilook_chunked(str(input_path), str(output_path), nalks=4, nrlks=4, chunk_lines=100)

    assert result == True
    assert output_path.exists()

    # 验证输出尺寸
    out_ds = gdal.Open(str(output_path))
    assert out_ds.RasterXSize == 250  # 1000/4
    assert out_ds.RasterYSize == 250
```

#### 步骤 2: 运行测试验证失败

Run: `cd /home/ysdong/Software/I2SAR && uv run pytest tests/test_multilook.py::test_multilook_chunked -v`
Expected: FAIL - "multilook_chunked not defined"

#### 步骤 3: 实现multilook_chunked函数

```python
from pathlib import Path
import numpy as np
from osgeo import gdal

def multilook_chunked(
    input_path: str | Path,
    output_path: str | Path,
    nalks: int,
    nrlks: int,
    chunk_lines: int = 1000
) -> bool:
    """
    分块多视处理（文件到文件）

    Args:
        input_path: 输入TIFF文件路径
        output_path: 输出TIFF文件路径
        nalks: 方位向视数
        nrlks: 距离向视数
        chunk_lines: 分块行数（输出行数）

    Returns:
        处理是否成功
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    ds = gdal.Open(str(input_path))
    if ds is None:
        raise ValueError(f"无法打开文件: {input_path}")

    band = ds.GetRasterBand(1)
    in_rows = ds.RasterYSize
    in_cols = ds.RasterXSize
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    ds = None

    out_rows = in_rows // nalks
    out_cols = in_cols // nrlks

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(str(output_path), out_cols, out_rows, 1, gdal.GDT_CFloat32,
                          options=['COMPRESS=LZW', 'TILED=YES'])
    out_ds.SetGeoTransform(geotransform)
    out_ds.SetProjection(projection)
    out_band = out_ds.GetRasterBand(1)

    for out_row in range(0, out_rows, chunk_lines):
        chunk_size = min(chunk_lines, out_rows - out_row)
        read_rows = chunk_size * nalks

        data = band.ReadAsArray(0, out_row * nalks, in_cols, read_rows)
        if data is None:
            raise ValueError(f"读取分块失败 at row {out_row * nalks}")

        ml_data = multilook(data.astype(np.complex64), nalks, nrlks)
        out_band.WriteArray(ml_data, 0, out_row)
        print(f"进度: {out_row + chunk_size}/{out_rows} 行")

    out_ds = None
    return True
```

#### 步骤 4: 运行测试验证通过

Run: `cd /home/ysdong/Software/I2SAR && uv run pytest tests/test_multilook.py::test_multilook_chunked -v`
Expected: PASS

#### 步骤 5: 提交代码

```bash
git add i2sar/processing/multilook.py tests/test_multilook.py
git commit -m "feat: 添加分块多视处理功能"
```

---

### Task 3: 添加相位保持验证测试

**Files:**
- Modify: `tests/test_multilook.py`

#### 步骤 1: 添加相位保持测试

```python
def test_multilook_preserves_phase():
    """测试相位保持多视处理"""
    # 创建恒定相位数据
    phase_const = np.pi / 4  # 45度
    data = np.ones((8, 8), dtype=np.complex64) * np.exp(1j * phase_const)
    nalks, nrlks = 2, 2

    result = multilook(data, nalks, nrlks)

    # 相位应该保持
    result_phase = np.angle(result)
    expected_phase = np.pi / 4
    assert np.allclose(result_phase, expected_phase, atol=1e-6)
```

#### 步骤 2: 运行测试验证通过

Run: `cd /home/ysdong/Software/I2SAR && uv run pytest tests/test_multilook.py::test_multilook_preserves_phase -v`
Expected: PASS

#### 步骤 3: 提交代码

```bash
git add tests/test_multilook.py
git commit -m "test: 添加相位保持验证测试"
```

---

### Task 4: 添加CLI命令

**Files:**
- Modify: `i2sar/cli/main.py:25-38`

#### 步骤 1: 添加CLI测试

```python
def test_multilook_cli(tmp_path):
    """测试multilook CLI命令"""
    from osgeo import gdal
    import subprocess

    # 创建测试输入
    input_path = tmp_path / "input.tiff"
    data = np.random.rand(100, 100).astype(np.float32) + 1j * np.random.rand(100, 100).astype(np.float32)
    ds = gdal.GetDriverByName('GTiff').Create(str(input_path), 100, 100, 1, gdal.GDT_CFloat32)
    ds.GetRasterBand(1).WriteArray(data)
    ds = None

    output_path = tmp_path / "output.tiff"

    # 运行CLI
    result = subprocess.run([
        'uv', 'run', 'i2sar', 'multilook',
        str(input_path), str(output_path),
        '--nalks', '2', '--nrlks', '2'
    ], capture_output=True, text=True)

    assert result.returncode == 0
    assert output_path.exists()
```

#### 步骤 2: 运行测试验证失败

Run: `cd /home/ysdong/Software/I2SAR && uv run pytest tests/test_cli.py::test_multilook_cli -v`
Expected: FAIL - "multilook command not found"

#### 步骤 3: 修改cli/main.py添加multilook子命令

在 `_build_parser()` 函数中添加：

```python
multilook_parser = subparsers.add_parser("multilook")
multilook_parser.add_argument("input", help="输入TIFF文件")
multilook_parser.add_argument("output", help="输出TIFF文件")
multilook_parser.add_argument("--nalks", type=int, default=1, help="方位向视数")
multilook_parser.add_argument("--nrlks", type=int, default=1, help="距离向视数")
multilook_parser.add_argument("--chunk-lines", type=int, default=1000, help="分块行数")
```

在 `main()` 函数中添加：

```python
if args.command == "multilook":
    from i2sar.processing.multilook import multilook_chunked
    from pathlib import Path

    success = multilook_chunked(
        input_path=args.input,
        output_path=args.output,
        nalks=args.nalks,
        nrlks=args.nrlks,
        chunk_lines=args.chunk_lines
    )
    return 0 if success else 1
```

#### 步骤 4: 运行测试验证通过

Run: `cd /home/ysdong/Software/I2SAR && uv run pytest tests/test_cli.py::test_multilook_cli -v`
Expected: PASS

#### 步骤 5: 提交代码

```bash
git add i2sar/cli/main.py tests/test_cli.py
git commit -m "feat: 添加multilook CLI命令"
```

---

### Task 5: 更新模块导出

**Files:**
- Modify: `i2sar/processing/__init__.py`

#### 步骤 1: 更新导出

```python
from .multilook import multilook, multilook_chunked

__all__ = ["multilook", "multilook_chunked"]
```

#### 步骤 2: 验证导入

Run: `cd /home/ysdong/Software/I2SAR && uv run python -c "from i2sar.processing import multilook, multilook_chunked; print('OK')"`
Expected: OK

#### 步骤 3: 提交代码

```bash
git add i2sar/processing/__init__.py
git commit -m "feat: 导出multilook模块"
```

---

## 验证清单

完成所有任务后，运行完整测试：

```bash
cd /home/ysdong/Software/I2SAR && uv run pytest tests/test_multilook.py -v
```

验证项：
- [ ] 基本多视处理正确
- [ ] 相位保持功能正确
- [ ] 分块处理输出尺寸正确
- [ ] CLI命令工作正常
- [ ] 地理参考信息正确传递