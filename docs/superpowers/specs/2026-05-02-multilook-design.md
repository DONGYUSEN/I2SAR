# 多视处理模块 (Multilook) 设计规格

## 概述

对复数SAR数据（SLC）进行多视处理，通过在距离向和方位向分别进行平均来降低分辨率、提高信噪比，同时保持相位信息。

## 功能需求

### 核心功能
1. **相位保持多视**：分别对振幅和相位进行处理，重建复数结果
2. **分块处理**：支持GB级大型SAR图像，避免内存溢出
3. **CLI工具**：命令行接口方便集成到处理流程
4. **Python API**：供其他模块调用

### 算法规格

#### 相位保持多视
```
输入: 复数数据 c = a * exp(jφ)
1. 振幅提取: amp = |c|
2. 相位提取: phase = arg(c)
3. 振幅多视: amp_ml = mean(amp, axis=look_dim)
4. 相位多视: phase_ml = angle(sum(exp(j*phase), axis=look_dim))
5. 重建: result = amp_ml * exp(j*phase_ml)
```

#### 参数
- `nalks`: 方位向视数 (1-16)
- `nrlks`: 距离向视数 (1-8)
- `boundary`: 'crop' 裁剪不能整除部分

## 技术规格

### 模块位置
`i2sar/processing/multilook.py`

### 函数接口

```python
def multilook(
    data: np.ndarray,
    nalks: int,
    nrlks: int,
    boundary: str = 'crop'
) -> np.ndarray:
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
```

### CLI接口

```bash
i2sar multilook <input> <output> --nalks <n> --nrlks <m> [--chunk-lines <k>]
```

### 数据格式
- 输入/输出: complex64 (复数float32)
- TIFF格式存储

## 实现细节

### 分块策略
1. 方位向按 `chunk_lines * nalks` 行读取
2. 每块处理流程：
   - 读取 `nalks * chunk_lines` 行原始数据
   - 对每 `nalks` 行进行多视平均
   - 产生 `chunk_lines` 行输出
3. 距离向无需分块，直接在块内处理

### 地理信息
- 输出GeoTIFF保持输入的地理参考
- 像素分辨率自动调整为 `pixel_size * nrlks` / `pixel_size * nalks`

## 组件清单

| 组件 | 路径 | 职责 |
|------|------|------|
| multilook() | processing/multilook.py | 内存数据多视 |
| multilook_chunked() | processing/multilook.py | 文件分块多视 |
| multilook_cli() | cli/main.py | 命令行入口 |

## 验收标准

1. **精度验证**：
   - 振幅多视结果与GMTSAR一致（误差<1e-6）
   - 相位多视使用矢量平均，与ISCE2一致

2. **性能要求**：
   - 分块处理10000x10000图像不超过2GB内存
   - 处理速度优于逐行处理

3. **边界处理**：
   - 不能整除时自动裁剪多余像素
   - 不进行填充或平滑

4. **输出正确性**：
   - 输出尺寸 = 输入尺寸 / (nalks * nrlks)
   - 地理参考信息正确传递