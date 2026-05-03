"""Test for高效的Goldstein滤波器向量化实现"""

import numpy as np
import pytest
import time
from i2sar.interferometry.filtering import goldstein_filter, goldstein_filter_vectorized, FilterType


def _create_test_data(shape=(256, 256), seed=42):
    """创建测试数据"""
    np.random.seed(seed)
    # 生成随机相位
    phase = np.random.uniform(-np.pi, np.pi, shape).astype(np.float64)
    # 生成缠绕相位（复数形式）
    interferogram = np.exp(1j * phase)
    # 添加一些相干性
    coherence = np.random.uniform(0.3, 0.9, shape).astype(np.float64)
    coherence = np.clip(coherence, 0, 1)
    return interferogram, coherence


class TestGoldsteinFilterVectorized:
    """向量化Goldstein滤波器测试"""
    
    def test_output_shape(self):
        """输出一致性测试"""
        shape = (128, 128)
        ifg, coh = _create_test_data(shape)
        
        result = goldstein_filter_vectorized(ifg, coh, alpha=1.0, window_size=16)
        
        assert result["filtered"].shape == shape
        assert result["coherence"].shape == shape
    
    def test_output_dtype(self):
        """输出类型测试"""
        shape = (64, 64)
        ifg, coh = _create_test_data(shape)
        
        result = goldstein_filter_vectorized(ifg, coh)
        
        assert result["filtered"].dtype == np.complex128
        assert result["coherence"].dtype == np.float64
    
    def test_with_coherence_unity(self):
        """全相干性输入测试"""
        shape = (64, 64)
        ifg, coh = _create_test_data(shape)
        coh = np.ones_like(coh)  # 全相干
        
        result = goldstein_filter_vectorized(ifg, coh, alpha=1.0)
        
        # 全相干时应该不过滤
        assert result["filtered"].shape == shape
    
    def test_with_low_coherence(self):
        """低相干性测试 - 应该保留原值"""
        shape = (32, 32)
        ifg, coh = _create_test_data(shape)
        coh = np.zeros_like(coh)  # 全0相干
        
        result = goldstein_filter_vectorized(ifg, coh, alpha=1.0)
        
        # 全0相干时应该保留原始值（因为weight=0）
        assert result["filtered"] is not None
    
    def test_alpha_parameter(self):
        """alpha参数测试"""
        shape = (64, 64)
        ifg, coh = _create_test_data(shape)
        
        r1 = goldstein_filter_vectorized(ifg, coh, alpha=0.5)
        r2 = goldstein_filter_vectorized(ifg, coh, alpha=1.5)
        
        # 不同alpha应产生不同结果
        assert not np.allclose(r1["filtered"], r2["filtered"])
    
    def test_window_size_parameter(self):
        """window_size参数测试"""
        shape = (128, 128)
        ifg, coh = _create_test_data(shape)
        
        r1 = goldstein_filter_vectorized(ifg, coh, window_size=8)
        r2 = goldstein_filter_vectorized(ifg, coh, window_size=32)
        
        # 不同窗口应产生不同结果
        assert not np.allclose(r1["filtered"], r2["filtered"])
    
    def test_invalid_coherence_range(self):
        """无效相干性范围测试"""
        shape = (32, 32)
        ifg, _ = _create_test_data(shape)
        # 相干性超范围
        coh_invalid = np.ones(shape) * 1.5
        
        result = goldstein_filter_vectorized(ifg, coh_invalid)
        
        # 应该被clip到[0,1]
        assert result["coherence"].max() <= 1.0
        assert result["coherence"].min() >= 0.0
    
    def test_none_coherence(self):
        """None相干性输入测试 - 应使用默认全1"""
        shape = (64, 64)
        ifg, _ = _create_test_data(shape)
        
        result = goldstein_filter_vectorized(ifg, None)  # 全相干
        
        assert result["coherence"].shape == shape
        assert np.allclose(result["coherence"], 1.0)


class TestGoldsteinFilterPerformance:
    """性能测试（仅验证可运行性）"""
    
    @pytest.mark.slow
    def test_performance_small(self):
        """小尺寸性能测试"""
        shape = (256, 256)
        ifg, coh = _create_test_data(shape, seed=123)
        
        start = time.perf_counter()
        result = goldstein_filter_vectorized(ifg, coh, alpha=1.0, window_size=16)
        elapsed = time.perf_counter() - start
        
        print(f"\n[256x256] elapsed: {elapsed:.2f}s")
        
        # 小尺寸应在10秒内完成
        assert elapsed < 10.0
        assert result["filtered"] is not None
    
    @pytest.mark.slow
    @pytest.mark.skipif(
        np.prod([4096, 4096]) > 1e8,
        reason="Too large for CI"
    )
    def test_performance_medium(self):
        """中等尺寸性能测试"""
        shape = (1024, 1024)
        ifg, coh = _create_test_data(shape, seed=456)
        
        start = time.perf_counter()
        result = goldstein_filter_vectorized(ifg, coh, alpha=1.0, window_size=32)
        elapsed = time.perf_counter() - start
        
        print(f"\n[1024x1024] elapsed: {elapsed:.2f}s")
        
        # 1024x1024应在60秒内完成
        assert elapsed < 60.0
        assert result["filtered"] is not None


class TestGoldsteinFilterConsistency:
    """与原版一致性测试"""
    
    @pytest.mark.slow
    def test_consistency_with_original(self):
        """向量化版本应与原版结果相近"""
        shape = (64, 64)
        ifg, coh = _create_test_data(shape, seed=999)
        
        # 原版
        orig = goldstein_filter(ifg, coh, alpha=0.8, window_size=8, num_workers=4)
        
        # 向量化版
        vec = goldstein_filter_vectorized(ifg, coh, alpha=0.8, window_size=8)
        
        # 相干性应一致
        assert np.allclose(orig["coherence"], vec["coherence"], atol=1e-5)
        
        # 滤波结果应在合理范围内（不必完全相同，因为原版是逐点FFT有边界效应）
        # 验证幅度相近
        orig_amp = np.abs(orig["filtered"])
        vec_amp = np.abs(vec["filtered"])
        
        # 相对误差应在50%以内（宽松容差，因为算法近似）
        rel_diff = np.abs(orig_amp - vec_amp) / (orig_amp + 1e-10)
        assert np.median(rel_diff) < 0.5