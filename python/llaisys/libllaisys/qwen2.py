#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen2模型C API的ctypes封装
"""

from ctypes import *
import numpy as np
import sys
import os

# 导入主库
from . import LIB_LLAISYS

# ========== 数据类型映射 ==========
def llaisys_dtype_to_numpy(dtype):
    """将LLAISYS数据类型转换为numpy数据类型"""
    dtype_map = {
        0: np.float32,   # LLAISYS_DTYPE_F32
        1: np.float16,   # LLAISYS_DTYPE_F16
        2: np.float16,   # LLAISYS_DTYPE_BF16 (近似)
        3: np.int64,     # LLAISYS_DTYPE_I64
    }
    return dtype_map.get(dtype, np.float32)

def numpy_to_llaisys_dtype(np_dtype):
    """将numpy数据类型转换为LLAISYS数据类型"""
    if np_dtype == np.float32:
        return 0  # LLAISYS_DTYPE_F32
    elif np_dtype == np.float16:
        return 1  # LLAISYS_DTYPE_F16
    elif np_dtype == np.int64:
        return 3  # LLAISYS_DTYPE_I64
    else:
        return 0  # 默认float32

# ========== C结构体定义 ==========
class LlaisysQwen2Meta(Structure):
    _fields_ = [
        ('dtype', c_int),
        ('nlayer', c_size_t),
        ('hs', c_size_t),
        ('nh', c_size_t),
        ('nkvh', c_size_t),
        ('dh', c_size_t),
        ('di', c_size_t),
        ('maxseq', c_size_t),
        ('voc', c_size_t),
        ('epsilon', c_float),
        ('theta', c_float),
        ('end_token', c_int64)
    ]

class LlaisysQwen2Weights(Structure):
    _fields_ = [
        ('in_embed', c_void_p),
        ('out_embed', c_void_p),
        ('out_norm_w', c_void_p),
        ('attn_norm_w', POINTER(c_void_p)),
        ('attn_q_w', POINTER(c_void_p)),
        ('attn_q_b', POINTER(c_void_p)),
        ('attn_k_w', POINTER(c_void_p)),
        ('attn_k_b', POINTER(c_void_p)),
        ('attn_v_w', POINTER(c_void_p)),
        ('attn_v_b', POINTER(c_void_p)),
        ('attn_o_w', POINTER(c_void_p)),
        ('mlp_norm_w', POINTER(c_void_p)),
        ('mlp_gate_w', POINTER(c_void_p)),
        ('mlp_up_w', POINTER(c_void_p)),
        ('mlp_down_w', POINTER(c_void_p))
    ]

# ========== 函数原型声明 ==========
def init_qwen2_functions():
    """初始化Qwen2相关C函数原型"""
    
    # 模型生命周期
    LIB_LLAISYS.llaisysQwen2ModelCreate.argtypes = [
        POINTER(LlaisysQwen2Meta),  # meta
        c_int,                      # device type
        POINTER(c_int),             # device_ids
        c_int                       # ndevice
    ]
    LIB_LLAISYS.llaisysQwen2ModelCreate.restype = c_void_p
    
    LIB_LLAISYS.llaisysQwen2ModelDestroy.argtypes = [c_void_p]
    LIB_LLAISYS.llaisysQwen2ModelDestroy.restype = None
    
    # 权重管理
    LIB_LLAISYS.llaisysQwen2ModelWeights.argtypes = [c_void_p]
    LIB_LLAISYS.llaisysQwen2ModelWeights.restype = POINTER(LlaisysQwen2Weights)
    
    LIB_LLAISYS.llaisysQwen2LoadWeightByName.argtypes = [
        POINTER(LlaisysQwen2Weights),
        c_char_p,
        c_void_p  # tensor_t
    ]
    LIB_LLAISYS.llaisysQwen2LoadWeightByName.restype = c_int
    
    # KV-Cache管理
    LIB_LLAISYS.llaisysQwen2ResetCache.argtypes = [c_void_p]
    LIB_LLAISYS.llaisysQwen2ResetCache.restype = None
    
    LIB_LLAISYS.llaisysQwen2CacheLength.argtypes = [c_void_p]
    LIB_LLAISYS.llaisysQwen2CacheLength.restype = c_size_t
    
    # 推理接口
    LIB_LLAISYS.llaisysQwen2Forward.argtypes = [
        c_void_p,      # model
        POINTER(c_int64),  # token_ids
        c_size_t,      # ntoken
        c_size_t       # past_len
    ]
    LIB_LLAISYS.llaisysQwen2Forward.restype = c_void_p  # tensor_t
    
    LIB_LLAISYS.llaisysQwen2ModelInfer.argtypes = [
        c_void_p,      # model
        POINTER(c_int64),  # token_ids
        c_size_t       # ntoken
    ]
    LIB_LLAISYS.llaisysQwen2ModelInfer.restype = c_int64
    
    # 信息获取
    LIB_LLAISYS.llaisysQwen2GetMeta.argtypes = [c_void_p]
    LIB_LLAISYS.llaisysQwen2GetMeta.restype = POINTER(LlaisysQwen2Meta)
    
    LIB_LLAISYS.llaisysQwen2GetDevice.argtypes = [c_void_p]
    LIB_LLAISYS.llaisysQwen2GetDevice.restype = c_int
    
    print("✅ Qwen2 C functions initialized")

# ========== Python辅助类 ==========
class Qwen2Tensor:
    """Python中包装LLAISYS张量"""
    def __init__(self, tensor_ptr, shape=None, dtype=None):
        self.ptr = tensor_ptr
        self.shape = shape
        self.dtype = dtype
    
    def numpy(self):
        """将张量转换为numpy数组（简化版）"""
        if not self.ptr:
            return None
        
        # 这里需要调用C函数获取张量数据
        # 简化实现：假设可以通过其他方式获取
        return None
    
    def __del__(self):
        if self.ptr:
            # 释放张量
            pass

# ========== 初始化函数 ==========
# 在模块导入时初始化
init_qwen2_functions()

# ========== 导出 ==========
__all__ = [
    'LlaisysQwen2Meta',
    'LlaisysQwen2Weights',
    'init_qwen2_functions',
    'Qwen2Tensor',
    'llaisys_dtype_to_numpy',
    'numpy_to_llaisys_dtype'
]