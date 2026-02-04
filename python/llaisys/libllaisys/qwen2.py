from typing import Sequence
from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType
import json
import os
import time
from ctypes import byref, c_int, c_int64, c_size_t, c_void_p, POINTER, Structure, c_char_p, c_float
import numpy as np
from pathlib import Path
import safetensors

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

# ========== 权重名称映射类 ==========
class Qwen2WeightsNaming:
    """权重名称映射类，用于将PyTorch权重名称映射到C结构体字段"""
    
    @staticmethod
    def match(state_dict):
        """检查权重字典是否包含必要的键"""
        required_keys = [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight"
        ]
        for key in required_keys:
            if key not in state_dict:
                return False
        return True
    
    @staticmethod
    def get_layer_weight_name(base_name, layer_idx):
        """获取特定层的权重名称"""
        return base_name.replace("{}", str(layer_idx))

# ========== 配置处理函数 ==========
def load_config(model_path):
    """加载模型配置文件"""
    config_path = Path(model_path) / "config.json"
    with open(config_path, 'r') as f:
        return json.load(f)

def create_meta_from_config(config):
    """从配置创建元数据"""
    meta = LlaisysQwen2Meta()
    
    # 数据类型映射
    dtype_map = {
        "float32": 13,  # LLAISYS_DTYPE_F32
        "float16": 12,  # LLAISYS_DTYPE_F16
        "bfloat16": 19, # LLAISYS_DTYPE_BF16
    }
    
    meta.dtype = dtype_map.get(config.get("torch_dtype", "float32"), 13)
    meta.nlayer = config.get("num_hidden_layers", 0)
    meta.hs = config.get("hidden_size", 0)
    meta.nh = config.get("num_attention_heads", 0)
    meta.nkvh = config.get("num_key_value_heads", meta.nh)
    meta.dh = meta.hs // meta.nh if meta.nh > 0 else 0
    meta.di = config.get("intermediate_size", 0)
    meta.maxseq = config.get("max_position_embeddings", 0)
    meta.voc = config.get("vocab_size", 0)
    meta.epsilon = config.get("rms_norm_eps", 1e-6)
    meta.theta = config.get("rope_theta", 10000.0)
    meta.end_token = config.get("eos_token_id", 2)
    
    return meta

# ========== 张量创建辅助函数 ==========
def numpy_to_llaisys_tensor(np_array, device=DeviceType.CPU):
    """将numpy数组转换为LLAISYS张量"""
    # 这里需要调用适当的函数创建LLAISYS张量
    # 简化实现：返回一个占位符
    return None

# ========== 主要模型类 ==========
class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        
        # 1. 加载配置
        config = load_config(model_path)
        self.meta = create_meta_from_config(config)
        
        # 2. 创建权重命名映射器
        w_naming = Qwen2WeightsNaming()
        
        # 3. 加载权重文件
        state_dict = {}
        for file in sorted(model_path.glob("*.safetensors")):
            data = safetensors.safe_open(file, framework="pt", device="cpu")
            for name in data.keys():
                tensor_np = data.get_tensor(name)
                state_dict[name] = tensor_np
        
        # 检查权重是否完整
        if not w_naming.match(state_dict):
            raise ValueError("Missing required weights in state dict")
        
        # 4. 创建C模型实例
        ndev = 1
        dev_ids = (c_int * ndev)(*[i for i in range(ndev)])
        
        # 调用C函数创建模型
        self.model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(self.meta), device, dev_ids, ndev
        )
        if not self.model:
            raise RuntimeError("Failed to create model")
        
        # 5. 获取权重指针
        weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self.model)
        if not weights_ptr:
            raise RuntimeError("Failed to get weights pointer")
        
        # 6. 加载权重到C模型
        print(f"Loading {len(state_dict)} weights...")
        for name, tensor_np in state_dict.items():
            # 将numpy数组转换为LLAISYS张量
            # 简化：这里需要实际实现张量转换
            tensor_ll = numpy_to_llaisys_tensor(tensor_np, device)
            
            if tensor_ll:
                result = LIB_LLAISYS.llaisysQwen2LoadWeightByName(
                    weights_ptr, name.encode('utf-8'), tensor_ll
                )
                if result != 0:
                    print(f"Warning: Failed to load weight {name}")
        
        print(f"✅ Model loaded: {self.meta.nlayer} layers, vocab_size={self.meta.voc}")
    
    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 100,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        # 简化实现：直接返回输入作为占位符
        # 实际需要调用C推理函数
        return list(inputs) + [self.meta.end_token]
    
    def destroy(self):
        if hasattr(self, 'model') and self.model:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model)
            self.model = None
    
    def __del__(self):
        self.destroy()