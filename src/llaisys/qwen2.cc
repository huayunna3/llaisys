#include "llaisys/models/qwen2.h"        // 首先包含qwen2.h
#include "llaisys_tensor.hpp"           // 然后包含llaisys_tensor.hpp，提供LlaisysTensor的完整定义
#include "../tensor/tensor.hpp"        // 张量类
#include "../core/llaisys_core.hpp"    // 核心运行时
#include "../src/utils.hpp"          // 工具函数

#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"

#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <functional>

namespace {
    // 内联辅助函数，确保能访问 LlaisysTensor 的完整定义
    inline llaisys::tensor_t llaisys_to_tensor(llaisysTensor_t lt) {
        if (!lt) return nullptr;
        // 现在编译器能看到 llaisys_tensor.hpp 中的完整定义
        return reinterpret_cast<LlaisysTensor*>(lt)->tensor;
    }
}

namespace llaisys {
namespace models {

// ========== 张量操作辅助函数声明 ==========
// 这些函数应该在项目的其他部分实现
tensor_t tensor_create(
    const size_t* shape, size_t ndim, 
    llaisysDataType_t dtype, llaisysDeviceType_t device
);

void tensor_release(tensor_t tensor);
void tensor_retain(tensor_t tensor);
void tensor_load(tensor_t tensor, const void* data);
std::byte* tensor_data(tensor_t tensor);
size_t tensor_numel(tensor_t tensor);
size_t tensor_elementSize(tensor_t tensor);
size_t tensor_ndim(tensor_t tensor);
const size_t* tensor_shape(tensor_t tensor);
llaisysDataType_t tensor_dtype(tensor_t tensor);
llaisysDeviceType_t tensor_deviceType(tensor_t tensor);

// ========== 内部实现结构体 ==========
struct Qwen2ModelImpl {
    LlaisysQwen2Meta meta;
    LlaisysQwen2Weights weights;
    
    // 运行时设备信息
    llaisysDeviceType_t device;
    int ndevice;
    std::vector<int> device_ids;
    
    // KV-Cache 内部状态（使用 tensor_t）
    std::vector<tensor_t> k_cache;  // [nlayer]
    std::vector<tensor_t> v_cache;  // [nlayer]
    size_t current_pos;             // 当前缓存位置
    
    Qwen2ModelImpl(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, 
                   int *device_ids, int ndevice)
        : meta(*meta),
          weights(),
          device(device),
          ndevice(ndevice),
          device_ids(device_ids, device_ids + ndevice),
          k_cache(),
          v_cache(),
          current_pos(0)
    {
        // 初始化权重数组为正确的类型 - llaisysTensor_t
        weights.attn_norm_w = new llaisysTensor_t[meta->nlayer];
        weights.attn_q_w = new llaisysTensor_t[meta->nlayer];
        weights.attn_q_b = new llaisysTensor_t[meta->nlayer];
        weights.attn_k_w = new llaisysTensor_t[meta->nlayer];
        weights.attn_k_b = new llaisysTensor_t[meta->nlayer];
        weights.attn_v_w = new llaisysTensor_t[meta->nlayer];
        weights.attn_v_b = new llaisysTensor_t[meta->nlayer];
        weights.attn_o_w = new llaisysTensor_t[meta->nlayer];
        weights.mlp_norm_w = new llaisysTensor_t[meta->nlayer];
        weights.mlp_gate_w = new llaisysTensor_t[meta->nlayer];
        weights.mlp_up_w = new llaisysTensor_t[meta->nlayer];
        weights.mlp_down_w = new llaisysTensor_t[meta->nlayer];
        
        // 初始化所有指针为nullptr
        for (size_t i = 0; i < meta->nlayer; ++i) {
            weights.attn_norm_w[i] = nullptr;
            weights.attn_q_w[i] = nullptr;
            weights.attn_q_b[i] = nullptr;
            weights.attn_k_w[i] = nullptr;
            weights.attn_k_b[i] = nullptr;
            weights.attn_v_w[i] = nullptr;
            weights.attn_v_b[i] = nullptr;
            weights.attn_o_w[i] = nullptr;
            weights.mlp_norm_w[i] = nullptr;
            weights.mlp_gate_w[i] = nullptr;
            weights.mlp_up_w[i] = nullptr;
            weights.mlp_down_w[i] = nullptr;
        }
        
        // 初始化其他单张量权重
        weights.in_embed = nullptr;
        weights.out_embed = nullptr;
        weights.out_norm_w = nullptr;
        
        // 初始化KV-Cache
        k_cache.resize(meta->nlayer, nullptr);
        v_cache.resize(meta->nlayer, nullptr);
    }
    
    ~Qwen2ModelImpl() {
        // 释放权重数组 - 注意：不释放数组元素，因为它们是外部管理的
        delete[] weights.attn_norm_w;
        delete[] weights.attn_q_w;
        delete[] weights.attn_q_b;
        delete[] weights.attn_k_w;
        delete[] weights.attn_k_b;
        delete[] weights.attn_v_w;
        delete[] weights.attn_v_b;
        delete[] weights.attn_o_w;
        delete[] weights.mlp_norm_w;
        delete[] weights.mlp_gate_w;
        delete[] weights.mlp_up_w;
        delete[] weights.mlp_down_w;
        
        // 释放KV-Cache张量
        for (auto &tensor : k_cache) if (tensor) tensor.reset();
        for (auto &tensor : v_cache) if (tensor) tensor.reset();
    }
};

// 解析层索引，例如从 "model.layers.0.input_layernorm.weight" 提取 0
static int parse_layer_index(const std::string &name) {
    size_t pos = name.find(".layers.");
    if (pos == std::string::npos) return -1;
    pos += 8;  // ".layers."的长度
    size_t end = name.find('.', pos);
    if (end == std::string::npos) return -1;
    try {
        return std::stoi(name.substr(pos, end - pos));
    } catch (...) {
        return -1;
    }
}

} // namespace models
} // namespace llaisys

// ========== 外部C函数实现 ==========

using namespace llaisys::models;

extern "C" {

// 创建模型
__export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(
    const LlaisysQwen2Meta *meta, 
    llaisysDeviceType_t device, 
    int *device_ids, 
    int ndevice
) {
    try {
        return reinterpret_cast<LlaisysQwen2Model*>(
            new Qwen2ModelImpl(meta, device, device_ids, ndevice)
        );
    } catch (...) {
        return nullptr;
    }
}

// 销毁模型
__export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
    if (model) {
        delete reinterpret_cast<Qwen2ModelImpl*>(model);
    }
}

// 获取权重指针
__export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
    if (!model) return nullptr;
    auto impl = reinterpret_cast<Qwen2ModelImpl*>(model);
    return &impl->weights;
}

// 按名称加载权重
__export int llaisysQwen2LoadWeightByName(
    struct LlaisysQwen2Weights *weights, 
    const char *name, 
    llaisysTensor_t tensor
) {
    if (!weights || !name || !tensor) return -1;
    
    std::string weight_name(name);
    int layer = parse_layer_index(weight_name);
    
    // 根据名称映射到权重字段
    if (weight_name == "model.embed_tokens.weight") {
        weights->in_embed = tensor;
        return 0;
    }
    else if (weight_name == "model.norm.weight") {
        weights->out_norm_w = tensor;
        return 0;
    }
    else if (weight_name == "lm_head.weight") {
        weights->out_embed = tensor;
        return 0;
    }
    else if (weight_name.find("model.layers.") != std::string::npos) {
        if (layer < 0) return -1;
        
        if (weight_name.find("input_layernorm.weight") != std::string::npos) {
            weights->attn_norm_w[layer] = tensor;
            return 0;
        }
        else if (weight_name.find("self_attn.q_proj.weight") != std::string::npos) {
            weights->attn_q_w[layer] = tensor;
            return 0;
        }
        else if (weight_name.find("self_attn.q_proj.bias") != std::string::npos) {
            weights->attn_q_b[layer] = tensor;
            return 0;
        }
        else if (weight_name.find("self_attn.k_proj.weight") != std::string::npos) {
            weights->attn_k_w[layer] = tensor;
            return 0;
        }
        else if (weight_name.find("self_attn.k_proj.bias") != std::string::npos) {
            weights->attn_k_b[layer] = tensor;
            return 0;
        }
        else if (weight_name.find("self_attn.v_proj.weight") != std::string::npos) {
            weights->attn_v_w[layer] = tensor;
            return 0;
        }
        else if (weight_name.find("self_attn.v_proj.bias") != std::string::npos) {
            weights->attn_v_b[layer] = tensor;
            return 0;
        }
        else if (weight_name.find("self_attn.o_proj.weight") != std::string::npos) {
            weights->attn_o_w[layer] = tensor;
            return 0;
        }
        else if (weight_name.find("post_attention_layernorm.weight") != std::string::npos) {
            weights->mlp_norm_w[layer] = tensor;
            return 0;
        }
        else if (weight_name.find("mlp.gate_proj.weight") != std::string::npos) {
            weights->mlp_gate_w[layer] = tensor;
            return 0;
        }
        else if (weight_name.find("mlp.up_proj.weight") != std::string::npos) {
            weights->mlp_up_w[layer] = tensor;
            return 0;
        }
        else if (weight_name.find("mlp.down_proj.weight") != std::string::npos) {
            weights->mlp_down_w[layer] = tensor;
            return 0;
        }
    }
    
    return -2;  // 未找到匹配
}

// 重置KV-Cache
__export void llaisysQwen2ResetCache(struct LlaisysQwen2Model *model) {
    if (!model) return;
    auto impl = reinterpret_cast<Qwen2ModelImpl*>(model);
    impl->current_pos = 0;
    
    // 同时清理缓存张量
    for (auto &tensor : impl->k_cache) if (tensor) tensor.reset();
    for (auto &tensor : impl->v_cache) if (tensor) tensor.reset();
    impl->k_cache.resize(impl->meta.nlayer, nullptr);
    impl->v_cache.resize(impl->meta.nlayer, nullptr);
}

// 获取缓存长度
__export size_t llaisysQwen2CacheLength(struct LlaisysQwen2Model *model) {
    if (!model) return 0;
    auto impl = reinterpret_cast<Qwen2ModelImpl*>(model);
    return impl->current_pos;
}

// 获取元数据
__export const LlaisysQwen2Meta* llaisysQwen2GetMeta(struct LlaisysQwen2Model *model) {
    if (!model) return nullptr;
    auto impl = reinterpret_cast<Qwen2ModelImpl*>(model);
    return &impl->meta;
}

// 获取设备信息
__export llaisysDeviceType_t llaisysQwen2GetDevice(struct LlaisysQwen2Model *model) {
    if (!model) return LLAISYS_DEVICE_CPU;
    auto impl = reinterpret_cast<Qwen2ModelImpl*>(model);
    return impl->device;
}

// ========== 核心推理函数 ==========

// 前向传播（返回logits） - 简化实现，先让编译通过
__export llaisysTensor_t llaisysQwen2Forward(
    struct LlaisysQwen2Model *model, 
    const int64_t *token_ids, 
    size_t ntoken, 
    size_t past_len
) {
    if (!model || !token_ids || ntoken == 0) return nullptr;
    
    auto impl = reinterpret_cast<Qwen2ModelImpl*>(model);
    
    // 创建一个简单的占位符张量返回，确保编译通过
    // 实际实现需要使用所有算子完成完整的transformer前向传播
    
    // 更新缓存位置（简化）
    impl->current_pos += ntoken;
    
    // 返回nullptr作为占位符，实际需要返回有效的张量
    return nullptr;
}

// 简化推理函数（单步生成）
__export int64_t llaisysQwen2ModelInfer(
    struct LlaisysQwen2Model *model, 
    const int64_t *token_ids, 
    size_t ntoken
) {
    if (!model || !token_ids || ntoken == 0) return -1;
    
    auto impl = reinterpret_cast<Qwen2ModelImpl*>(model);
    
    // 获取当前缓存位置
    size_t past_len = impl->current_pos;
    
    // 调用前向传播获取logits
    llaisysTensor_t logits_tensor = llaisysQwen2Forward(model, token_ids, ntoken, past_len);
    if (!logits_tensor) return -1;
    
    // 将llaisysTensor_t转换为tensor_t
    llaisys::tensor_t logits = llaisys_to_tensor(logits_tensor);  // 使用完全限定名
    if (!logits) return -1;
    
    // 简化：返回第一个token的ID
    int64_t result = token_ids[0];
    
    return result;
}

} // extern "C"