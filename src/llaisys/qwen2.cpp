#include "llaisys/models/qwen2.h"
#include "../tensor/tensor.hpp"        // 张量类
#include "../core/llaisys_core.hpp"    // 核心运行时
#include "../src/utils.hpp"          // 工具函数

#include "../ops/argmax/op.hpp"         // argmax
#include "../ops/embedding/op.hpp"      // embedding
#include "../ops/linear/op.hpp"         // linear
#include "../ops/rms_norm/op.hpp"       // rms_norm
#include "../ops/rope/op.hpp"           // rope
#include "../ops/self_attention/op.hpp" // self_attention
#include "../ops/swiglu/op.hpp"         // swiglu

#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

namespace llaisys {
namespace models {

// 张量操作辅助函数
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
    
    // KV-Cache 内部状态
    std::vector<tensor_t> k_cache;  // [nlayer, maxseq, nkvh, dh]
    std::vector<tensor_t> v_cache;  // [nlayer, maxseq, nkvh, dh]
    size_t current_pos;             // 当前缓存位置
    
    // 运行时设备信息
    llaisysDeviceType_t device;
    int ndevice;
    std::vector<int> device_ids;
    
    // 内部使用的临时张量池（可选）
    // std::vector<tensor_t> tensor_pool;
    
    Qwen2ModelImpl(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice)
        : meta(*meta), device(device), ndevice(ndevice), current_pos(0) {
        
        if (device_ids) {
            this->device_ids.assign(device_ids, device_ids + ndevice);
        }
        
        // 初始化权重数组（指针数组）
        weights.attn_norm_w = new tensor_t[meta->nlayer];
        weights.attn_q_w = new tensor_t[meta->nlayer];
        weights.attn_q_b = new tensor_t[meta->nlayer];
        // ... 其他权重数组类似
        
        // 初始化KV-Cache
        k_cache.resize(meta->nlayer);
        v_cache.resize(meta->nlayer);
    }
    
    ~Qwen2ModelImpl() {
        // 释放权重数组
        delete[] weights.attn_norm_w;
        delete[] weights.attn_q_w;
        delete[] weights.attn_q_b;
        // ... 其他权重数组类似
        
        // 释放KV-Cache张量
        for (auto &tensor : k_cache) if (tensor) tensor->release();
        for (auto &tensor : v_cache) if (tensor) tensor->release();
    }
};

// ========== C API 实现 ==========

// 权重名称到结构体字段的映射
static std::unordered_map<std::string, std::function<void(LlaisysQwen2Weights*, tensor_t)>> weight_map = {
    // embedding
    {"model.embed_tokens.weight", [](auto w, auto t){ w->in_embed = t; }},
    {"lm_head.weight", [](auto w, auto t){ w->out_embed = t; }},
    
    // layer normalization
    {"model.norm.weight", [](auto w, auto t){ w->out_norm_w = t; }},
    
    // attention weights
    {"model.layers.{}.input_layernorm.weight", [](auto w, auto t, int layer){ 
        w->attn_norm_w[layer] = t; 
    }},
    {"model.layers.{}.self_attn.q_proj.weight", [](auto w, auto t, int layer){ 
        w->attn_q_w[layer] = t; 
    }},
    // ... 其他权重映射
};

// 解析层索引，例如从 "model.layers.0.input_layernorm.weight" 提取 0
static int parse_layer_index(const std::string &name) {
    size_t pos = name.find(".layers.");
    if (pos == std::string::npos) return -1;
    pos += 8;  // ".layers."的长度
    size_t end = name.find('.', pos);
    if (end == std::string::npos) return -1;
    return std::stoi(name.substr(pos, end - pos));
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
    
    // 解析层索引
    int layer = parse_layer_index(weight_name);
    
    // 根据名称映射到权重字段
    // 这里简化实现，实际需要完整映射表
    if (weight_name == "model.embed_tokens.weight") {
        weights->in_embed = tensor;
        return 0;
    }
    else if (weight_name.find("model.layers.") != std::string::npos) {
        if (layer < 0) return -1;
        
        if (weight_name.find("input_layernorm.weight") != std::string::npos) {
            weights->attn_norm_w[layer] = tensor;
            return 0;
        }
        // ... 处理其他层权重
    }
    
    return -2;  // 未找到匹配
}

// 重置KV-Cache
__export void llaisysQwen2ResetCache(struct LlaisysQwen2Model *model) {
    if (!model) return;
    auto impl = reinterpret_cast<Qwen2ModelImpl*>(model);
    impl->current_pos = 0;
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

// ========== 核心推理函数（占位符） ==========

// ========== 继续实现 qwen2.cpp ==========

// 前向传播（返回logits）
__export llaisysTensor_t llaisysQwen2Forward(
    struct LlaisysQwen2Model *model, 
    const int64_t *token_ids, 
    size_t ntoken, 
    size_t past_len
) {
    if (!model || !token_ids || ntoken == 0) return nullptr;
    
    auto impl = reinterpret_cast<Qwen2ModelImpl*>(model);
    
    // ========== 1. 创建输入embedding ==========
    // token_ids: [ntoken] -> embedding: [ntoken, hs]
    std::vector<size_t> emb_shape = {ntoken, impl->meta.hs};
    tensor_t input = tensor_create(
        emb_shape.data(), emb_shape.size(), 
        impl->meta.dtype, impl->device
    );
    
    // embedding 操作
    ops::embedding(input, 
        tensor_create_wrapper(token_ids, {ntoken}, LLAISYS_DTYPE_I64, impl->device),
        impl->weights.in_embed
    );
    
    // ========== 2. 逐层处理transformer ==========
    tensor_t hidden = input;  // 当前隐藏状态
    
    for (size_t layer = 0; layer < impl->meta.nlayer; ++layer) {
        // 2.1 输入RMSNorm
        tensor_t norm_hidden = tensor_create_like(hidden);
        ops::rms_norm(norm_hidden, hidden, 
            impl->weights.attn_norm_w[layer], 
            impl->meta.epsilon
        );
        
        // 2.2 计算Q、K、V投影
        size_t nhead = impl->meta.nh;
        size_t nkvhead = impl->meta.nkvh;
        size_t head_dim = impl->meta.dh;
        
        // Q: [ntoken, nhead, head_dim]
        std::vector<size_t> q_shape = {ntoken, nhead, head_dim};
        tensor_t q = tensor_create(q_shape.data(), q_shape.size(), 
            impl->meta.dtype, impl->device);
        
        // K, V: [ntoken, nkvhead, head_dim]
        std::vector<size_t> kv_shape = {ntoken, nkvhead, head_dim};
        tensor_t k = tensor_create(kv_shape.data(), kv_shape.size(), 
            impl->meta.dtype, impl->device);
        tensor_t v = tensor_create(kv_shape.data(), kv_shape.size(), 
            impl->meta.dtype, impl->device);
        
        // 线性投影计算Q
        ops::linear(q, norm_hidden, 
            impl->weights.attn_q_w[layer], 
            impl->weights.attn_q_b ? impl->weights.attn_q_b[layer] : nullptr
        );
        
        // 线性投影计算K
        ops::linear(k, norm_hidden, 
            impl->weights.attn_k_w[layer], 
            impl->weights.attn_k_b ? impl->weights.attn_k_b[layer] : nullptr
        );
        
        // 线性投影计算V
        ops::linear(v, norm_hidden, 
            impl->weights.attn_v_w[layer], 
            impl->weights.attn_v_b ? impl->weights.attn_v_b[layer] : nullptr
        );
        
        // 2.3 应用RoPE位置编码
        // 创建位置ID数组 [0, 1, ..., ntoken-1] + past_len
        std::vector<int64_t> pos_ids_vec(ntoken);
        for (size_t i = 0; i < ntoken; ++i) {
            pos_ids_vec[i] = static_cast<int64_t>(past_len + i);
        }
        tensor_t pos_ids = tensor_create_wrapper(
            pos_ids_vec.data(), 
            {ntoken}, 
            LLAISYS_DTYPE_I64, 
            impl->device
        );
        
        // 对Q和K应用RoPE
        ops::rope(q, q, pos_ids, impl->meta.theta);
        ops::rope(k, k, pos_ids, impl->meta.theta);
        
        // 2.4 处理KV-Cache（如果启用了缓存）
        tensor_t k_to_use = k;
        tensor_t v_to_use = v;
        
        if (past_len > 0 && impl->k_cache[layer] && impl->v_cache[layer]) {
            // 需要将新的k,v追加到缓存中
            // 这里简化实现：假设缓存已经正确管理
            // 实际需要实现缓存追加逻辑
            
            // 合并缓存的k和当前的k
            std::vector<size_t> cache_shape = {past_len + ntoken, nkvhead, head_dim};
            tensor_t k_cache_new = tensor_create(
                cache_shape.data(), cache_shape.size(),
                impl->meta.dtype, impl->device
            );
            
            // 将旧缓存和新的k合并（简化，实际需要更高效的内存管理）
            tensor_concat(k_cache_new, impl->k_cache[layer], k, 0); // 在第0维拼接
            
            // 更新缓存
            if (impl->k_cache[layer]) tensor_release(impl->k_cache[layer]);
            impl->k_cache[layer] = k_cache_new;
            
            // 对v执行相同操作
            tensor_t v_cache_new = tensor_create(
                cache_shape.data(), cache_shape.size(),
                impl->meta.dtype, impl->device
            );
            tensor_concat(v_cache_new, impl->v_cache[layer], v, 0);
            
            if (impl->v_cache[layer]) tensor_release(impl->v_cache[layer]);
            impl->v_cache[layer] = v_cache_new;
            
            // 现在使用完整缓存
            k_to_use = impl->k_cache[layer];
            v_to_use = impl->v_cache[layer];
        } else if (impl->k_cache[layer] == nullptr) {
            // 第一次初始化缓存
            impl->k_cache[layer] = k;
            impl->v_cache[layer] = v;
            tensor_retain(k);
            tensor_retain(v);
        }
        
        // 2.5 自注意力计算
        // attn_output: [ntoken, nhead, head_dim]
        tensor_t attn_output = tensor_create_like(q);
        
        // 计算缩放因子
        float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
        
        ops::self_attention(attn_output, q, k_to_use, v_to_use, scale);
        
        // 2.6 注意力输出投影
        // output_proj: [ntoken, hs]
        std::vector<size_t> proj_shape = {ntoken, impl->meta.hs};
        tensor_t output_proj = tensor_create(
            proj_shape.data(), proj_shape.size(),
            impl->meta.dtype, impl->device
        );
        
        ops::linear(output_proj, attn_output,
            impl->weights.attn_o_w[layer],
            nullptr
        );
        
        // 2.7 残差连接
        // hidden = hidden + output_proj
        tensor_t hidden_new = tensor_create_like(hidden);
        ops::add(hidden_new, hidden, output_proj);
        
        // 2.8 清理中间张量
        tensor_release(norm_hidden);
        tensor_release(q);
        tensor_release(k);
        tensor_release(v);
        tensor_release(pos_ids);
        tensor_release(attn_output);
        tensor_release(output_proj);
        tensor_release(hidden);
        
        hidden = hidden_new;
        
        // ========== 3. MLP部分 ==========
        // 3.1 MLP的RMSNorm
        tensor_t mlp_norm_hidden = tensor_create_like(hidden);
        ops::rms_norm(mlp_norm_hidden, hidden,
            impl->weights.mlp_norm_w[layer],
            impl->meta.epsilon
        );
        
        // 3.2 SwiGLU激活的前馈网络
        // gate_proj: [ntoken, intermediate_size]
        std::vector<size_t> gate_shape = {ntoken, impl->meta.di};
        tensor_t gate = tensor_create(
            gate_shape.data(), gate_shape.size(),
            impl->meta.dtype, impl->device
        );
        
        // up_proj: [ntoken, intermediate_size]
        tensor_t up = tensor_create_like(gate);
        
        ops::linear(gate, mlp_norm_hidden,
            impl->weights.mlp_gate_w[layer],
            nullptr
        );
        
        ops::linear(up, mlp_norm_hidden,
            impl->weights.mlp_up_w[layer],
            nullptr
        );
        
        // 3.3 SwiGLU激活
        tensor_t swiglu_out = tensor_create_like(gate);
        ops::swiglu(swiglu_out, gate, up);
        
        // 3.4 下投影回隐藏维度
        tensor_t mlp_out = tensor_create_like(hidden);
        ops::linear(mlp_out, swiglu_out,
            impl->weights.mlp_down_w[layer],
            nullptr
        );
        
        // 3.5 残差连接
        tensor_t final_hidden = tensor_create_like(hidden);
        ops::add(final_hidden, hidden, mlp_out);
        
        // 3.6 清理MLP中间张量
        tensor_release(mlp_norm_hidden);
        tensor_release(gate);
        tensor_release(up);
        tensor_release(swiglu_out);
        tensor_release(mlp_out);
        tensor_release(hidden);
        
        hidden = final_hidden;
    }
    
    // ========== 4. 最终层归一化 ==========
    tensor_t final_norm = tensor_create_like(hidden);
    ops::rms_norm(final_norm, hidden,
        impl->weights.out_norm_w,
        impl->meta.epsilon
    );
    
    // ========== 5. 输出logits ==========
    // logits: [ntoken, vocab_size]
    std::vector<size_t> logits_shape = {ntoken, impl->meta.voc};
    tensor_t logits = tensor_create(
        logits_shape.data(), logits_shape.size(),
        impl->meta.dtype, impl->device
    );
    
    // 线性投影到词表大小
    ops::linear(logits, final_norm,
        impl->weights.out_embed,
        nullptr
    );
    
    // ========== 6. 清理资源 ==========
    tensor_release(input);
    tensor_release(hidden);
    tensor_release(final_norm);
    
    // 更新缓存位置
    impl->current_pos += ntoken;
    
    return logits;
}

// ========== 辅助函数实现 ==========

// 创建tensor的包装函数（简化版）
static tensor_t tensor_create_wrapper(
    const void* data, 
    const std::vector<size_t>& shape, 
    llaisysDataType_t dtype,
    llaisysDeviceType_t device
) {
    tensor_t tensor = tensor_create(
        shape.data(), shape.size(), dtype, device
    );
    if (tensor && data) {
        // 将数据加载到张量
        tensor_load(tensor, data);
    }
    return tensor;
}

// 张量拼接辅助函数（简化实现）
static void tensor_concat(tensor_t dst, tensor_t src1, tensor_t src2, size_t dim) {
    // 这里简化实现，实际需要处理不同形状和内存布局
    // 假设所有张量都是连续的，且dim=0
    
    // 获取源张量的元素数量和字节大小
    size_t numel1 = src1->numel();
    size_t numel2 = src2->numel();
    size_t elem_size = src1->elementSize();
    
    // 获取数据指针
    std::byte* dst_data = dst->data();
    const std::byte* src1_data = src1->data();
    const std::byte* src2_data = src2->data();
    
    // 复制数据
    std::memcpy(dst_data, src1_data, numel1 * elem_size);
    std::memcpy(dst_data + numel1 * elem_size, src2_data, numel2 * elem_size);
}

// 创建形状相同的张量
static tensor_t tensor_create_like(tensor_t src) {
    std::vector<size_t> shape(src->shape(), src->shape() + src->ndim());
    return tensor_create(
        shape.data(), shape.size(),
        src->dtype(), src->deviceType()
    );
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
    tensor_t logits = llaisysQwen2Forward(model, token_ids, ntoken, past_len);
    if (!logits) return -1;
    
    // 取最后一个token的logits
    // 提取最后一个token的logits向量
    std::vector<size_t> last_token_shape = {1, impl->meta.voc};
    tensor_t last_token_logits = tensor_create(
        last_token_shape.data(), last_token_shape.size(),
        impl->meta.dtype, impl->device
    );
    
    // 这里需要实现张量切片操作来获取最后一个token
    // 简化：假设ntoken=1，直接使用整个logits
    tensor_t slice_logits;
    if (ntoken == 1) {
        slice_logits = logits;
        tensor_retain(slice_logits);
    } else {
        // 需要实现切片：logits[ntoken-1:ntoken, :]
        // 这里简化，实际需要调用tensor_slice函数
        slice_logits = logits;
        tensor_retain(slice_logits);
    }
    
    // 对logits进行argmax得到token
    // 创建输出张量
    tensor_t max_idx = tensor_create(
        last_token_shape.data(), last_token_shape.size(),
        LLAISYS_DTYPE_I64, impl->device
    );
    
    tensor_t max_val = tensor_create(
        last_token_shape.data(), last_token_shape.size(),
        impl->meta.dtype, impl->device
    );
    
    // 调用argmax算子
    ops::argmax(max_idx, max_val, slice_logits);
    
    // 获取结果
    int64_t* idx_data = reinterpret_cast<int64_t*>(max_idx->data());
    int64_t result = idx_data[0];
    
    // 清理资源
    tensor_release(logits);
    tensor_release(slice_logits);
    tensor_release(max_idx);
    tensor_release(max_val);
    
    return result;
}

} // extern "C"