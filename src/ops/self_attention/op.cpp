#include "op.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <vector>

using namespace llaisys;

// Softmax函数
void causal_softmax_row(float* row, size_t total_len, size_t seq_idx, size_t cache_len, float* max_vals = nullptr) {
    // 查找最大值（数值稳定性）
    float max_val = row[0];
    for (size_t j = 1; j < total_len; j++) {
        // 掩码：mask
        if (j <= cache_len + seq_idx) {
            max_val = std::max(max_val, row[j]);
        } else {
            row[j] = -1e9f;  // 最小值，负值
        }
    }
    
    if (max_vals) max_vals[seq_idx] = max_val;
    
    // 计算指数和
    float exp_sum = 0.0f;
    for (size_t j = 0; j < total_len; j++) {
        if (j <= cache_len + seq_idx) {
            float exp_val = std::exp(row[j] - max_val);
            row[j] = exp_val;
            exp_sum += exp_val;
        } else {
            row[j] = 0.0f;
        }
    }
    
    // 归一化
    if (exp_sum > 0.0f) {
        for (size_t j = 0; j < total_len; j++) {
            if (j <= cache_len + seq_idx) {
                row[j] /= exp_sum;
            }
        }
    }
}

// float，支持分组注意力
void self_attention_float(float* attn_val, const float* q, const float* k, const float* v,
                         size_t seq_len, size_t total_len, size_t n_head, size_t nkvhead, size_t d, size_t dv,
                         float scale) {
    size_t cache_len = total_len - seq_len;
    
    // 检查分组注意力配置
    if (n_head % nkvhead != 0) {
        throw std::invalid_argument("SelfAttention: nhead必须是nkvhead的整数倍");
    }
    size_t group_size = n_head / nkvhead;  // 每个键值头对应的查询头数
    
    // 为每个查询头计算注意力
    for (size_t h = 0; h < n_head; h++) {
        // 计算对应的键值头索引
        size_t kv_head_idx = h / group_size;
        
        // 为当前头分配注意力分数矩阵 [seq_len, total_len]
        std::vector<float> scores(seq_len * total_len, 0.0f);
        
        // 计算 Q * K^T
        for (size_t i = 0; i < seq_len; i++) {
            const float* q_row = q + (i * n_head + h) * d;
            float* score_row = scores.data() + i * total_len;
            
            for (size_t j = 0; j < total_len; j++) {
                const float* k_row = k + (j * nkvhead + kv_head_idx) * d;
                float dot = 0.0f;
                
                // 点积计算
                for (size_t m = 0; m < d; m++) {
                    dot += q_row[m] * k_row[m];
                }
                
                score_row[j] = dot * scale;
            }
            
            // mask softmax
            causal_softmax_row(score_row, total_len, i, cache_len);
        }
        
        // 计算加权和: A * V
        for (size_t i = 0; i < seq_len; i++) {
            const float* score_row = scores.data() + i * total_len;
            float* attn_row = attn_val + (i * n_head + h) * dv;
            
            // 初始化输出为0
            for (size_t m = 0; m < dv; m++) {
                attn_row[m] = 0.0f;
            }
            
            // 加权求和
            for (size_t j = 0; j < total_len; j++) {
                const float* v_row = v + (j * nkvhead + kv_head_idx) * dv;
                float weight = score_row[j];
                
                for (size_t m = 0; m < dv; m++) {
                    attn_row[m] += weight * v_row[m];
                }
            }
        }
    }
}

// fp16
void self_attention_fp16(fp16_t* attn_val, const fp16_t* q, const fp16_t* k, const fp16_t* v,
                        size_t seq_len, size_t total_len, size_t n_head, size_t nkvhead, size_t d, size_t dv,
                        float scale) {
    // 转换为float计算
    std::vector<float> q_float(seq_len * n_head * d);
    std::vector<float> k_float(total_len * nkvhead * d);
    std::vector<float> v_float(total_len * nkvhead * dv);
    std::vector<float> attn_float(seq_len * n_head * dv);
    
    // 转换输入数据
    for (size_t i = 0; i < seq_len * n_head * d; i++) {
        q_float[i] = utils::cast<float>(q[i]);
    }
    for (size_t i = 0; i < total_len * nkvhead * d; i++) {
        k_float[i] = utils::cast<float>(k[i]);
    }
    for (size_t i = 0; i < total_len * nkvhead * dv; i++) {
        v_float[i] = utils::cast<float>(v[i]);
    }
    
    // 调用float版本
    self_attention_float(attn_float.data(), q_float.data(), k_float.data(), v_float.data(),
                        seq_len, total_len, n_head, nkvhead, d, dv, scale);
    
    // 转换回fp16
    for (size_t i = 0; i < seq_len * n_head * dv; i++) {
        attn_val[i] = utils::cast<fp16_t>(attn_float[i]);
    }
}

// bf16
void self_attention_bf16(bf16_t* attn_val, const bf16_t* q, const bf16_t* k, const bf16_t* v,
                        size_t seq_len, size_t total_len, size_t n_head, size_t nkvhead, size_t d, size_t dv,
                        float scale) {
    // 转换为float计算
    std::vector<float> q_float(seq_len * n_head * d);
    std::vector<float> k_float(total_len * nkvhead * d);
    std::vector<float> v_float(total_len * nkvhead * dv);
    std::vector<float> attn_float(seq_len * n_head * dv);
    
    // 转换输入数据
    for (size_t i = 0; i < seq_len * n_head * d; i++) {
        q_float[i] = utils::cast<float>(q[i]);
    }
    for (size_t i = 0; i < total_len * nkvhead * d; i++) {
        k_float[i] = utils::cast<float>(k[i]);
    }
    for (size_t i = 0; i < total_len * nkvhead * dv; i++) {
        v_float[i] = utils::cast<float>(v[i]);
    }
    
    // 调用float版本
    self_attention_float(attn_float.data(), q_float.data(), k_float.data(), v_float.data(),
                        seq_len, total_len, n_head, nkvhead, d, dv, scale);
    
    // 转换回bf16
    for (size_t i = 0; i < seq_len * n_head * dv; i++) {
        attn_val[i] = utils::cast<bf16_t>(attn_float[i]);
    }
}
//匿名命名空间

namespace llaisys::ops {

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    //A = Q * K^T * scale
    // Y = causal_softmax(A) * V

    //  验证输入参数
    if (!attn_val || !q || !k || !v) {
        throw std::invalid_argument("SelfAttention: 输入张量不能为空");
    }
    
    //  获取形状信息
    size_t seq_len = q->shape()[0];
    size_t n_head = q->shape()[1];
    size_t d = q->shape()[2];
    
    size_t total_len = k->shape()[0];
    size_t nkvhead = k->shape()[1];
    
    size_t dv = v->shape()[2];
    
    //  获取数据指针
    std::byte* attn_data = attn_val->data();
    const std::byte* q_data = q->data();
    const std::byte* k_data = k->data();
    const std::byte* v_data = v->data();
    
    // 根据数据类型调用对应的实现
    auto dtype = q->dtype();
    switch (dtype) {
        case LLAISYS_DTYPE_F32: {
            auto* attn_f = reinterpret_cast<float*>(attn_data);
            const auto* q_f = reinterpret_cast<const float*>(q_data);
            const auto* k_f = reinterpret_cast<const float*>(k_data);
            const auto* v_f = reinterpret_cast<const float*>(v_data);
            
            self_attention_float(attn_f, q_f, k_f, v_f, seq_len, total_len, n_head, nkvhead, d, dv, scale);
            break;
        }
        
        case LLAISYS_DTYPE_F16: {
            auto* attn_h = reinterpret_cast<fp16_t*>(attn_data);
            const auto* q_h = reinterpret_cast<const fp16_t*>(q_data);
            const auto* k_h = reinterpret_cast<const fp16_t*>(k_data);
            const auto* v_h = reinterpret_cast<const fp16_t*>(v_data);
            
            self_attention_fp16(attn_h, q_h, k_h, v_h, seq_len, total_len, n_head, nkvhead, d, dv, scale);
            break;
        }
        
        case LLAISYS_DTYPE_BF16: {
            auto* attn_b = reinterpret_cast<bf16_t*>(attn_data);
            const auto* q_b = reinterpret_cast<const bf16_t*>(q_data);
            const auto* k_b = reinterpret_cast<const bf16_t*>(k_data);
            const auto* v_b = reinterpret_cast<const bf16_t*>(v_data);
            
            self_attention_bf16(attn_b, q_b, k_b, v_b, seq_len, total_len, n_head, nkvhead, d, dv, scale);
            break;
        }
        
        default:
            throw std::invalid_argument("SelfAttention: 不支持的数据类型");
    }
}

} // namespace llaisys::ops