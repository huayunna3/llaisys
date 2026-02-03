#include "op.hpp"
#include "../src/utils.hpp"
#include <cmath>
#include <stdexcept>

using namespace llaisys;

namespace llaisys::ops {
// 预计算频率：theta^(-2j/d)
void precompute_freqs(float* freqs, size_t d_half, float theta) {
    float d = static_cast<float>(d_half * 2);  // 总维度d
    for (size_t j = 0; j < d_half; j++) {
        // 计算 θ^(-2j/d)
        float exponent = -2.0f * static_cast<float>(j) / d;
        freqs[j] = std::pow(theta, exponent);
    }
}

// float
void rope_float(float* out, const float* in, const int64_t* pos_ids,size_t seq_len, size_t n_head, size_t d, float theta) {
    size_t d_half = d / 2;
    
    // 预计算频率
    std::vector<float> freqs(d_half);
    precompute_freqs(freqs.data(), d_half, theta);
    
    // 遍历所有位置、所有头
    for (size_t i = 0; i < seq_len; i++) {
        float pos = static_cast<float>(pos_ids[i]);
        
        for (size_t h = 0; h < n_head; h++) {
            size_t base_idx = (i * n_head + h) * d;
            
            // 对每对(a_j, b_j)应用旋转
            for (size_t j = 0; j < d_half; j++) {
                float angle = pos * freqs[j];
                float cos_val = std::cos(angle);
                float sin_val = std::sin(angle);
                
                // 原始a和b
                float a = in[base_idx + j];
                float b = in[base_idx + j + d_half];
                
                // 旋转后的a'和b'
                out[base_idx + j] = a * cos_val - b * sin_val;
                out[base_idx + j + d_half] = b * cos_val + a * sin_val;
            }
        }
    }
}

// fp16
void rope_fp16(fp16_t* out, const fp16_t* in, const int64_t* pos_ids,size_t seq_len, size_t n_head, size_t d, float theta) {
    size_t d_half = d / 2;
    
    // 预计算频率
    std::vector<float> freqs(d_half);
    precompute_freqs(freqs.data(), d_half, theta);
    
    // 遍历所有位置、所有头
    for (size_t i = 0; i < seq_len; i++) {
        float pos = static_cast<float>(pos_ids[i]);
        
        for (size_t h = 0; h < n_head; h++) {
            size_t base_idx = (i * n_head + h) * d;
            
            // 对每对(a_j, b_j)应用旋转
            for (size_t j = 0; j < d_half; j++) {
                float angle = pos * freqs[j];
                float cos_val = std::cos(angle);
                float sin_val = std::sin(angle);
                
                // 原始a和b
                float a = utils::cast<float>(in[base_idx + j]);
                float b = utils::cast<float>(in[base_idx + j + d_half]);
                
                // 旋转后的a'和b'
                float a_rotated = a * cos_val - b * sin_val;
                float b_rotated = b * cos_val + a * sin_val;
                
                out[base_idx + j] = utils::cast<fp16_t>(a_rotated);
                out[base_idx + j + d_half] = utils::cast<fp16_t>(b_rotated);
            }
        }
    }
}

// bf16
void rope_bf16(bf16_t* out, const bf16_t* in, const int64_t* pos_ids,
              size_t seq_len, size_t n_head, size_t d, float theta) {
    size_t d_half = d / 2;
    
    // 预计算频率
    std::vector<float> freqs(d_half);
    precompute_freqs(freqs.data(), d_half, theta);
    
    // 遍历所有位置、所有头
    for (size_t i = 0; i < seq_len; i++) {
        float pos = static_cast<float>(pos_ids[i]);
        
        for (size_t h = 0; h < n_head; h++) {
            size_t base_idx = (i * n_head + h) * d;
            
            // 对每对(a_j, b_j)应用旋转
            for (size_t j = 0; j < d_half; j++) {
                float angle = pos * freqs[j];
                float cos_val = std::cos(angle);
                float sin_val = std::sin(angle);
                
                // 原始a和b
                float a = utils::cast<float>(in[base_idx + j]);
                float b = utils::cast<float>(in[base_idx + j + d_half]);
                
                // 旋转后的a'和b'
                float a_rotated = a * cos_val - b * sin_val;
                float b_rotated = b * cos_val + a * sin_val;
                
                out[base_idx + j] = utils::cast<bf16_t>(a_rotated);
                out[base_idx + j + d_half] = utils::cast<bf16_t>(b_rotated);
            }
        }
    }
}

}// 匿名命名空间
namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    /*
    out：结果q或k张量。形状应该是 [seqlen, nhead, d] 或 [seqlen, nkvhead, d]。你暂时可以假设张量是连续的。
    in：原始q或k张量。形状应该是 [seqlen, nhead, d] 或 [seqlen, nkvhead, d]。你暂时可以假设张量是连续的。
    pos_ids：输入序列中每个token的位置id（整个上下文中的索引）。形状应该是 [seqlen,]，dtype应该是int64。
    theta：频率向量的基值。
*/
    // 合法性验证
    if (!out || !in || !pos_ids) {
        throw std::invalid_argument("RoPE: 输入张量不能为空");
    }
    if (in->ndim() != 3) {
        throw std::invalid_argument("RoPE: 输入必须是3维张量 [seqlen, nhead, d]");
    }
    if (out->ndim() != 3) {
        throw std::invalid_argument("RoPE: 输出必须是3维张量 [seqlen, nhead, d]");
    }
    if (pos_ids->ndim() != 1) {
        throw std::invalid_argument("RoPE: pos_ids必须是1维张量 [seqlen]");
    }
    
    // 获取形状信息
    size_t seq_len = in->shape()[0];
    size_t n_head = in->shape()[1];
    size_t d = in->shape()[2];
    
    auto dtype = in->dtype();
 
    // 获取数据指针
    std::byte* out_data = out->data();
    const std::byte* in_data = in->data();
    const std::byte* pos_data = pos_ids->data();
    const int64_t* pos_ids_ptr = reinterpret_cast<const int64_t*>(pos_data);
    
    // 根据数据类型调用对应的实现
    switch (dtype) {
        case LLAISYS_DTYPE_F32: {
            auto* out_f = reinterpret_cast<float*>(out_data);
            const auto* in_f = reinterpret_cast<const float*>(in_data);
            rope_float(out_f, in_f, pos_ids_ptr, seq_len, n_head, d, theta);
            break;
        }
        
        case LLAISYS_DTYPE_F16: {
            auto* out_h = reinterpret_cast<fp16_t*>(out_data);
            const auto* in_h = reinterpret_cast<const fp16_t*>(in_data);
            rope_fp16(out_h, in_h, pos_ids_ptr, seq_len, n_head, d, theta);
            break;
        }
        
        case LLAISYS_DTYPE_BF16: {
            auto* out_b = reinterpret_cast<bf16_t*>(out_data);
            const auto* in_b = reinterpret_cast<const bf16_t*>(in_data);
            rope_bf16(out_b, in_b, pos_ids_ptr, seq_len, n_head, d, theta);
            break;
        }
        
        default:
            throw std::invalid_argument("RoPE: 不支持的数据类型");
    }
}

} // namespace llaisys::ops