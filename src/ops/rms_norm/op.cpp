#include "op.hpp"
#include <cmath>
#include "../src/utils.hpp"
#include <stdexcept>

using namespace llaisys;

// float
void rms_norm_float(float* out, const float* in, const float* weight, size_t M, size_t d, float eps) {
    for (size_t i = 0; i < M; i++) {
        // 计算当前行的平方和
        float sum_sq = 0.0f;
        for (size_t j = 0; j < d; j++) {
            float val = in[i * d + j];
            sum_sq += val * val;
        }
        
        // 计算RMS：sqrt(mean(x^2) + eps)
        float rms = std::sqrt(sum_sq / static_cast<float>(d) + eps);
        
        // 归一化并乘以权重
        for (size_t j = 0; j < d; j++) {
            float norm_val = in[i * d + j] / rms;
            out[i * d + j] = weight[j] * norm_val;
        }
    }
}

// fp16
void rms_norm_fp16(fp16_t* out, const fp16_t* in, const fp16_t* weight, size_t M, size_t d, float eps){
    for (size_t i = 0; i < M; i++) {
        // 计算当前行的平方和
        float sum_sq = 0.0f;
        for (size_t j = 0; j < d; j++) {
            float val = utils::cast<float>(in[i * d + j]);
            sum_sq += val * val;
        }
        
        // 计算RMS：sqrt(mean(x^2) + eps)
        float rms = std::sqrt(sum_sq / static_cast<float>(d) + eps);
        
        // 归一化并乘以权重
        for (size_t j = 0; j < d; j++) {
            float in_val = utils::cast<float>(in[i * d + j]);
            float weight_val = utils::cast<float>(weight[j]);
            float norm_val = in_val / rms;
            out[i * d + j] = utils::cast<fp16_t>(weight_val * norm_val);
        }
    }
}

// bf16
void rms_norm_bf16(bf16_t* out, const bf16_t* in, const bf16_t* weight,size_t M, size_t d, float eps) {
    for (size_t i = 0; i < M; i++) {
        // 计算当前行的平方和
        float sum_sq = 0.0f;
        for (size_t j = 0; j < d; j++) {
            float val = utils::cast<float>(in[i * d + j]);
            sum_sq += val * val;
        }

        // 计算RMS：sqrt(mean(x^2) + eps)
        float rms = std::sqrt(sum_sq / static_cast<float>(d) + eps);

        // 归一化并乘以权重
        for (size_t j = 0; j < d; j++) {
            float in_val = utils::cast<float>(in[i * d + j]);
            float weight_val = utils::cast<float>(weight[j]);
            float norm_val = in_val / rms;
            out[i * d + j] = utils::cast<bf16_t>(weight_val * norm_val);
        }
    }
}//匿名命名空间

namespace llaisys::ops {

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    //  获取形状信息
    // 输入和输出：2D
    // 权重：1D
    /*out：输出 𝑌。你暂时可以假设输出是一个2D连续张量，不涉及广播。
    • input：输入 𝑋。你暂时可以假设输入是一个2D连续张量，不涉及广播。标准化沿输入张量的最后一个维度（即每一行，长度为 𝑑 ）执行。
    • weight：权重 𝑊 。1D张量，与输入张量的一行长度相同。
    • eps：小值 𝜖 以避免除以零。*/
    
    if (in->ndim() != 2) {
        throw std::invalid_argument("RMSNorm: 输入类型错误");
    }
    
    if (weight->ndim() != 1) {
        throw std::invalid_argument("RMSNorm: 权重类型错误");
    }
    
    size_t M = in->shape()[0];  // 行数
    size_t d = in->shape()[1];  // 每行元素数
    
    // 获得类型
    auto dtype = in->dtype();
    
    // 2获取数据指针
    std::byte* out_data = out->data();
    const std::byte* in_data = in->data();
    const std::byte* weight_data = weight->data();
    
    // 根据数据类型调用对应的实现
    switch (dtype) {
        case LLAISYS_DTYPE_F32: {
            auto* out_f = reinterpret_cast<float*>(out_data);
            const auto* in_f = reinterpret_cast<const float*>(in_data);
            const auto* weight_f = reinterpret_cast<const float*>(weight_data);
            rms_norm_float(out_f, in_f, weight_f, M, d, eps);
            break;
        }
        
        case LLAISYS_DTYPE_F16: {
            auto* out_h = reinterpret_cast<fp16_t*>(out_data);
            const auto* in_h = reinterpret_cast<const fp16_t*>(in_data);
            const auto* weight_h = reinterpret_cast<const fp16_t*>(weight_data);
            rms_norm_fp16(out_h, in_h, weight_h, M, d, eps);
            break;
        }
        
        case LLAISYS_DTYPE_BF16: {
            auto* out_b = reinterpret_cast<bf16_t*>(out_data);
            const auto* in_b = reinterpret_cast<const bf16_t*>(in_data);
            const auto* weight_b = reinterpret_cast<const bf16_t*>(weight_data);
            rms_norm_bf16(out_b, in_b, weight_b, M, d, eps);
            break;
        }
        
        default:
            throw std::invalid_argument("RMSNorm: 不支持的数据类型");
    }
}    

} // namespace llaisys::ops
