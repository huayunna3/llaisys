#include "op.hpp"
#include <cstring>
#include <stdexcept>

namespace llaisys::ops {

namespace {

// 模板：执行矩阵乘法和添加偏置
template <typename T>
void linear_impl(T* out, const T* in, const T* weight, const T* bias,size_t M, size_t K, size_t N) {
    // 检查偏置是否存在
    bool has_bias = bias;
    
    if (has_bias) {
        // 有偏置的情况：先初始化输出为偏置的广播，然后累加矩阵乘法
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                out[i * N + j] = bias[j];  
            }
        }
    } else {
        // 没有偏置：先初始化为0
        for (size_t i = 0; i < M * N; i++) {
            out[i] = static_cast<T>(0);
        }
    }
    
    // 矩阵乘法：Y = X * W^T
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            T sum = has_bias ? out[i * N + j] : static_cast<T>(0);
            for (size_t k = 0; k < K; k++) {
                // X[i, k] * W[j, k] (W[j, k] 是 W^T[k, j])
                sum += in[i * K + k] * weight[j * K + k];
            }
            out[i * N + j] = sum;
        }
    }
}

// fp16_t
void linear_impl_fp16(fp16_t* out, const fp16_t* in, const fp16_t* weight, const fp16_t* bias,size_t M, size_t K, size_t N) {
    bool has_bias = bias;
    
    if (has_bias) {
        // 有偏置的情况
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                float bias_val = utils::cast<float>(bias[j]);
                out[i * N + j] = utils::cast<fp16_t>(bias_val);// 广播偏置
            }
        }
    } else {
        // 没有偏置：初始化为0
        for (size_t i = 0; i < M * N; i++) {
            out[i] = utils::cast<fp16_t>(0.0f);
        }
    }
    
    // 矩阵乘法：Y = X * W^T
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = has_bias ? utils::cast<float>(out[i * N + j]) : 0.0f;
            for (size_t k = 0; k < K; k++) {
                float in_val = utils::cast<float>(in[i * K + k]);
                float weight_val = utils::cast<float>(weight[j * K + k]);
                sum += in_val * weight_val;
            }
            out[i * N + j] = utils::cast<fp16_t>(sum);
        }
    }
}

// bf16_t
void linear_impl_bf16(bf16_t* out, const bf16_t* in, const bf16_t* weight, const bf16_t* bias,size_t M, size_t K, size_t N) {
    bool has_bias = bias;
    
    if (has_bias) {
        // 有偏置的情况
        for (size_t i = 0; i < M; i++) {
            for (size_t j = 0; j < N; j++) {
                float bias_val = utils::cast<float>(bias[j]);
                out[i * N + j] = utils::cast<bf16_t>(bias_val);// 广播偏置
            }
        }
    } else {
        // 没有偏置：初始化为0
        for (size_t i = 0; i < M * N; i++) {
            out[i] = utils::cast<bf16_t>(0.0f);
        }
    }
    
    // 矩阵乘法：Y = X * W^T
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = has_bias ? utils::cast<float>(out[i * N + j]) : 0.0f;
            for (size_t k = 0; k < K; k++) {
                float in_val = utils::cast<float>(in[i * K + k]);
                float weight_val = utils::cast<float>(weight[j * K + k]);
                sum += in_val * weight_val;
            }
            out[i * N + j] = utils::cast<bf16_t>(sum);
        }
    }
}

} // 匿名命名空间

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // 获取张量形状
    size_t M = in->shape()[0];  // batch size
    size_t K = in->shape()[1];  // input features
    size_t N = weight->shape()[0]; // output features
    
    // 获取数据类型
    auto dtype = in->dtype();
    
    // 根据数据类型调用不同的实现
    switch (dtype) {
        case LLAISYS_DTYPE_F32: {
            auto* out_f = reinterpret_cast<float*>(out->data());
            const auto* in_f = reinterpret_cast<const float*>(in->data());
            const auto* weight_f = reinterpret_cast<const float*>(weight->data());
            const float* bias_f = bias ? reinterpret_cast<const float*>(bias->data()) : nullptr;
            
            linear_impl<float>(out_f, in_f, weight_f, bias_f, M, K, N);
            break;
        }
        
        case LLAISYS_DTYPE_F16: {
            auto* out_h = reinterpret_cast<fp16_t*>(out->data());
            const auto* in_h = reinterpret_cast<const fp16_t*>(in->data());
            const auto* weight_h = reinterpret_cast<const fp16_t*>(weight->data());
            const fp16_t* bias_h = bias ? reinterpret_cast<const fp16_t*>(bias->data()) : nullptr;
            
            linear_impl_fp16(out_h, in_h, weight_h, bias_h, M, K, N);
            break;
        }
        
        case LLAISYS_DTYPE_BF16: {
            auto* out_b = reinterpret_cast<bf16_t*>(out->data());
            const auto* in_b = reinterpret_cast<const bf16_t*>(in->data());
            const auto* weight_b = reinterpret_cast<const bf16_t*>(weight->data());
            const bf16_t* bias_b = bias ? reinterpret_cast<const bf16_t*>(bias->data()) : nullptr;
            
            linear_impl_bf16(out_b, in_b, weight_b, bias_b, M, K, N);
            break;
        }
        
        default:
            throw std::invalid_argument("Linear: unsupported data type");
    }
}

} // namespace llaisys::ops