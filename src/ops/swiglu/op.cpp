#include "op.hpp"

#include "../src/tensor/tensor.hpp"
#include "../src/utils.hpp"
#include <cmath>
#include <stdexcept>

using namespace llaisys;

namespace {

// float
void swiglu_float(float* out, const float* gate, const float* up, size_t total_elements) {
    for (size_t i = 0; i < total_elements; i++) {
        float g = gate[i];
        float swish_g = g / (1.0f + std::exp(-g));  // swish(g) = g * sigmoid(g) = g / (1 + exp(-g))
        out[i] = up[i] * swish_g;
    }
}

// fp16
void swiglu_fp16(fp16_t* out, const fp16_t* gate, const fp16_t* up, size_t total_elements) {
    for (size_t i = 0; i < total_elements; i++) {
        float g = utils::cast<float>(gate[i]);
        float u = utils::cast<float>(up[i]);
        float swish_g = g / (1.0f + std::exp(-g));
        out[i] = utils::cast<fp16_t>(u * swish_g);
    }
}

// bf16
void swiglu_bf16(bf16_t* out, const bf16_t* gate, const bf16_t* up, size_t total_elements) {
    for (size_t i = 0; i < total_elements; i++) {
        float g = utils::cast<float>(gate[i]);
        float u = utils::cast<float>(up[i]);
        float swish_g = g / (1.0f + std::exp(-g));
        out[i] = utils::cast<bf16_t>(u * swish_g);
    }
}

} // 匿名命名空间

namespace llaisys::ops {

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    // 合法性验证
    if (!out || !gate || !up) {
        throw std::invalid_argument("SwiGLU: 输入张量不能为空");
    }
    
    // 获取形状信息
    size_t seq_len = gate->shape()[0];
    size_t intermediate_size = gate->shape()[1];
    size_t total_elements = seq_len * intermediate_size;
    
    // 获取数据指针
    std::byte* out_data = out->data();
    const std::byte* gate_data = gate->data();
    const std::byte* up_data = up->data();
    
    // 根据数据类型调用对应的实现
    auto dtype = gate->dtype();
    switch (dtype) {
        case LLAISYS_DTYPE_F32: {
            auto* out_f = reinterpret_cast<float*>(out_data);
            const auto* gate_f = reinterpret_cast<const float*>(gate_data);
            const auto* up_f = reinterpret_cast<const float*>(up_data);
            swiglu_float(out_f, gate_f, up_f, total_elements);
            break;
        }
        
        case LLAISYS_DTYPE_F16: {
            auto* out_h = reinterpret_cast<fp16_t*>(out_data);
            const auto* gate_h = reinterpret_cast<const fp16_t*>(gate_data);
            const auto* up_h = reinterpret_cast<const fp16_t*>(up_data);
            swiglu_fp16(out_h, gate_h, up_h, total_elements);
            break;
        }
        
        case LLAISYS_DTYPE_BF16: {
            auto* out_b = reinterpret_cast<bf16_t*>(out_data);
            const auto* gate_b = reinterpret_cast<const bf16_t*>(gate_data);
            const auto* up_b = reinterpret_cast<const bf16_t*>(up_data);
            swiglu_bf16(out_b, gate_b, up_b, total_elements);
            break;
        }
        
        default:
            throw std::invalid_argument("SwiGLU: 不支持的数据类型");
    }
}

} // namespace llaisys::ops