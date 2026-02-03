#include "op.hpp"
#include <cstring>

// 模板实现 - 注意这里需要和头文件中的声明匹配
template<typename T>
void embedding_impl(T* out, const int64_t* index, const T* weight, 
                    int64_t embedding_dim, int64_t index_size, int64_t vocab_size) {
    for (int64_t i = 0; i < index_size; i++) {
        int64_t token_id = index[i];
        if (token_id < 0 || token_id >= vocab_size) {
            throw std::runtime_error("embedding: index out of range");
        }
        const T* src_row = weight + token_id * embedding_dim;
        T* dst_row = out + i * embedding_dim;
        std::memcpy(dst_row, src_row, embedding_dim * sizeof(T));
    }
}
namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // 词嵌入
    // 从weight（2-D）中复制index（1-D）中的行到output（2-D）。index必须是Int64类型
    if (weight->ndim() != 2) {
        throw std::runtime_error("embedding: 'weight' tensor must be 2-D.");
    }
    if (index->dtype() != LLAISYS_DTYPE_I64) {
        throw std::runtime_error("embedding: 'index' tensor must be of type Int64.");
    }
    if (index->ndim() != 1) {
        throw std::runtime_error("embedding: 'index' tensor must be 1-D.");
    }
    
    // 获取维度信息 - 通过shape()方法获取维度数组
    const auto& weight_shape = weight->shape();
    const auto& index_shape = index->shape();
    const auto& out_shape = out->shape();
    
    int64_t index_size = static_cast<int64_t>(index_shape[0]);
    int64_t embedding_dim = static_cast<int64_t>(weight_shape[1]);
    int64_t vocab_size = static_cast<int64_t>(weight_shape[0]);

    // 检查输出形状
    if (out->ndim() != 2 || out_shape.size() != 2) {
        throw std::runtime_error("embedding: 'out' tensor must be 2-D.");
    }
    if (static_cast<int64_t>(out_shape[0]) != index_size || 
        static_cast<int64_t>(out_shape[1]) != embedding_dim) {
        throw std::runtime_error("Output tensor shape mismatch");
    }
    
    // 检查索引范围
    auto* index_data = reinterpret_cast<const int64_t*>(index->data());
    for (int64_t i = 0; i < index_size; i++) {
        int64_t token_id = index_data[i];
        if (token_id < 0 || token_id >= vocab_size) {
            throw std::runtime_error("embedding: index out of range");
        }
    }
    
    // 根据数据类型调用不同的实现
    switch (out->dtype()) {
    case LLAISYS_DTYPE_F32:
        embedding_impl<float>(
            reinterpret_cast<float*>(out->data()),
            reinterpret_cast<const int64_t*>(index->data()),
            reinterpret_cast<const float*>(weight->data()),
            embedding_dim, index_size, vocab_size);
        break;
    case LLAISYS_DTYPE_F16:
        embedding_impl<llaisys::fp16_t>(
            reinterpret_cast<llaisys::fp16_t*>(out->data()),
            reinterpret_cast<const int64_t*>(index->data()),
            reinterpret_cast<const llaisys::fp16_t*>(weight->data()),
            embedding_dim, index_size, vocab_size);
        break;
    case LLAISYS_DTYPE_BF16:
        embedding_impl<llaisys::bf16_t>(
            reinterpret_cast<llaisys::bf16_t*>(out->data()),
            reinterpret_cast<const int64_t*>(index->data()),
            reinterpret_cast<const llaisys::bf16_t*>(weight->data()),
            embedding_dim, index_size, vocab_size);
        break;
    default:
        throw std::runtime_error("Unsupported data type in embedding operation");
    }
}
} // namespace llaisys::ops