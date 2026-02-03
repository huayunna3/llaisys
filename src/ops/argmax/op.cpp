#include "op.hpp"
#include <cstring> 
namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    //算子应该至少支持Float32、Float16和BFloat16数据类型
    // 获取张量vals的最大值及其索引，
    // 并分别存储在max_val和max_idx中。
    // 你暂时可以假设vals是一个1D张量，
    // max_idx和max_val都是包含单个元素的1D张量（这意味着保留了vals的维度）。
    // argmax用于返回输入张量中最大值的索引。
    if (vals->numel() == 0) {
        throw std::runtime_error("argmax: input tensor 'vals' is empty.");
    }
    //读取张量
    size_t num_elements = vals->numel();
    llaisysDataType_t dtype = vals->dtype();
    const auto vals_data = vals->data();
    std::byte* max_value = max_val->data();
    //初始化最大值和索引（设为第一个值
   
    size_t max_index = 0;
    std::memcpy(max_value, vals_data, vals->elementSize());  // 将第一个值复制到max_val

    //遍历张量，找到最大值和索引
    for (size_t i = 1; i < num_elements; ++i) {
        const auto current_val = vals_data + vals->elementSize() * i;
        bool is_greater = false;
        switch (dtype) {
        case LLAISYS_DTYPE_F32: {
            float val1 = *reinterpret_cast<const float *>(current_val);
            float val2 = *reinterpret_cast<const float *>(max_value); 
            is_greater = (val1 > val2);
            break;
        }
        case LLAISYS_DTYPE_F16: {
            fp16_t val1 = *reinterpret_cast<const fp16_t *>(current_val);
            fp16_t val2 = *reinterpret_cast<const fp16_t *>(max_value);
            is_greater = (utils::cast<float>(val1) > utils::cast<float>(val2));
            break;
        }
        case LLAISYS_DTYPE_BF16: {
            bf16_t val1 = *reinterpret_cast<const bf16_t *>(current_val);
            bf16_t val2 = *reinterpret_cast<const bf16_t *>(max_value);
            is_greater = (utils::cast<float>(val1) > utils::cast<float>(val2));
            break;
        }
        default:
            throw std::runtime_error("argmax: unsupported data type.");
        }
        if (is_greater) {
            max_index = i;
            std::memcpy(max_value, current_val, vals->elementSize());
        }
    }

  // 将最大索引写入 max_idx 张量
  size_t* max_idx_ptr = reinterpret_cast<size_t*>(max_idx->data());
  *max_idx_ptr = max_index;
}
} // namespace llaisys::ops