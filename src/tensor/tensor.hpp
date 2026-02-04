#pragma once
#include "../core/llaisys_core.hpp"

#include <vector>
namespace llaisys {
class Tensor;
using tensor_t = std::shared_ptr<Tensor>;

// 张量操作接口
tensor_t tensor_create(const size_t* shape, size_t ndim, 
                      llaisysDataType_t dtype, llaisysDeviceType_t device);
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

// 张量操作函数
void tensor_concat(tensor_t dst, tensor_t src1, tensor_t src2, size_t dim);
tensor_t tensor_slice(tensor_t src, size_t dim, size_t start, size_t end);


struct TensorMeta {
    llaisysDataType_t dtype;
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> strides;
};

class Tensor {
private:
    TensorMeta _meta;
    core::storage_t _storage;
    size_t _offset;
    Tensor(TensorMeta meta, core::storage_t storage, size_t offset = 0);

public:
    static tensor_t create(
        const std::vector<size_t> &shape,
        llaisysDataType_t dtype,
        llaisysDeviceType_t device_type = LLAISYS_DEVICE_CPU,
        int device = 0);
    ~Tensor() = default;
    // Info
    std::byte *data();
    const std::byte *data() const;
    size_t ndim() const;
    const std::vector<size_t> &shape() const;
    const std::vector<ptrdiff_t> &strides() const;
    llaisysDataType_t dtype() const;
    llaisysDeviceType_t deviceType() const;
    int deviceId() const;
    size_t numel() const;
    size_t elementSize() const;

    std::string info() const;
    void debug() const;

    bool isContiguous() const;

    // Meta Transform
    tensor_t permute(const std::vector<size_t> &order) const;
    tensor_t slice(size_t dim, size_t start, size_t end) const;
    tensor_t view(const std::vector<size_t> &shape) const;

    // Load data from host memory
    void load(const void *src);

    // Challenging features
    tensor_t contiguous() const;
    tensor_t reshape(const std::vector<size_t> &shape) const;
    tensor_t to(llaisysDeviceType_t device_type, int device = -1) const;
};

} // namespace llaisys
