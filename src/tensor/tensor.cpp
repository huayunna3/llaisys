#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    //检查张量形状和步长，判断在内存是否连续
    //元素是按照"逻辑行"的顺序连续存储的
    // shape: [2 ,3, 4] strides: [12, 4, 1]
    const std::vector<size_t> &shape = _meta.shape;
    const std::vector<ptrdiff_t> &strides = _meta.strides;
    size_t ndim = shape.size();

    if (shape[ndim-1] == 0) {
            return true; // 空张量被视为连续的
        }

    if (strides[ndim-1] != 1) {
        return false;
    }
    size_t expected_stride = 1;
    for (size_t i = ndim-1; i > 0; i--) {
        expected_stride = shape[i] * expected_stride ;
        if (strides[i-1] != static_cast<ptrdiff_t>(expected_stride)) {
            return false;
        }
        
    }
    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    //创建一个新张量，改变原始张量维度的顺序。
    //转置可以通过这个函数实现，而无需移动数据。
    /*
    a
    tensor([[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9]])

    a.stride()
        (5, 1)
    a.permute(1,0)
    tensor([[0, 5],
            [1, 6],
            [2, 7],
            [3, 8],
            [4, 9]])

    a.stride()
        (1, 5)
    */
    size_t ndim = _meta.shape.size();
    if (ndim != order.size()){
        throw std::invalid_argument("Tensor::permute: 维度不匹配,重输");
    }
    std::vector<size_t> new_shape(ndim);
    std::vector<ptrdiff_t> new_strides(ndim);

    for (size_t i=0;i<_meta.strides.size();i++){
        new_shape[i] = _meta.shape[order[i]];
        new_strides[i]=_meta.strides[order[i]];
    }
    // 创建新的元数据
    TensorMeta new_meta{
            _meta.dtype,      // 数据类型不变
            new_shape,            // 新形状
            new_strides       // 新步长
        };
    //返回新张量
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage,_offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    //检查空张量
    if (this->numel() == 0) {
        throw std::invalid_argument("Tensor::view:空的，退回");
    }
    //连续性检查
    if (!this->isContiguous()) {
        throw std::invalid_argument("Tensor::view: 不是连续的我不会算");
    }
    //检查元素数量是否匹配
    size_t new_numel = 1;
    for (long unsigned int i=0; i < shape.size(); i++) {
        new_numel *= shape[i];
    }
    size_t old_numel = this->numel();
    if (new_numel != old_numel) {
        throw std::invalid_argument("Tensor::view: 元素数量不匹配，这事办不了");
    }
    //计算新的步长
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);

    strides[ndim_-1] = 1;//最后一维步长为1
    for (size_t i = ndim_-1; i > 0; i--) {
        strides[i-1] = strides[i] * shape[i];
    }

    // 创建新meta
 TensorMeta new_meta{
        _meta.dtype,      // 数据类型不变
        shape,            // 新形状
        strides       // 新步长
    };

    // 返回新的张量
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage,_offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    // 创建一个新张量，沿给定维度，start（包含）和end（不包含）索引
    // 对原始张量进行切片操作。
    if (dim >= _meta.shape.size()) {
        throw std::invalid_argument("Tensor::slice: 维度超出范围");
    }
    if (start >= end || end > _meta.shape[dim]) {
        throw std::invalid_argument("Tensor::slice: 切片索引无效");
    }
    //计算新形状和偏移
    const auto slice_size = end - start;
    size_t new_offset = _offset + start * _meta.strides[dim] * this->elementSize();
    std::vector<size_t> new_shape = _meta.shape;
    new_shape[dim] = slice_size;
    // 创建新的元数据
    TensorMeta new_meta{
        _meta.dtype,      // 数据类型不变
        new_shape,            // 新形状
        _meta.strides       // 步长保持不变
    };
    //返回新张量
    return std::shared_ptr<Tensor>(new Tensor(new_meta, _storage, new_offset));
}

void Tensor::load(const void *src_) {
    // 检查源指针
    if (src_ == nullptr) {
        throw std::invalid_argument("Tensor::load: 空的，退回");
    }
    
    // 获取需要复制的字节数
    size_t num_elements = this->numel();
    size_t elem_size = this->elementSize();
    size_t total_bytes = num_elements * elem_size;
    
    // 检查是否有数据需要复制
    if (total_bytes == 0) {
        return;  // 空张量，无需加载
    }
    
    // 设置当前设备上下文
    core::context().setDevice(this->deviceType(), this->deviceId());
    
    // 获取运行时API
    auto* runtime_api = core::context().runtime().api();
    
    // 执行内存复制
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        // CPU到CPU的复制
        std::memcpy(this->data(), src_, total_bytes);
    } else {
        // 主机到设备的复制
        runtime_api->memcpy_sync(
            this->data(),        // 目标设备内存
            src_,               // 源主机内存
            total_bytes,        // 字节数
            LLAISYS_MEMCPY_H2D  // 复制方向
        );
    }
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
