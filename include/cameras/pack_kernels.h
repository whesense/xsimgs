#ifndef XSIMGS_CAMERA_PACK_KERNELS_H
#define XSIMGS_CAMERA_PACK_KERNELS_H

#include "camera.h"

#include <torch/csrc/autograd/custom_function.h>

#include <toast/utils/tensor_helpers.h>


template <typename CameraType>
at::Tensor pack_camera(const std::vector<at::Tensor>& tensors) {
    check_same_device(tensors);
    check_dtypes(at::kFloat, tensors);
    constexpr auto total_args = (
        CameraType::tensor_dims_base.size() + CameraType::ImplType::tensor_dims.size());
    if (tensors.size() != total_args) {
        throw std::runtime_error(
            "Tensor list size mismatch. Expected " +
            std::to_string(total_args) + ", but found: "
            + std::to_string(tensors.size()));
    }
    check_operand_last_dims(
        std::vector(tensors.begin(), tensors.begin() + CameraType::tensor_dims_base.size()),
        CameraType::tensor_dims_base);
    check_operand_last_dims(
        std::vector(tensors.begin() + CameraType::tensor_dims_base.size(), tensors.end()),
        CameraType::ImplType::tensor_dims);

    auto ptr_tensor = make_ptr_tensor(tensors);
    auto ptr_tensor_ptr = static_cast<const char **>(ptr_tensor.data_ptr());
    uint num_cameras = tensors[0].numel() / tensors[0].size(-1);

    auto out = at::zeros(
        {num_cameras, sizeof(CameraType) / sizeof(float)},
        at::TensorOptions().dtype(at::kFloat).device(ptr_tensor.device())
    );

    const auto data = std::make_tuple(
        static_cast<CameraType *>(out.data_ptr()), ptr_tensor_ptr
    );

    INLINE_KERNEL(num_cameras, ptr_tensor.device(), {
        const auto [out_ptr, data_ptrs] = data;
        out_ptr[idx] = CameraType::pack(data_ptrs, idx);
    })
    return out;
}


template <typename CameraType>
std::vector<at::Tensor> unpack_to_tensors(const at::Tensor& pack) {
    TORCH_CHECK(
        pack.size(-1) == CameraType::pack_elements(),
        "Pack size mismatch. Expected: ",
        CameraType::pack_elements(),
        ", but found: ", pack.size(-1));

    uint num_cameras = pack.numel() / CameraType::pack_elements();
    std::vector<at::Tensor> tensors;
    constexpr uint num_tensors = CameraType::tensor_dims_base.size() + CameraType::ImplType::tensor_dims.size();
    tensors.reserve(num_tensors);
    for (uint i = 0; i != CameraType::tensor_dims_base.size(); ++i) {
        tensors.push_back(at::zeros({num_cameras, CameraType::tensor_dims_base[i]}, pack.options()));
    }
    for (uint i = 0; i != CameraType::ImplType::tensor_dims.size(); ++i) {
        tensors.push_back(at::zeros({num_cameras, CameraType::ImplType::tensor_dims[i]}, pack.options()));
    }

    auto ptr_tensor = make_ptr_tensor(tensors);
    auto ptr_tensor_ptr = static_cast<char **>(ptr_tensor.data_ptr());
    auto data = std::make_tuple(
        ptr_tensor_ptr, static_cast<const CameraType *>(pack.data_ptr()));

    INLINE_KERNEL(num_cameras, ptr_tensor.device(), {
        const auto [data_ptrs, pack_ptr] = data;
        pack_ptr[idx].unpack(data_ptrs, idx);
    })

    return tensors;
}


template <typename CameraType>
class CameraPackFunction: public torch::autograd::Function<CameraPackFunction<CameraType>> {
public:
    template <typename... Args>
    static at::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        Args&&... args
    ) {
        // ctx->save_for_backward(tensors);
        return pack_camera<CameraType>(
            std::vector<at::Tensor>{std::forward<Args>(args)...});
    }

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        // const auto packed_grad = ctx->get_saved_variables()[0];
        return unpack_to_tensors<CameraType>(grad_outputs[0]);
    }
};

template <typename CameraType, std::size_t... Is>
auto pack_from_vector_impl(
    const std::vector<at::Tensor>& v,
    std::index_sequence<Is...>
) {
    return CameraPackFunction<CameraType>::apply(v[Is]...);
}


template <typename CameraType>
at::Tensor camera_pack(const std::vector<at::Tensor>& tensors) {
    constexpr uint N =
        CameraType::ImplType::tensor_dims.size() +
            CameraType::tensor_dims_base.size();

    TORCH_CHECK(
        tensors.size() == N,
        typeid(CameraType).name(),
        " expected ", N,
        " tensors as input, but got ", tensors.size())

    return pack_from_vector_impl<CameraType>(
        tensors, std::make_index_sequence<N>{});
}


#endif //XSIMGS_CAMERA_PACK_KERNELS_H
