// Copyright 2025 Nerfstudio Team (original gsplat code)
// Modifications Copyright 2026 the xsimgs authors
// SPDX-License-Identifier: Apache-2.0
//
// Derived from gsplat (https://github.com/nerfstudio-project/gsplat),
// licensed under the Apache License 2.0.
// This file has been substantially modified from the original.

#include "sh_fwd.h"

#include <toast/utils/vector_math.h>
#include <toast/utils/launch_helpers.h>
#include <toast/utils/tensor_helpers.h>


template <int degree>
HOSTDEVICE float sh_coeffs_to_color_fast(
    const float3 &dir,        // [3]
    const float* coeffs, // [K, 3]; if dc_coeffs != nullptr, starts at degree 1
    const float* dc_coeffs, // optional zero-degree coeffs; nullptr => coeffs[0]
    const uint band_offset
) {
    // When dc_coeffs is provided, coeffs holds bands starting from degree 1, so
    // shift the base such that coeffs_sh[i * band_offset] is still SH band i.
    float result;
    const float* coeffs_sh;
    if (dc_coeffs != nullptr) {
        result = 0.2820947917738781f * dc_coeffs[0];
        coeffs_sh = coeffs - band_offset;
    } else {
        result = 0.2820947917738781f * coeffs[0];
        coeffs_sh = coeffs;
    }
    if (degree >= 1) {
        const auto [x, y, z] = dir;

        result +=
            0.48860251190292f * (-y * coeffs_sh[band_offset]
                                 +z * coeffs_sh[2 * band_offset]
                                 -x * coeffs_sh[3 * band_offset]);
        if (degree >= 2) {
            float z2 = z * z;

            float fTmp0B = -1.092548430592079f * z;
            float fC1 = x * x - y * y;
            float fS1 = 2.f * x * y;
            float pSH6 = (0.9461746957575601f * z2 - 0.3153915652525201f);
            float pSH7 = fTmp0B * x;
            float pSH5 = fTmp0B * y;
            float pSH8 = 0.5462742152960395f * fC1;
            float pSH4 = 0.5462742152960395f * fS1;

            result += pSH4 * coeffs_sh[4 * band_offset] +
                      pSH5 * coeffs_sh[5 * band_offset] +
                      pSH6 * coeffs_sh[6 * band_offset] +
                      pSH7 * coeffs_sh[7 * band_offset] +
                      pSH8 * coeffs_sh[8 * band_offset];
            if (degree >= 3) {
                float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
                float fTmp1B = 1.445305721320277f * z;
                float fC2 = x * fC1 - y * fS1;
                float fS2 = x * fS1 + y * fC1;
                float pSH12 =
                    z * (1.865881662950577f * z2 - 1.119528997770346f);
                float pSH13 = fTmp0C * x;
                float pSH11 = fTmp0C * y;
                float pSH14 = fTmp1B * fC1;
                float pSH10 = fTmp1B * fS1;
                float pSH15 = -0.5900435899266435f * fC2;
                float pSH9 = -0.5900435899266435f * fS2;

                result +=
                    pSH9 * coeffs_sh[9 * band_offset] +
                    pSH10 * coeffs_sh[10 * band_offset] +
                    pSH11 * coeffs_sh[11 * band_offset] +
                    pSH12 * coeffs_sh[12 * band_offset] +
                    pSH13 * coeffs_sh[13 * band_offset] +
                    pSH14 * coeffs_sh[14 * band_offset] +
                    pSH15 * coeffs_sh[15 * band_offset];

                if (degree >= 4) {
                    float fTmp0D =
                        z * (-4.683325804901025f * z2 + 2.007139630671868f);
                    float fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
                    float fTmp2B = -1.770130769779931f * z;
                    float fC3 = x * fC2 - y * fS2;
                    float fS3 = x * fS2 + y * fC2;
                    float pSH20 =
                        (1.984313483298443f * z * pSH12 -
                         1.006230589874905f * pSH6);
                    float pSH21 = fTmp0D * x;
                    float pSH19 = fTmp0D * y;
                    float pSH22 = fTmp1C * fC1;
                    float pSH18 = fTmp1C * fS1;
                    float pSH23 = fTmp2B * fC2;
                    float pSH17 = fTmp2B * fS2;
                    float pSH24 = 0.6258357354491763f * fC3;
                    float pSH16 = 0.6258357354491763f * fS3;

                    result += pSH16 * coeffs_sh[16 * band_offset] +
                              pSH17 * coeffs_sh[17 * band_offset] +
                              pSH18 * coeffs_sh[18 * band_offset] +
                              pSH19 * coeffs_sh[19 * band_offset] +
                              pSH20 * coeffs_sh[20 * band_offset] +
                              pSH21 * coeffs_sh[21 * band_offset] +
                              pSH22 * coeffs_sh[22 * band_offset] +
                              pSH23 * coeffs_sh[23 * band_offset] +
                              pSH24 * coeffs_sh[24 * band_offset];
                }
            }
        }
    }

    return result;
}



template <bool layout_3kn, int degree = 3>
__global__ void spherical_harmonics_fused_fwd_kernel(
    const uint N,
    const uint dim,
    const float3* origin,
    const float3* __restrict__ positions,
    const float* __restrict__ coeffs,
    const float* __restrict__ dc_coeffs,
    const uint* __restrict__ mask,

    float* __restrict__ output
) {
    uint thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
    uint idx = thread_rank / 3;
    uint color_channel = thread_rank % 3;
    if (idx >= N) return;

    const bool is_valid = mask ? ((mask[idx / 32] & (1 << (idx % 32))) > 0) : true;
    if (is_valid) {
        const float3 dir = normalize(positions[idx] - origin[0]);

        if constexpr (layout_3kn) {
            // layout = [3, K, N]; dc layout = [3, 1, N]
            const float* coeffs_cur = coeffs + color_channel * N * dim + idx;
            const float* dc_cur =
                dc_coeffs ? dc_coeffs + color_channel * N + idx : nullptr;
            const uint band_offset = N;
            output[3 * idx + color_channel] = sh_coeffs_to_color_fast<degree>(
                dir, coeffs_cur, dc_cur, band_offset) + 0.5f;
        } else {
            // layout = [N, K, 3]; dc layout = [N, 1, 3]
            const float* coeffs_cur = coeffs + idx * dim * 3 + color_channel;
            const float* dc_cur =
                dc_coeffs ? dc_coeffs + idx * 3 + color_channel : nullptr;
            constexpr uint band_offset = 3;
            output[3 * idx + color_channel] = sh_coeffs_to_color_fast<degree>(
                dir, coeffs_cur, dc_cur, band_offset) + 0.5f;
        }

    }
}


torch::Tensor spherical_harmonics_fused_fwd(
    const uint degree_to_use,
    const torch::Tensor& origin,
    const torch::Tensor& positions,
    const torch::Tensor& coeffs,
    const std::optional<torch::Tensor>& mask,
    const bool layout_3kn,
    const std::optional<torch::Tensor>& coeffs_dc,
    const std::optional<torch::Tensor>& output
) {
    const bool has_dc = coeffs_dc.has_value();
    check_same_device(
        origin, positions, coeffs,
        has_dc ? coeffs_dc.value() : origin,
        mask.has_value() ? mask.value() : origin,
        output.has_value() ? output.value() : origin);
    all_contiguous(
        origin, positions, coeffs,
        has_dc ? coeffs_dc.value() : origin,
        mask.has_value() ? mask.value() : origin,
        output.has_value() ? output.value() : origin);

    const uint N = layout_3kn ? coeffs.size(2) : coeffs.size(0);
    const uint dim = coeffs.size(1);
    TORCH_CHECK(coeffs.size(layout_3kn ? 0 : 2) == 3,
        "coeffs tensor must be [3, K, N] if layout_3kn else [N, K, 3]")

    // When the zero-degree coeffs are passed separately, the coeffs tensor holds
    // bands starting from degree 1, i.e. one band fewer.
    constexpr std::array<uint, 5> ref_dims{1, 1+3, 1+3+5, 1+3+5+7, 1+3+5+7+9};
    const uint required_dim = ref_dims[degree_to_use] - (has_dc ? 1u : 0u);
    TORCH_CHECK(dim >= required_dim,
        "Number of harmonics band in coeffs tensor must be higher or equal than ",
        required_dim, " for degree_to_use=", degree_to_use,
        has_dc ? " (zero-degree coeffs passed separately)" : "");
    if (has_dc) {
        // Accept the single DC band with or without the singleton band axis:
        // [N, 3] / [N, 1, 3]  (!layout_3kn)  or  [3, N] / [3, 1, N]  (layout_3kn).
        const auto dc_dim = coeffs_dc->dim();
        TORCH_CHECK(dc_dim == 2 || dc_dim == 3,
            "coeffs_dc must be 2D or 3D");
        const int64_t ch_axis = layout_3kn ? 0 : dc_dim - 1;
        const int64_t n_axis = layout_3kn ? dc_dim - 1 : 0;
        TORCH_CHECK(coeffs_dc->size(ch_axis) == 3 &&
                    coeffs_dc->size(n_axis) == N &&
                    coeffs_dc->numel() == 3 * N,
            "coeffs_dc must hold a single SH band: [N, 3] or [N, 1, 3] if "
            "!layout_3kn else [3, N] or [3, 1, N]");
    }

    auto out = output.has_value() ? output.value() :
        torch::zeros({N, 3}, positions.options());

#define launch_case(deg, layout_v)\
    launch_kernel_cuda(N * 3, origin.device(), \
        spherical_harmonics_fused_fwd_kernel<layout_v, deg>,\
        N, dim, (float3*)origin.data_ptr(),\
        (float3*)positions.data_ptr(),\
        (float*)coeffs.data_ptr(),\
        has_dc ? (float*)coeffs_dc->data_ptr() : nullptr,\
        mask.has_value() ? (uint*)mask->data_ptr() : nullptr,\
        (float*)out.data_ptr()\
    );

#define launch_deg_cases(layout_v)\
    switch (degree_to_use) {\
        case 0: {launch_case(0, layout_v) break;}\
        case 1: {launch_case(1, layout_v) break;}\
        case 2: {launch_case(2, layout_v) break;}\
        case 3: {launch_case(3, layout_v) break;}\
        case 4: {launch_case(4, layout_v) break;}\
        default: throw std::runtime_error(\
            "Unsupported spherical harmonics degree: " +\
                std::to_string(degree_to_use));\
    }

    if (layout_3kn) {
        launch_deg_cases(true);
    } else {
        launch_deg_cases(false);
    }

    return out;
}


void bind_sh_fwd(py::module_& m) {
    m.def("spherical_harmonics_fused_fwd", &spherical_harmonics_fused_fwd,
        py::arg("degree_to_use"),
        py::arg("origin"),
        py::arg("positions"),
        py::arg("coeffs"),
        py::arg("mask"),
        py::arg("layout_3kn"),
        py::arg("coeffs_dc") = nullptr,
        py::arg("output") = nullptr
    );
}