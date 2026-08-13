// Copyright 2025 Nerfstudio Team (original gsplat code)
// Modifications Copyright 2026 the xsimgs authors
// SPDX-License-Identifier: Apache-2.0
//
// Derived from gsplat (https://github.com/nerfstudio-project/gsplat),
// licensed under the Apache License 2.0.
// This file has been substantially modified from the original.

#include "sh_bwd.h"

#include <toast/utils/vector_math.h>
#include <toast/utils/bwd_math_utils.h>
#include <toast/math/scalar.h>
#include <toast/utils/launch_helpers.h>
#include <toast/utils/tensor_helpers.h>

#include <utils/reduce_helpers.cuh>

template <int degree>
HOSTDEVICE void sh_coeffs_to_color_fast_backward(
    const float3 &dir,
    const float* coeffs,    // if d_dc_coeffs != nullptr, starts at degree 1
    const uint band_offset,
    const float d_out,
    // output
    float* d_coeffs,        // if d_dc_coeffs != nullptr, starts at degree 1
    float* d_dc_coeffs,     // optional zero-degree gradient; nullptr => d_coeffs[0]
    float3& d_dir
) {
    // When the zero-degree gradient is written to a separate tensor, both coeffs
    // and d_coeffs hold bands starting from degree 1; shift the base so that
    // index [i * band_offset] still refers to SH band i.
    const float* coeffs_sh;
    float* d_coeffs_sh;
    if (d_dc_coeffs != nullptr) {
        d_dc_coeffs[0] = 0.2820947917738781f * d_out;
        coeffs_sh = coeffs - band_offset;
        d_coeffs_sh = d_coeffs - band_offset;
    } else {
        d_coeffs[0] = 0.2820947917738781f * d_out;
        coeffs_sh = coeffs;
        d_coeffs_sh = d_coeffs;
    }
    if (degree < 1) {
        return;
    }
    const auto [x, y, z] = dir;

    d_coeffs_sh[    band_offset] = -0.48860251190292f * y * d_out;
    d_coeffs_sh[2 * band_offset] = 0.48860251190292f * z * d_out;
    d_coeffs_sh[3 * band_offset] = -0.48860251190292f * x * d_out;

    d_dir.x += -0.48860251190292f * coeffs_sh[3 * band_offset] * d_out;
    d_dir.y += -0.48860251190292f * coeffs_sh[1 * band_offset] * d_out;
    d_dir.z += 0.48860251190292f * coeffs_sh[2 * band_offset] * d_out;

    if (degree < 2) {
        return;
    }

    float z2 = z * z;
    float fTmp0B = -1.092548430592079f * z;
    float fC1 = x * x - y * y;
    float fS1 = 2.f * x * y;
    float pSH6 = (0.9461746957575601f * z2 - 0.3153915652525201f);
    float pSH7 = fTmp0B * x;
    float pSH5 = fTmp0B * y;
    float pSH8 = 0.5462742152960395f * fC1;
    float pSH4 = 0.5462742152960395f * fS1;
    d_coeffs_sh[4 * band_offset] = pSH4 * d_out;
    d_coeffs_sh[5 * band_offset] = pSH5 * d_out;
    d_coeffs_sh[6 * band_offset] = pSH6 * d_out;
    d_coeffs_sh[7 * band_offset] = pSH7 * d_out;
    d_coeffs_sh[8 * band_offset] = pSH8 * d_out;

    float fTmp0B_z, fC1_x, fC1_y, fS1_x, fS1_y, pSH6_z, pSH7_x, pSH7_z, pSH5_y,
        pSH5_z, pSH8_x, pSH8_y, pSH4_x, pSH4_y;

    fTmp0B_z = -1.092548430592079f;
    fC1_x = 2.f * x;
    fC1_y = -2.f * y;
    fS1_x = 2.f * y;
    fS1_y = 2.f * x;
    pSH6_z = 2.f * 0.9461746957575601f * z;
    pSH7_x = fTmp0B;
    pSH7_z = fTmp0B_z * x;
    pSH5_y = fTmp0B;
    pSH5_z = fTmp0B_z * y;
    pSH8_x = 0.5462742152960395f * fC1_x;
    pSH8_y = 0.5462742152960395f * fC1_y;
    pSH4_x = 0.5462742152960395f * fS1_x;
    pSH4_y = 0.5462742152960395f * fS1_y;

    d_dir.x += d_out *
           (pSH4_x * coeffs_sh[4 * band_offset] + pSH8_x * coeffs_sh[8 * band_offset] +
            pSH7_x * coeffs_sh[7 * band_offset]);
    d_dir.y += d_out *
           (pSH4_y * coeffs_sh[4 * band_offset] + pSH8_y * coeffs_sh[8 * band_offset] +
            pSH5_y * coeffs_sh[5 * band_offset]);
    d_dir.z += d_out *
           (pSH6_z * coeffs_sh[6 * band_offset] + pSH7_z * coeffs_sh[7 * band_offset] +
            pSH5_z * coeffs_sh[5 * band_offset]);

    if (degree < 3) {
        return;
    }

    float fTmp0C = -2.285228997322329f * z2 + 0.4570457994644658f;
    float fTmp1B = 1.445305721320277f * z;
    float fC2 = x * fC1 - y * fS1;
    float fS2 = x * fS1 + y * fC1;
    float pSH12 = z * (1.865881662950577f * z2 - 1.119528997770346f);
    float pSH13 = fTmp0C * x;
    float pSH11 = fTmp0C * y;
    float pSH14 = fTmp1B * fC1;
    float pSH10 = fTmp1B * fS1;
    float pSH15 = -0.5900435899266435f * fC2;
    float pSH9 = -0.5900435899266435f * fS2;
    d_coeffs_sh[9 * band_offset] = pSH9 * d_out;
    d_coeffs_sh[10 * band_offset] = pSH10 * d_out;
    d_coeffs_sh[11 * band_offset] = pSH11 * d_out;
    d_coeffs_sh[12 * band_offset] = pSH12 * d_out;
    d_coeffs_sh[13 * band_offset] = pSH13 * d_out;
    d_coeffs_sh[14 * band_offset] = pSH14 * d_out;
    d_coeffs_sh[15 * band_offset] = pSH15 * d_out;

    float fTmp0C_z, fTmp1B_z, fC2_x, fC2_y, fS2_x, fS2_y, pSH12_z, pSH13_x,
        pSH13_z, pSH11_y, pSH11_z, pSH14_x, pSH14_y, pSH14_z, pSH10_x, pSH10_y,
        pSH10_z, pSH15_x, pSH15_y, pSH9_x, pSH9_y;

    fTmp0C_z = -2.285228997322329f * 2.f * z;
    fTmp1B_z = 1.445305721320277f;
    fC2_x = fC1 + x * fC1_x - y * fS1_x;
    fC2_y = x * fC1_y - fS1 - y * fS1_y;
    fS2_x = fS1 + x * fS1_x + y * fC1_x;
    fS2_y = x * fS1_y + fC1 + y * fC1_y;
    pSH12_z = 3.f * 1.865881662950577f * z2 - 1.119528997770346f;
    pSH13_x = fTmp0C;
    pSH13_z = fTmp0C_z * x;
    pSH11_y = fTmp0C;
    pSH11_z = fTmp0C_z * y;
    pSH14_x = fTmp1B * fC1_x;
    pSH14_y = fTmp1B * fC1_y;
    pSH14_z = fTmp1B_z * fC1;
    pSH10_x = fTmp1B * fS1_x;
    pSH10_y = fTmp1B * fS1_y;
    pSH10_z = fTmp1B_z * fS1;
    pSH15_x = -0.5900435899266435f * fC2_x;
    pSH15_y = -0.5900435899266435f * fC2_y;
    pSH9_x = -0.5900435899266435f * fS2_x;
    pSH9_y = -0.5900435899266435f * fS2_y;

    d_dir.x += d_out *
           (pSH9_x * coeffs_sh[9 * band_offset] + pSH15_x * coeffs_sh[15 * band_offset] +
            pSH10_x * coeffs_sh[10 * band_offset] + pSH14_x * coeffs_sh[14 * band_offset] +
            pSH13_x * coeffs_sh[13 * band_offset]);

    d_dir.y += d_out *
           (pSH9_y * coeffs_sh[9 * band_offset] + pSH15_y * coeffs_sh[15 * band_offset] +
            pSH10_y * coeffs_sh[10 * band_offset] + pSH14_y * coeffs_sh[14 * band_offset] +
            pSH11_y * coeffs_sh[11 * band_offset]);

    d_dir.z += d_out *
           (pSH12_z * coeffs_sh[12 * band_offset] + pSH13_z * coeffs_sh[13 * band_offset] +
            pSH11_z * coeffs_sh[11 * band_offset] + pSH14_z * coeffs_sh[14 * band_offset] +
            pSH10_z * coeffs_sh[10 * band_offset]);

    if (degree < 4) {
        return;
    }

    float fTmp0D = z * (-4.683325804901025f * z2 + 2.007139630671868f);
    float fTmp1C = 3.31161143515146f * z2 - 0.47308734787878f;
    float fTmp2B = -1.770130769779931f * z;
    float fC3 = x * fC2 - y * fS2;
    float fS3 = x * fS2 + y * fC2;
    float pSH20 = (1.984313483298443f * z * pSH12 + -1.006230589874905f * pSH6);
    float pSH21 = fTmp0D * x;
    float pSH19 = fTmp0D * y;
    float pSH22 = fTmp1C * fC1;
    float pSH18 = fTmp1C * fS1;
    float pSH23 = fTmp2B * fC2;
    float pSH17 = fTmp2B * fS2;
    float pSH24 = 0.6258357354491763f * fC3;
    float pSH16 = 0.6258357354491763f * fS3;
    d_coeffs_sh[16 * band_offset] = pSH16 * d_out;
    d_coeffs_sh[17 * band_offset] = pSH17 * d_out;
    d_coeffs_sh[18 * band_offset] = pSH18 * d_out;
    d_coeffs_sh[19 * band_offset] = pSH19 * d_out;
    d_coeffs_sh[20 * band_offset] = pSH20 * d_out;
    d_coeffs_sh[21 * band_offset] = pSH21 * d_out;
    d_coeffs_sh[22 * band_offset] = pSH22 * d_out;
    d_coeffs_sh[23 * band_offset] = pSH23 * d_out;
    d_coeffs_sh[24 * band_offset] = pSH24 * d_out;

    float fTmp0D_z, fTmp1C_z, fTmp2B_z, fC3_x, fC3_y, fS3_x, fS3_y, pSH20_z,
        pSH21_x, pSH21_z, pSH19_y, pSH19_z, pSH22_x, pSH22_y, pSH22_z, pSH18_x,
        pSH18_y, pSH18_z, pSH23_x, pSH23_y, pSH23_z, pSH17_x, pSH17_y, pSH17_z,
        pSH24_x, pSH24_y, pSH16_x, pSH16_y;

    fTmp0D_z = 3.f * -4.683325804901025f * z2 + 2.007139630671868f;
    fTmp1C_z = 2.f * 3.31161143515146f * z;
    fTmp2B_z = -1.770130769779931f;
    fC3_x = fC2 + x * fC2_x - y * fS2_x;
    fC3_y = x * fC2_y - fS2 - y * fS2_y;
    fS3_x = fS2 + y * fC2_x + x * fS2_x;
    fS3_y = x * fS2_y + fC2 + y * fC2_y;
    pSH20_z = 1.984313483298443f * (pSH12 + z * pSH12_z) +
              -1.006230589874905f * pSH6_z;
    pSH21_x = fTmp0D;
    pSH21_z = fTmp0D_z * x;
    pSH19_y = fTmp0D;
    pSH19_z = fTmp0D_z * y;
    pSH22_x = fTmp1C * fC1_x;
    pSH22_y = fTmp1C * fC1_y;
    pSH22_z = fTmp1C_z * fC1;
    pSH18_x = fTmp1C * fS1_x;
    pSH18_y = fTmp1C * fS1_y;
    pSH18_z = fTmp1C_z * fS1;
    pSH23_x = fTmp2B * fC2_x;
    pSH23_y = fTmp2B * fC2_y;
    pSH23_z = fTmp2B_z * fC2;
    pSH17_x = fTmp2B * fS2_x;
    pSH17_y = fTmp2B * fS2_y;
    pSH17_z = fTmp2B_z * fS2;
    pSH24_x = 0.6258357354491763f * fC3_x;
    pSH24_y = 0.6258357354491763f * fC3_y;
    pSH16_x = 0.6258357354491763f * fS3_x;
    pSH16_y = 0.6258357354491763f * fS3_y;

    d_dir.x += d_out *
           (pSH16_x * coeffs_sh[16 * band_offset] + pSH24_x * coeffs_sh[24 * band_offset] +
            pSH17_x * coeffs_sh[17 * band_offset] + pSH23_x * coeffs_sh[23 * band_offset] +
            pSH18_x * coeffs_sh[18 * band_offset] + pSH22_x * coeffs_sh[22 * band_offset] +
            pSH21_x * coeffs_sh[21 * band_offset]);
    d_dir.y += d_out *
           (pSH16_y * coeffs_sh[16 * band_offset] + pSH24_y * coeffs_sh[24 * band_offset] +
            pSH17_y * coeffs_sh[17 * band_offset] + pSH23_y * coeffs_sh[23 * band_offset] +
            pSH18_y * coeffs_sh[18 * band_offset] + pSH22_y * coeffs_sh[22 * band_offset] +
            pSH19_y * coeffs_sh[19 * band_offset]);
    d_dir.z += d_out *
           (pSH20_z * coeffs_sh[20 * band_offset] + pSH21_z * coeffs_sh[21 * band_offset] +
            pSH19_z * coeffs_sh[19 * band_offset] + pSH22_z * coeffs_sh[22 * band_offset] +
            pSH18_z * coeffs_sh[18 * band_offset] + pSH23_z * coeffs_sh[23 * band_offset] +
            pSH17_z * coeffs_sh[17 * band_offset]);
}


template <bool layout_3kn, int degree, int num_threads>
__global__ void spherical_harmonics_fused_bwd_kernel(
    const uint N,
    const uint dim,
    const float3* origin,
    const float3* __restrict__ positions,
    const float* __restrict__ coeffs,
    const uint* __restrict__ mask,
    const float* __restrict__ d_outputs,

    float* __restrict__ d_coeffs,
    float* __restrict__ d_dc_coeffs,
    float* __restrict__ d_positions,
    float3* __restrict__ d_origin_block
) {
    uint thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
    uint idx = thread_rank / 3;
    uint color_channel = thread_rank % 3;

    float3 d_origin{0.0f, 0.0f, 0.0f};

    if (idx < N) {
        const bool is_valid = mask ? ((mask[idx / 32] & (1 << (idx % 32))) > 0) : true;

        if (is_valid) {
            const float3 diff = positions[idx] - origin[0];
            const float d_out = d_outputs[3 * idx + color_channel];
            float3 d_dir_norm{0.0f, 0.0f, 0.0f};

            if constexpr (layout_3kn) {
                // layout = [3, K, N]; dc layout = [3, 1, N]
                const float* coeffs_cur = coeffs + color_channel * N * dim + idx;
                float* d_coeffs_cur = d_coeffs + color_channel * N * dim + idx;
                float* d_dc_cur =
                    d_dc_coeffs ? d_dc_coeffs + color_channel * N + idx : nullptr;

                sh_coeffs_to_color_fast_backward<degree>(
                    normalize(diff), coeffs_cur, N, d_out,
                    d_coeffs_cur, d_dc_cur, d_dir_norm);
            } else {
                // layout = [N, K, 3]; dc layout = [N, 1, 3]
                const float* coeffs_cur = coeffs + idx * dim * 3 + color_channel;
                float* d_coeffs_cur = d_coeffs + idx * dim * 3 + color_channel;
                float* d_dc_cur =
                    d_dc_coeffs ? d_dc_coeffs + idx * 3 + color_channel : nullptr;
                sh_coeffs_to_color_fast_backward<degree>(
                    normalize(diff), coeffs_cur, 3, d_out,
                    d_coeffs_cur, d_dc_cur, d_dir_norm);
            }

            const float3 d_diff = normalize_bwd<Float32>(diff, d_dir_norm);

            atomicAdd(d_positions + 3*idx, d_diff.x);
            atomicAdd(d_positions + 3*idx+1, d_diff.y);
            atomicAdd(d_positions + 3*idx+2, d_diff.z);

            d_origin = -d_diff;
        }
    }

    __shared__  typename cub::BlockReduce<float3, num_threads>::TempStorage sh_mem;
    const float3 tmp = cub::BlockReduce<float3, num_threads>(sh_mem).Sum(d_origin);
    if (threadIdx.x == 0) {
        d_origin_block[blockIdx.x] = tmp;
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
spherical_harmonics_fused_bwd(
    const uint degree_to_use,
    const torch::Tensor& origin,
    const torch::Tensor& positions,
    const torch::Tensor& coeffs,
    const std::optional<torch::Tensor>& mask,
    const bool layout_3kn,
    const torch::Tensor& d_output,
    const std::optional<torch::Tensor>& coeffs_dc
) {
    all_contiguous(d_output);
    const bool has_dc = coeffs_dc.has_value();
    const uint N = layout_3kn ? coeffs.size(2) : coeffs.size(0);
    const uint dim = coeffs.size(1);

    constexpr uint num_threads = 1024;
    const uint num_blocks = (3 * N + num_threads - 1) / num_threads;

    auto d_coeffs = torch::zeros_like(coeffs);
    // Gradient for the separate zero-degree coeffs (empty when not used).
    auto d_coeffs_dc = has_dc ? torch::zeros_like(coeffs_dc.value())
                              : torch::empty({0}, coeffs.options());
    auto d_positions = torch::zeros_like(positions);
    auto d_origin_block = torch::zeros({num_blocks, 3}, origin.options());

#define launch_case(deg, layout_v)\
    launch_kernel_cuda<num_threads>(N * 3, origin.device(), \
        spherical_harmonics_fused_bwd_kernel<layout_v, deg, num_threads>,\
        N, dim, (float3*)origin.data_ptr(),\
        (float3*)positions.data_ptr(),\
        (float*)coeffs.data_ptr(),\
        mask.has_value() ? (uint*)mask->data_ptr() : nullptr,\
        (float*)d_output.data_ptr(),\
        (float*)d_coeffs.data_ptr(),\
        has_dc ? (float*)d_coeffs_dc.data_ptr() : nullptr,\
        (float*)d_positions.data_ptr(),\
        (float3*)d_origin_block.data_ptr()\
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

    return {d_coeffs, d_coeffs_dc, d_positions, d_origin_block.sum(0)};
}


void bind_sh_bwd(py::module &m) {
    m.def("spherical_harmonics_fused_bwd", &spherical_harmonics_fused_bwd,
        py::arg("degree_to_use"),
        py::arg("origin"),
        py::arg("positions"),
        py::arg("coeffs"),
        py::arg("mask"),
        py::arg("layout_3kn"),
        py::arg("d_output"),
        py::arg("coeffs_dc") = nullptr
    );
}
