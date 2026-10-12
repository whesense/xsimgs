#include "bind_structs.h"

#include <structs/OrientedBox3D.h>
#include <toast/utils/launch_helpers.h>
#include <toast/utils/tensor_helpers.h>


__global__ void point_boxes_intersection_kernel(
    const uint num_points,
    const uint num_boxes,

    const float3* __restrict__ points,
    const quatf* __restrict__ q,
    const float3* __restrict__ t,
    const float3* __restrict__ size,

    int* __restrict__ output
) {
    const uint point_idx = blockDim.y * blockIdx.x + threadIdx.y;
    if (point_idx >= num_points) {return;}
    uint cur_box_id = threadIdx.x;

    const float3 cur_point = points[point_idx];
    int isect_idx = -1;
    const uint num_steps = (num_boxes + 31) / 32;

    for (uint i = 0; i != num_steps; ++i) {
        bool cur_result = false;
        if (cur_box_id < num_boxes) {
            OrientedBox3D<> box(q[cur_box_id], t[cur_box_id], size[cur_box_id]);
            cur_result = box.is_inside(cur_point);
        }
        uint mask = __ballot_sync(0xFFFFFFFF, cur_result);
        if (mask != 0) {
            isect_idx = static_cast<int>(cur_box_id + __ffs(mask)) - 1;
            break;
        }

        cur_box_id += 32;
    }

    if (threadIdx.x == 0) {
        output[point_idx] = isect_idx;
    }
}


at::Tensor point_boxes_intersection(
    const at::Tensor& points,
    const at::Tensor& q,
    const at::Tensor& t,
    const at::Tensor& size
) {
    // p: [N, 3], q: [K, 4], t: [K, 3], size: [K, 3]
    check_dtype_in(q, {at::kFloat});
    check_same_dtype(points, q, t, size);
    check_same_device(points, q, t, size);
    const uint num_boxes = q.numel() / 4;
    const uint num_points = points.numel() / 3;

    auto out = at::empty({num_points},
        at::TensorOptions().dtype(at::kInt).device(points.device()));

    if (q.device().is_cuda()) {
        const at::cuda::OptionalCUDAGuard device_guard(points.device());
        const dim3 threads(32, 32, 1);
        const uint num_blocks = (num_points + 31) / 32;

        point_boxes_intersection_kernel
        <<<num_blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_points, num_boxes, (float3*)points.data_ptr(),
            (quatf*)q.data_ptr(), (float3*)t.data_ptr(),
            (float3*)size.data_ptr(), (int*)out.data_ptr()
        );
    } else {
        const auto* point_ptr = static_cast<float3*>(points.data_ptr());
        const auto* q_ptr = static_cast<quatf*>(q.data_ptr());
        const auto* t_ptr = static_cast<float3*>(t.data_ptr());
        const auto* size_ptr = static_cast<float3*>(size.data_ptr());
        auto* out_ptr = out.data_ptr<int>();

        at::parallel_for(0, num_points, 0,
            [&](const uint64_t begin, const uint64_t end) {
                for (uint64_t point_idx = begin; point_idx < end; ++point_idx) {
                    int result = -1;
                    const float3 cur_point = point_ptr[point_idx];

                    for (int box_idx = 0; box_idx != num_boxes; ++box_idx) {
                        auto box = OrientedBox3D<>(
                            q_ptr[box_idx],
                            t_ptr[box_idx],
                            size_ptr[box_idx]);
                        if (box.is_inside(cur_point)) {
                            result = box_idx;
                            break;
                        }
                    }
                    out_ptr[point_idx] = result;
                }
        });
    }

    return out;
}



void bind_oriented_box(py::module_& m) {
    m.def("point_boxes_intersection", &point_boxes_intersection,
        py::arg("points"),
        py::arg("q"),
        py::arg("t"),
        py::arg("size"));
}
