#ifndef ROIBOX3D_H
#define ROIBOX3D_H


#include <toast/utils/vector_math.h>
#include <tuple>

template <typename dtype>
auto tensor_to_vec3(const torch::Tensor& t) {
    auto cpu = t.cpu().contiguous();
    const auto* d = cpu.data_ptr<typename dtype::T>();
    return typename dtype::vec3{d[0], d[1], d[2]};
}


template <typename dtype>
struct ROIBox3D {
    using vec3 = typename dtype::vec3;
    using scalar_t = typename dtype::T;
    vec3 vmin;
    vec3 vmax;

    HOSTDEVICE ROIBox3D(const vec3& _vmin, const vec3& _vmax): vmin(_vmin), vmax(_vmax) {}

    HOSTDEVICE void ray_intersection(
       const vec3& ray_origin,
       const vec3& ray_direction,
       scalar_t& t_near, scalar_t& t_far) const
    {
        auto t_min = (vmin - ray_origin) / ray_direction;
        auto t_max = (vmax - ray_origin) / ray_direction;

        auto t1 = typename dtype::vec3{
            dtype::min(t_min.x, t_max.x),
            dtype::min(t_min.y, t_max.y),
            dtype::min(t_min.z, t_max.z)
        };
        auto t2 = typename dtype::vec3{
            dtype::max(t_min.x, t_max.x),
            dtype::max(t_min.y, t_max.y),
            dtype::max(t_min.z, t_max.z)
        };
        t_near = dtype::max(t1.x, dtype::max(t1.y, t1.z));
        t_far = dtype::min(t2.x, dtype::min(t2.y, t2.z));
    }

    [[nodiscard]] HOSTDEVICE bool is_inside(const vec3& point) const {
        return (point.x >= vmin.x) & (point.x < vmax.x)
            & (point.y >= vmin.y) & (point.y < vmax.y)
            & (point.z >= vmin.z) & (point.z < vmax.z);
    }

    [[nodiscard]] HOSTDEVICE bool is_outside(const float3& point) const {
        return !is_inside(point);
    }

    [[nodiscard]] HOSTDEVICE vec3 size() const {
        return vmax - vmin;
    }

    [[nodiscard]] HOSTDEVICE vec3 normalize_into_roi(const vec3& point, bool _clamp = false) const {
        vec3 result = (point - vmin) / size();
        if (_clamp) {
            result = {
                clamp(result.x, scalar_t(0.0), scalar_t(1.0)),
                clamp(result.y, scalar_t(0.0), scalar_t(1.0)),
                clamp(result.z, scalar_t(0.0), scalar_t(1.0)),
            };
        }
        return result;
    }

    [[nodiscard]] HOSTDEVICE vec3 normalized_roi_into_world(const vec3& point) const {
        return point * size() + vmin;
    }

    [[nodiscard]] HOSTDEVICE bool ray_clamp(
       const vec3& ray_origin,
       const vec3& end_point,
       vec3& new_origin,
       vec3& new_point) const
    {
        scalar_t t_near, t_far;
        ray_intersection(ray_origin, end_point - ray_origin, t_near, t_far);

        bool origin_inside = is_inside(ray_origin);
        if (!origin_inside && (t_near < 0 || t_far < t_near)) {
            // current ray does not intersect with ROIBox3D
            return false;
        }

        scalar_t t_orig = origin_inside ? scalar_t(0): t_near;
        scalar_t t_end = scalar_t(1);

        bool point_inside = is_inside(end_point);
        scalar_t new_ts = (dtype::abs(t_near - t_end) < dtype::abs(t_far - t_end)) ? t_near : t_far;
        t_end = point_inside ? t_end : new_ts;

        new_origin = lerp(ray_origin, end_point, t_orig);
        new_point = lerp(ray_origin, end_point, t_end);
        return true;
    }
};


#endif //ROIBOX3D_H
