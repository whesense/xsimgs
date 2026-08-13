#ifndef ORIENTEDBOX3D_H
#define ORIENTEDBOX3D_H


#include <toast/math/se3.h>


template <typename scalar_t = Float32>
struct OrientedBox3D {
    using vec3 = typename scalar_t::vec3;
    using quat_t = quat<scalar_t>;
    using se3_t = se3<scalar_t>;
    quat_t q;
    vec3 t;
    vec3 size;

    HOSTDEVICE OrientedBox3D(const quat_t& q, const vec3& t, const vec3& size):
        q(q), t(t), size(size) {}

    [[nodiscard]] HOSTDEVICE float length() const {return size.x;}
    [[nodiscard]] HOSTDEVICE float width() const {return size.y;}
    [[nodiscard]] HOSTDEVICE float height() const {return size.z;}

    [[nodiscard]] HOSTDEVICE bool is_inside(const vec3& p) const {
        const vec3 local = se3_t(q, t).apply_inv(p);
        return (scalar_t::abs(local.x) <= 0.5f*size.x) &&
                (scalar_t::abs(local.y) <= 0.5f*size.y) &&
                (scalar_t::abs(local.z) <= 0.5f*size.z);
    }
    [[nodiscard]] HOSTDEVICE bool is_outside(const vec3& p) const {
        return !is_inside(p);
    }
};

#endif //ORIENTEDBOX3D_H
