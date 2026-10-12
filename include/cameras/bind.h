#ifndef XSIMGS_BIND_H
#define XSIMGS_BIND_H

#include <torch/csrc/utils/pybind.h>

#include "camera.h"
#include "distortion.h"
#include "pack_kernels.h"
#include "projection_kernels.h"
#include "ray_kernels.h"
#include "time_kernels.h"
#include "ut_kernels.h"
#include "ut_phm_kernels.h"
#include "base_camera_impl.h"


#define NUM_PROJECTION_ITERATIONS 5
#define DEFAULT_TIME_RESIDUAL_EARLY_STOP 1e-4f
#define DEFAULT_TIME_RESIDUAL_MAX 0.002f
#define DEFAULT_MAX_MARGIN 1.0f


template <typename CameraType, uint num_projection_iterations = NUM_PROJECTION_ITERATIONS>
py::class_<CameraType> pybind_camera(
    py::module& m,
    const char* name,
    float default_time_residual_early_stop = DEFAULT_TIME_RESIDUAL_EARLY_STOP,
    float default_time_residual_max = DEFAULT_TIME_RESIDUAL_MAX,
    float default_max_margin = DEFAULT_MAX_MARGIN,
    float default_ut_alpha = DEFAULT_UT_ALPHA,
    float default_ut_beta = DEFAULT_UT_BETA,
    float default_ut_kappa = DEFAULT_UT_KAPPA
) {
    if ((std::getenv("XSIMGS_VERBOSE") != nullptr)) {
        printf(
            "Registering \"%s\" camera ["
            "proj_type=%s, "
            "with_undistort_maps=%s]\n",
            name,
            has_camera_point_to_uvd_jacob_v<CameraType> ? "jac_camera" : (has_camera_point_to_uvd_jacob_impl_v<typename CameraType::ImplType> ? "jac_impl": "simple_iteration"),
            has_distort_uv_v<typename CameraType::ImplType> ? "true" : "false"
        );
    }

    auto cls_object = py::class_<CameraType>(m, name)
        .def_static("pack", &camera_pack<CameraType>, py::arg("tensors"))
        .def_static("camera_rays",
            &camera_rays_function<CameraType>,
            py::arg("pack"), py::arg("width"), py::arg("height"),
            py::arg("world_space") = true,
            py::arg("normalized") = true,
            py::arg("return_origins") = true,
            py::arg("return_valid") = false)
        .def_static("pixel_time",
            &camera_pixel_time<CameraType>,
            py::arg("pack"), py::arg("width"), py::arg("height"))
        .def_static("pixel_time_delta",
            &camera_pixel_time_delta<CameraType>,
            py::arg("pack"), py::arg("width"), py::arg("height"))
        .def_static("pixel_time_from_uv",
            &camera_pixel_time_from_uv<CameraType>,
            py::arg("pack"), py::arg("uvs"))
        .def_static("pixel_time_delta_from_uv",
            &camera_pixel_time_delta_from_uv<CameraType>,
            py::arg("pack"), py::arg("uvs"))
        .def_static("projection", &camera_projection_fwd<CameraType, false, num_projection_iterations>,
            py::arg("pack"),
            py::arg("points"),
            py::arg("points_velocity") = nullptr,
            py::arg("residual_thr_early_stop") = default_time_residual_early_stop,
            py::arg("max_residual") = default_time_residual_max,
            py::arg("max_margin") = default_max_margin,
            py::arg("projection_mode") = ProjectionMode::Central)
        .def_static("projection_nojac", &camera_projection_fwd<CameraType, true, num_projection_iterations>,
            py::arg("pack"),
            py::arg("points"),
            py::arg("points_velocity") = nullptr,
            py::arg("residual_thr_early_stop") = default_time_residual_early_stop,
            py::arg("max_residual") = default_time_residual_max,
            py::arg("max_margin") = default_max_margin,
            py::arg("projection_mode") = ProjectionMode::Central)
        .def_static("projection_utgs", &launch_unscented_transform<num_projection_iterations, false, CameraType>,
            py::arg("pack"),
            py::arg("points"),
            py::arg("points_velocity"),
            py::arg("scales"),
            py::arg("quats"),
            py::arg("opacity"),
            py::arg("width"),
            py::arg("height"),
            py::arg("ut_alpha") = default_ut_alpha,
            py::arg("ut_beta") = default_ut_beta,
            py::arg("ut_kappa") = default_ut_kappa,
            py::arg("residual_thr_early_stop") = default_time_residual_early_stop,
            py::arg("max_residual") = default_time_residual_max,
            py::arg("max_margin") = default_max_margin);

    if constexpr (CameraType::with_phase_modeling) {
        cls_object.def_static("projection_utgs_phm", &launch_unscented_transform_phm<num_projection_iterations, false, CameraType>,
            py::arg("pack"),
            py::arg("points"),
            py::arg("points_velocity"),
            py::arg("scales"),
            py::arg("quats"),
            py::arg("opacity"),
            py::arg("width"),
            py::arg("height"),
            py::arg("ut_alpha") = default_ut_alpha,
            py::arg("ut_beta") = default_ut_beta,
            py::arg("ut_kappa") = default_ut_kappa,
            py::arg("residual_thr_early_stop") = default_time_residual_early_stop,
            py::arg("max_residual") = default_time_residual_max,
            py::arg("max_margin") = default_max_margin);
    }

    if constexpr (has_distort_uv_v<typename CameraType::ImplType>) {
        cls_object.def_static(
            "undistort_map", &undistort_map<CameraType>,
            py::arg("pack"),
            py::arg("width"),
            py::arg("height"),
            py::arg("normalized_range") = false);
    }

    return cls_object;
}


#endif //XSIMGS_BIND_H
