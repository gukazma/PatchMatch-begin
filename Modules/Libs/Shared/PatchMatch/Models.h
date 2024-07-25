#pragma  once
#include <string>
#include <stdexcept>

enum class CameraModelId {
    kInvalid = -1,
    kSimplePinhole = 0,
    kPinhole = 1,
    kSimpleRadial = 2,
    kRadial = 3,
    kOpenCV = 4,
    kOpenCVFisheye = 5,
    kFullOpenCV = 6,
    kFOV = 7,
    kSimpleRadialFisheye = 8,
    kRadialFisheye = 9,
    kThinPrismFisheye = 10,
};

#ifndef CAMERA_MODEL_DEFINITIONS
#define CAMERA_MODEL_DEFINITIONS(model_id_val,                                \
                                 model_name_val,                              \
                                 num_focal_params_val,                        \
                                 num_pp_params_val,                           \
                                 num_extra_params_val)                        \
  static constexpr size_t num_params =                                        \
      (num_focal_params_val) + (num_pp_params_val) + (num_extra_params_val);  \
  static constexpr size_t num_focal_params = num_focal_params_val;            \
  static constexpr size_t num_pp_params = num_pp_params_val;                  \
  static constexpr size_t num_extra_params = num_extra_params_val;            \
  static constexpr CameraModelId model_id = model_id_val;                     \
  static const std::string model_name;                                        \
  static const std::string params_info;                                       \
  static const std::array<size_t, (num_focal_params_val)> focal_length_idxs;  \
  static const std::array<size_t, (num_pp_params_val)> principal_point_idxs;  \
  static const std::array<size_t, (num_extra_params_val)> extra_params_idxs;  \
                                                                              \
  static inline CameraModelId InitializeModelId() { return model_id_val; };   \
  static inline std::string InitializeModelName() { return model_name_val; }; 
#endif

#ifndef CAMERA_MODEL_CASES
#define CAMERA_MODEL_CASES                          \
  CAMERA_MODEL_CASE(SimplePinholeCameraModel)       \
  CAMERA_MODEL_CASE(PinholeCameraModel)             \
  CAMERA_MODEL_CASE(SimpleRadialCameraModel)        \
  CAMERA_MODEL_CASE(SimpleRadialFisheyeCameraModel) \
  CAMERA_MODEL_CASE(RadialCameraModel)              \
  CAMERA_MODEL_CASE(RadialFisheyeCameraModel)       \
  CAMERA_MODEL_CASE(OpenCVCameraModel)              \
  CAMERA_MODEL_CASE(OpenCVFisheyeCameraModel)       \
  CAMERA_MODEL_CASE(FullOpenCVCameraModel)          \
  CAMERA_MODEL_CASE(FOVCameraModel)                 \
  CAMERA_MODEL_CASE(ThinPrismFisheyeCameraModel)
#endif


#ifndef CAMERA_MODEL_SWITCH_CASES
#define CAMERA_MODEL_SWITCH_CASES         \
  CAMERA_MODEL_CASES                      \
  default:                                \
    CAMERA_MODEL_DOES_NOT_EXIST_EXCEPTION \
    break;
#endif

#define CAMERA_MODEL_DOES_NOT_EXIST_EXCEPTION \
  throw std::domain_error("Camera model does not exist");

template <typename CameraModel>
struct BaseCameraModel {
};


struct SimplePinholeCameraModel
    : public BaseCameraModel<SimplePinholeCameraModel> {
    CAMERA_MODEL_DEFINITIONS(
        CameraModelId::kSimplePinhole, "SIMPLE_PINHOLE", 1, 2, 0)
};
// Pinhole camera model.
//
// No Distortion is assumed. Only focal length and principal point is modeled.
//
// Parameter list is expected in the following order:
//
//    fx, fy, cx, cy
//
// See https://en.wikipedia.org/wiki/Pinhole_camera_model
struct PinholeCameraModel : public BaseCameraModel<PinholeCameraModel> {
    CAMERA_MODEL_DEFINITIONS(CameraModelId::kPinhole, "PINHOLE", 2, 2, 0)
};

// Simple camera model with one focal length and one radial distortion
// parameter.
//
// This model is similar to the camera model that VisualSfM uses with the
// difference that the distortion here is applied to the projections and
// not to the measurements.
//
// Parameter list is expected in the following order:
//
//    f, cx, cy, k
//
struct SimpleRadialCameraModel
    : public BaseCameraModel<SimpleRadialCameraModel> {
    CAMERA_MODEL_DEFINITIONS(
        CameraModelId::kSimpleRadial, "SIMPLE_RADIAL", 1, 2, 1)
};

// Simple camera model with one focal length and two radial distortion
// parameters.
//
// This model is equivalent to the camera model that Bundler uses
// (except for an inverse z-axis in the camera coordinate system).
//
// Parameter list is expected in the following order:
//
//    f, cx, cy, k1, k2
//
struct RadialCameraModel : public BaseCameraModel<RadialCameraModel> {
    CAMERA_MODEL_DEFINITIONS(CameraModelId::kRadial, "RADIAL", 1, 2, 2)
};

// OpenCV camera model.
//
// Based on the pinhole camera model. Additionally models radial and
// tangential distortion (up to 2nd degree of coefficients). Not suitable for
// large radial distortions of fish-eye cameras.
//
// Parameter list is expected in the following order:
//
//    fx, fy, cx, cy, k1, k2, p1, p2
//
// See
// http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
struct OpenCVCameraModel : public BaseCameraModel<OpenCVCameraModel> {
    CAMERA_MODEL_DEFINITIONS(CameraModelId::kOpenCV, "OPENCV", 2, 2, 4)
};

// OpenCV fish-eye camera model.
//
// Based on the pinhole camera model. Additionally models radial distortion
// (up to 4th degree of coefficients). Suitable for
// large radial distortions of fish-eye cameras.
//
// Parameter list is expected in the following order:
//
//    fx, fy, cx, cy, k1, k2, k3, k4
//
// See
// http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
struct OpenCVFisheyeCameraModel
    : public BaseCameraModel<OpenCVFisheyeCameraModel> {
    CAMERA_MODEL_DEFINITIONS(
        CameraModelId::kOpenCVFisheye, "OPENCV_FISHEYE", 2, 2, 4)
};

// Full OpenCV camera model.
//
// Based on the pinhole camera model. Additionally models radial and
// tangential Distortion.
//
// Parameter list is expected in the following order:
//
//    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
//
// See
// http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
struct FullOpenCVCameraModel : public BaseCameraModel<FullOpenCVCameraModel> {
    CAMERA_MODEL_DEFINITIONS(CameraModelId::kFullOpenCV, "FULL_OPENCV", 2, 2, 8)
};

// FOV camera model.
//
// Based on the pinhole camera model. Additionally models radial distortion.
// This model is for example used by Project Tango for its equidistant
// calibration type.
//
// Parameter list is expected in the following order:
//
//    fx, fy, cx, cy, omega
//
// See:
// Frederic Devernay, Olivier Faugeras. Straight lines have to be straight:
// Automatic calibration and removal of distortion from scenes of structured
// environments. Machine vision and applications, 2001.
struct FOVCameraModel : public BaseCameraModel<FOVCameraModel> {
    CAMERA_MODEL_DEFINITIONS(CameraModelId::kFOV, "FOV", 2, 2, 1)

        template <typename T>
    static void Undistortion(const T* extra_params, T u, T v, T* du, T* dv);
};

// Simple camera model with one focal length and one radial distortion
// parameter, suitable for fish-eye cameras.
//
// This model is equivalent to the OpenCVFisheyeCameraModel but has only one
// radial distortion coefficient.
//
// Parameter list is expected in the following order:
//
//    f, cx, cy, k
//
struct SimpleRadialFisheyeCameraModel
    : public BaseCameraModel<SimpleRadialFisheyeCameraModel> {
    CAMERA_MODEL_DEFINITIONS(
        CameraModelId::kSimpleRadialFisheye, "SIMPLE_RADIAL_FISHEYE", 1, 2, 1)
};

// Simple camera model with one focal length and two radial distortion
// parameters, suitable for fish-eye cameras.
//
// This model is equivalent to the OpenCVFisheyeCameraModel but has only two
// radial distortion coefficients.
//
// Parameter list is expected in the following order:
//
//    f, cx, cy, k1, k2
//
struct RadialFisheyeCameraModel
    : public BaseCameraModel<RadialFisheyeCameraModel> {
    CAMERA_MODEL_DEFINITIONS(
        CameraModelId::kRadialFisheye, "RADIAL_FISHEYE", 1, 2, 2)
};

// Camera model with radial and tangential distortion coefficients and
// additional coefficients accounting for thin-prism distortion.
//
// This camera model is described in
//
//    "Camera Calibration with Distortion Models and Accuracy Evaluation",
//    J Weng et al., TPAMI, 1992.
//
// Parameter list is expected in the following order:
//
//    fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1
//
struct ThinPrismFisheyeCameraModel
    : public BaseCameraModel<ThinPrismFisheyeCameraModel> {
    CAMERA_MODEL_DEFINITIONS(
        CameraModelId::kThinPrismFisheye, "THIN_PRISM_FISHEYE", 2, 2, 8)
};

size_t CameraModelNumParams(const CameraModelId model_id);