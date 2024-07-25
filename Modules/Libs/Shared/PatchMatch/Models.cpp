#include "Models.h"

size_t CameraModelNumParams(const CameraModelId model_id) {
    switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel) \
  case CameraModel::model_id:          \
    return CameraModel::num_params;

        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
    }

    return 0;
}


bool CameraModelVerifyParams(const CameraModelId model_id,
    const std::vector<double>& params) {
    switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)              \
  case CameraModel::model_id:                       \
    if (params.size() == CameraModel::num_params) { \
      return true;                                  \
    }                                               \
    break;

        CAMERA_MODEL_SWITCH_CASES

#undef CAMERA_MODEL_CASE
    }

    return false;
}
