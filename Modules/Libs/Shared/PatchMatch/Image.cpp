#include "Image.h"

Image::Image()
    : image_id_(kInvalidImageId),
    name_(""),
    camera_id_(kInvalidCameraId),
    registered_(false)
    {}

Eigen::Vector3d Image::ProjectionCenter() const {
    return cam_from_world_.rotation.inverse() * -cam_from_world_.translation;
}

Eigen::Vector3d Image::ViewingDirection() const {
    return cam_from_world_.rotation.toRotationMatrix().row(2);
}