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

void Image::SetPoints2D(const std::vector<Eigen::Vector2d>& points) {
    points2D_.resize(points.size());
    for (point2D_t point2D_idx = 0; point2D_idx < points.size(); ++point2D_idx) {
        points2D_[point2D_idx].xy = points[point2D_idx];
    }
}

void Image::SetPoints2D(const std::vector<struct Point2D>& points) {
    points2D_ = points;
    num_points3D_ = 0;
    for (const auto& point2D : points2D_) {
        if (point2D.HasPoint3D()) {
            num_points3D_ += 1;
        }
    }
}


void Image::SetPoint3DForPoint2D(const point2D_t point2D_idx,
    const point3D_t point3D_id) {
    struct Point2D& point2D = points2D_.at(point2D_idx);
    if (!point2D.HasPoint3D()) {
        num_points3D_ += 1;
    }
    point2D.point3D_id = point3D_id;
}