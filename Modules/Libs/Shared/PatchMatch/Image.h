// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include <string>
#include <vector>

#include <Eigen/Core>
#include "Rigid3d.h"
#include "Point2D.h"
// Class that holds information about an image. An image is the product of one
// camera shot at a certain location (parameterized as the pose). An image may
// share a camera with multiple other images, if its intrinsics are the same.
class Image {
public:
	Image();

	// Access the unique identifier of the image.
	inline image_t ImageId() const;
	inline void SetImageId(image_t image_id);

	// Access the name of the image.
	inline const std::string& Name() const;
	inline std::string& Name();
	inline void SetName(const std::string& name);

	// Access the unique identifier of the camera. Note that multiple images
	// might share the same camera.
	inline camera_t CameraId() const;
	inline void SetCameraId(camera_t camera_id);
	// Check whether identifier of camera has been set.
	inline bool HasCamera() const;

	// Check if image is registered.
	inline bool IsRegistered() const;
	inline void SetRegistered(bool registered);

	// World to camera pose.
	inline const Rigid3d& CamFromWorld() const;
	inline Rigid3d& CamFromWorld();

	// Extract the projection center in world space.
	Eigen::Vector3d ProjectionCenter() const;

	// Extract the viewing direction of the image.
	Eigen::Vector3d ViewingDirection() const;

	// Get the number of image points.
	inline point2D_t NumPoints2D() const;

	void SetPoints2D(const std::vector<Eigen::Vector2d>& points);
	void SetPoints2D(const std::vector<struct Point2D>& points);

	void SetPoint3DForPoint2D(point2D_t point2D_idx, point3D_t point3D_id);

private:
	// Identifier of the image, if not specified `kInvalidImageId`.
	image_t image_id_;

	// The name of the image, i.e. the relative path.
	std::string name_;

	// The identifier of the associated camera. Note that multiple images might
	// share the same camera. If not specified `kInvalidCameraId`.
	camera_t camera_id_;

	// Whether the image is successfully registered in the reconstruction.
	bool registered_;

	// The pose of the image, defined as the transformation from world to camera.
	Rigid3d cam_from_world_;

	// The number of 3D points the image observes, i.e. the sum of its `points2D`
  // where `point3D_id != kInvalidPoint3DId`.
	point2D_t num_points3D_;

	std::vector<struct Point2D> points2D_;
};

	////////////////////////////////////////////////////////////////////////////////
	// Implementation
	////////////////////////////////////////////////////////////////////////////////

image_t Image::ImageId() const { return image_id_; }

void Image::SetImageId(const image_t image_id) { image_id_ = image_id; }

const std::string& Image::Name() const { return name_; }

std::string& Image::Name() { return name_; }

void Image::SetName(const std::string& name) { name_ = name; }

inline camera_t Image::CameraId() const { return camera_id_; }

inline void Image::SetCameraId(const camera_t camera_id) {
	//THROW_CHECK_NE(camera_id, kInvalidCameraId);
	camera_id_ = camera_id;
}

inline bool Image::HasCamera() const { return camera_id_ != kInvalidCameraId; }

bool Image::IsRegistered() const { return registered_; }

void Image::SetRegistered(const bool registered) { registered_ = registered; }


const Rigid3d& Image::CamFromWorld() const { return cam_from_world_; }

Rigid3d& Image::CamFromWorld() { return cam_from_world_; }

point2D_t Image::NumPoints2D() const {
	return static_cast<point2D_t>(points2D_.size());
}