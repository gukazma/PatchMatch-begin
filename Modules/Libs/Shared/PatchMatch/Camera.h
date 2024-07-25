#pragma once
#include "types.h"
#include "Models.h"
#include  <vector>
class Camera
{
public:


	Camera();
	~Camera();

	// The unique identifier of the camera.
	camera_t camera_id = kInvalidCameraId;

	// The identifier of the camera model.
	CameraModelId model_id = CameraModelId::kInvalid;

	// The dimensions of the image, 0 if not initialized.
	size_t width = 0;
	size_t height = 0;

	// The focal length, principal point, and extra parameters. If the camera
	// model is not specified, this vector is empty.
	std::vector<double> params;

	// Whether there is a safe prior for the focal length,
	// e.g. manually provided or extracted from EXIF
	bool has_prior_focal_length = false;

private:

};