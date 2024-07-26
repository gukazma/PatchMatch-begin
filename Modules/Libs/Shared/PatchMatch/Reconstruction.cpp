#include "Reconstruction.h"

Reconstruction::Reconstruction()
{
}

Reconstruction::~Reconstruction()
{
}

void Reconstruction::AddCamera(Camera camera)
{
	const camera_t camera_id = camera.camera_id;
	THROW_CHECK(camera.VerifyParams());
	THROW_CHECK(cameras_.emplace(camera_id, std::move(camera)).second);
}

void Reconstruction::AddImage(Image image)
{
	const image_t image_id = image.ImageId();
	const bool is_registered = image.IsRegistered();
	THROW_CHECK(images_.emplace(image_id, std::move(image)).second);
	if (is_registered) {
		//THROW_CHECK_NE(image_id, kInvalidImageId);
		reg_image_ids_.push_back(image_id);
	}
}

