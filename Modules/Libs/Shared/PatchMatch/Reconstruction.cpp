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
}

