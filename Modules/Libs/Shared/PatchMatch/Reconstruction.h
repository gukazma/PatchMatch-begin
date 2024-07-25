#pragma once
#include "Camera.h"
#include "Image.h"
#include <vector>
#include "Common/Export.h"
#include <unordered_map>
class DLL_API Reconstruction
{
public:
	Reconstruction();
	~Reconstruction();
	void AddCamera(Camera camera);

	// Add new image.
	void AddImage(Image image);

private:
	std::unordered_map<camera_t, Camera> cameras_;
	std::unordered_map<image_t, Image> images_;

	// { image_id, ... } where `images_.at(image_id).registered == true`.
	std::vector<image_t> reg_image_ids_;

	// Total number of added 3D points, used to generate unique identifiers.
	point3D_t max_point3D_id_;
};