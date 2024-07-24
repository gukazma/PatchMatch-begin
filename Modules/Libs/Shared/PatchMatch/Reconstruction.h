#pragma once
#include "Camera.h"
#include "Image.h"
#include <vector>
class Reconstruction
{
public:
	Reconstruction();
	~Reconstruction();

private:
	std::vector<Camera> m_cameras;
	std::vector<Image> m_images;
};