#pragma once
#include <string>

class Image
{
public:
	Image();
	~Image();

private:
	std::string path_;
	size_t width_;
	size_t height_;
	float K_[9];
	float R_[9];
	float T_[3];
	float P_[12];
	float inv_P_[12];
};