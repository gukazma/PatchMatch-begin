#include <gtest/gtest.h>
#include "PatchMatch/IO.h"
TEST(MYTEST0, A)
{
	Reconstruction reconstruction;
	ReadCamerasBinary(reconstruction, "C:/projects/colmap1/sparse/0/cameras.bin");
}