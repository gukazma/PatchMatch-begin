#pragma once
#include "Reconstruction.h"
#include "Common/Export.h"

void DLL_API ReadCamerasBinary(Reconstruction& reconstruction,
    const std::string& path);

void DLL_API ReadImagesBinary(Reconstruction& reconstruction, const std::string& path);