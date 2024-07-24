#include "IO.h"
#include "types.h"
#include "endian.h"
#include <fstream>
void ReadCamerasBinary(Reconstruction& reconstruction, const std::string& path)
{
    std::ifstream file(path, std::ios::binary);

    const size_t num_cameras = ReadBinaryLittleEndian<uint64_t>(&file);
    for (size_t i = 0; i < num_cameras; ++i) {
        struct Camera camera;
        camera.camera_id = ReadBinaryLittleEndian<camera_t>(&file);
        camera.model_id =
            static_cast<CameraModelId>(ReadBinaryLittleEndian<int>(&file));
        camera.width = ReadBinaryLittleEndian<uint64_t>(&file);
        camera.height = ReadBinaryLittleEndian<uint64_t>(&file);
        //camera.params.resize(CameraModelNumParams(camera.model_id), 0.);
        ReadBinaryLittleEndian<double>(&file, &camera.params);
        //THROW_CHECK(camera.VerifyParams());
        //reconstruction.AddCamera(std::move(camera));
    }
}
