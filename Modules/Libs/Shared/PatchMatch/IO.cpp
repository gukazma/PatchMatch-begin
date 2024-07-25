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
        camera.params.resize(CameraModelNumParams(camera.model_id), 0.);
        ReadBinaryLittleEndian<double>(&file, &camera.params);
        camera.VerifyParams();
        reconstruction.AddCamera(std::move(camera));
    }
}



void ReadImagesBinary(Reconstruction& reconstruction, const std::string& path) {
    std::ifstream file(path, std::ios::binary);

    const size_t num_reg_images = ReadBinaryLittleEndian<uint64_t>(&file);
    for (size_t i = 0; i < num_reg_images; ++i) {
        class Image image;

        image.SetImageId(ReadBinaryLittleEndian<image_t>(&file));

        Rigid3d& cam_from_world = image.CamFromWorld();
        cam_from_world.rotation.w() = ReadBinaryLittleEndian<double>(&file);
        cam_from_world.rotation.x() = ReadBinaryLittleEndian<double>(&file);
        cam_from_world.rotation.y() = ReadBinaryLittleEndian<double>(&file);
        cam_from_world.rotation.z() = ReadBinaryLittleEndian<double>(&file);
        cam_from_world.rotation.normalize();
        cam_from_world.translation.x() = ReadBinaryLittleEndian<double>(&file);
        cam_from_world.translation.y() = ReadBinaryLittleEndian<double>(&file);
        cam_from_world.translation.z() = ReadBinaryLittleEndian<double>(&file);

        image.SetCameraId(ReadBinaryLittleEndian<camera_t>(&file));

        char name_char;
        do {
            file.read(&name_char, 1);
            if (name_char != '\0') {
                image.Name() += name_char;
            }
        } while (name_char != '\0');

        const size_t num_points2D = ReadBinaryLittleEndian<uint64_t>(&file);

        std::vector<Eigen::Vector2d> points2D;
        points2D.reserve(num_points2D);
        std::vector<point3D_t> point3D_ids;
        point3D_ids.reserve(num_points2D);
        for (size_t j = 0; j < num_points2D; ++j) {
            const double x = ReadBinaryLittleEndian<double>(&file);
            const double y = ReadBinaryLittleEndian<double>(&file);
            points2D.emplace_back(x, y);
            point3D_ids.push_back(ReadBinaryLittleEndian<point3D_t>(&file));
        }

       /* image.SetPoints2D(points2D);

        for (point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D();
            ++point2D_idx) {
            if (point3D_ids[point2D_idx] != kInvalidPoint3DId) {
                image.SetPoint3DForPoint2D(point2D_idx, point3D_ids[point2D_idx]);
            }
        }*/

        image.SetRegistered(true);
        reconstruction.AddImage(std::move(image));
    }
}
