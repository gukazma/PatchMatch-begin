#include <gtest/gtest.h>
#include <rerun.hpp>

#include <iostream>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rerun.hpp>

#include "collection_adapters.hpp"
#include <colmap/scene/reconstruction.h>
#include <colmap/mvs/patch_match.h>
#include <colmap/controllers/option_manager.h>
#include <colmap/mvs/workspace.h>
#include <colmap/controllers/automatic_reconstruction.h>
#include <PatchMatch/PatchMatch.h>
TEST(PatchMatch, IO)
{
    colmap::Reconstruction reconstruction;
    reconstruction.ReadBinary("C:\\projects\\colmap1\\sparse\\0");
}

TEST(PatchMatch, MyPatchmatch)
{
    GU::PatchMatch patchMatch;
    GU::PatchMatch::Options options;
    options.workingspace = "C:\\projects\\colmap1\\dense\\0";
    patchMatch.Init(options);
    patchMatch.Run();
}


//TEST(PatchMatch, RunPatchMatchStereo)
//{
//    std::string workspace_path = "C:/projects/colmap1/dense/0";
//    std::string workspace_format = "COLMAP";
//    std::string pmvs_option_name = "option-all";
//    std::string config_path;
//    colmap::OptionManager  options;
//    
//    options.AddPatchMatchStereoOptions();
//    options.patch_match_stereo->geom_consistency = true;
//    colmap::mvs::PatchMatchController controller(*options.patch_match_stereo, workspace_path, workspace_format, pmvs_option_name);
//    
//    controller.Start();
//    controller.Wait();
//
//}

std::vector<Eigen::Vector3f> generate_random_points_vector(int num_points) {
    std::vector<Eigen::Vector3f> points(num_points);
    for (auto& point : points) {
        point.setRandom();
    }
    return points;
}

rerun::Collection<rerun::TensorDimension> tensor_shape(const cv::Mat& img) {
    return { img.rows, img.cols, img.channels() };
};
TEST(PatchMatch, RERUN)
{
    std::cout << "Rerun SDK Version: " << rerun::version_string() << std::endl;

    const auto rec = rerun::RecordingStream("rerun_example_cpp");
    rec.spawn().exit_on_failure();

    rec.log_timeless("world", rerun::ViewCoordinates::RIGHT_HAND_Z_UP); // Set an up-axis

    const int num_points = 1000;

    // Points represented by std::vector<Eigen::Vector3f>
    const auto points3d_vector = generate_random_points_vector(1000);
    rec.log("world/points_from_vector", rerun::Points3D(points3d_vector));

    // Points represented by Eigen::Mat3Xf (3xN matrix)
    const Eigen::Matrix3Xf points3d_matrix = Eigen::Matrix3Xf::Random(3, num_points);
    rec.log("world/points_from_matrix", rerun::Points3D(points3d_matrix));

    // Posed pinhole camera:
    rec.log(
        "world/camera",
        rerun::Pinhole::from_focal_length_and_resolution({ 500.0, 500.0 }, { 640.0, 480.0 })
    );

    rec.log(
        "world/camera",
        rerun::Transform3D({ 1,2,0 })
        );

    const Eigen::Vector3f camera_position{ 0.0, -1.0, 0.0 };
    Eigen::Matrix3f camera_orientation;
    // clang-format off
    camera_orientation <<
        +1.0, +0.0, +0.0,
        +0.0, +0.0, +1.0,
        +0.0, -1.0, +0.0;
    // clang-format on
    rec.log(
        "world/camera",
        rerun::Transform3D(
            rerun::Vec3D(camera_position.data()),
            rerun::Mat3x3(camera_orientation.data())
        )
    );

    // Read image
    const auto image_path = "C:/codes/PatchMatch-begin/Datas/Cone/im2.png";
    cv::Mat img = imread(image_path, cv::IMREAD_COLOR);
    EXPECT_FALSE(img.empty());

    // Rerun expects RGB format
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    rec.log("world/camera", rerun::Image(tensor_shape(img), rerun::TensorBuffer::u8(img)));

    // Log image to rerun using the tensor buffer adapter defined in `collection_adapters.hpp`.
    rec.log("image0", rerun::Image(tensor_shape(img), rerun::TensorBuffer::u8(img)));

    // Or by passing a pointer to the image data.
    // The pointer cast here is redundant since `data` is already uint8_t in this case, but if you have e.g. a float image it may be necessary to cast to float*.
    rec.log("image1", rerun::Image(tensor_shape(img), reinterpret_cast<const uint8_t*>(img.data)));


}