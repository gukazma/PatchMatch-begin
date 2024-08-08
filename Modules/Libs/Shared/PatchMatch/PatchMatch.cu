#include "PatchMatch.h"
#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <device_launch_parameters.h>
#include <Eigen/Core>
#include <colmap/math/math.h>
#include <unordered_set>
#include <colmap/util/misc.h>
__global__ void squareKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        output[idx] = input[idx] * input[idx];
    }
}

#define PrintOption(option) LOG(INFO) << #option ": " << option << std::endl
namespace GU
{

    void PatchMatch::Init(Options _options)
    {
        options_ = _options;
        colmap::mvs::Workspace::Options workspace_options;
        workspace_options.max_image_size = 1000;
        workspace_options.image_as_rgb = false;
        workspace_options.cache_size = 32;
        workspace_options.workspace_path = options_.workingspace;
        workspace_options.workspace_format = "COLMAP";
        workspace_options.input_type = "photometric";

        workspace_ = std::make_unique<colmap::mvs::CachedWorkspace>(workspace_options);
        auto image = workspace_->GetModel().images[0];
        auto depth_ranges_ = workspace_->GetModel().ComputeDepthRanges();


        const auto& model = workspace_->GetModel();

        const std::string config_path = options_.configpath.empty() ? colmap::JoinPaths(options_.workingspace,
            workspace_->GetOptions().stereo_folder,
            "patch-match.cfg") : options_.configpath;
        std::vector<std::string> config = colmap::ReadTextFileLines(config_path);

        std::vector<std::map<int, int>> shared_num_points;
        std::vector<std::map<int, float>> triangulation_angles;

        const float min_triangulation_angle_rad =
            colmap::DegToRad(options_.min_triangulation_angle);

        std::string ref_image_name;
        std::unordered_set<int> ref_image_idxs;

        struct ProblemConfig {
            std::string ref_image_name;
            std::vector<std::string> src_image_names;
        };
        std::vector<ProblemConfig> problem_configs;

        for (size_t i = 0; i < config.size(); ++i) {
            std::string& config_line = config[i];
            colmap::StringTrim(&config_line);

            if (config_line.empty() || config_line[0] == '#') {
                continue;
            }

            if (ref_image_name.empty()) {
                ref_image_name = config_line;
                continue;
            }

            ref_image_idxs.insert(model.GetImageIdx(ref_image_name));

            ProblemConfig problem_config;
            problem_config.ref_image_name = ref_image_name;
            problem_config.src_image_names = colmap::CSVToVector<std::string>(config_line);
            problem_configs.push_back(problem_config);

            ref_image_name.clear();
        }

        for (const auto& problem_config : problem_configs) {
            Problem problem;

            problem.ref_image_idx = model.GetImageIdx(problem_config.ref_image_name);

            if (problem_config.src_image_names.size() == 1 &&
                problem_config.src_image_names[0] == "__all__") {
                // Use all images as source images.
                problem.src_image_idxs.clear();
                problem.src_image_idxs.reserve(model.images.size() - 1);
                for (size_t image_idx = 0; image_idx < model.images.size(); ++image_idx) {
                    if (static_cast<int>(image_idx) != problem.ref_image_idx) {
                        problem.src_image_idxs.push_back(image_idx);
                    }
                }
            }
            else if (problem_config.src_image_names.size() == 2 &&
                problem_config.src_image_names[0] == "__auto__") {
                // Use maximum number of overlapping images as source images. Overlapping
                // will be sorted based on the number of shared points to the reference
                // image and the top ranked images are selected. Note that images are only
                // selected if some points have a sufficient triangulation angle.

                if (shared_num_points.empty()) {
                    shared_num_points = model.ComputeSharedPoints();
                }
                if (triangulation_angles.empty()) {
                    const float kTriangulationAnglePercentile = 75;
                    triangulation_angles =
                        model.ComputeTriangulationAngles(kTriangulationAnglePercentile);
                }

                const size_t max_num_src_images =
                    std::stoll(problem_config.src_image_names[1]);

                const auto& overlapping_images =
                    shared_num_points.at(problem.ref_image_idx);
                const auto& overlapping_triangulation_angles =
                    triangulation_angles.at(problem.ref_image_idx);

                std::vector<std::pair<int, int>> src_images;
                src_images.reserve(overlapping_images.size());
                for (const auto& image : overlapping_images) {
                    if (overlapping_triangulation_angles.at(image.first) >=
                        min_triangulation_angle_rad) {
                        src_images.emplace_back(image.first, image.second);
                    }
                }

                const size_t eff_max_num_src_images =
                    std::min(src_images.size(), max_num_src_images);

                std::partial_sort(src_images.begin(),
                    src_images.begin() + eff_max_num_src_images,
                    src_images.end(),
                    [](const std::pair<int, int>& image1,
                        const std::pair<int, int>& image2) {
                            return image1.second > image2.second;
                    });

                problem.src_image_idxs.reserve(eff_max_num_src_images);
                for (size_t i = 0; i < eff_max_num_src_images; ++i) {
                    problem.src_image_idxs.push_back(src_images[i].first);
                }
            }
            else {
                problem.src_image_idxs.reserve(problem_config.src_image_names.size());
                for (const auto& src_image_name : problem_config.src_image_names) {
                    problem.src_image_idxs.push_back(model.GetImageIdx(src_image_name));
                }
            }

            if (problem.src_image_idxs.empty()) {
                LOG(WARNING) << colmap::StringPrintf(
                    "Ignoring reference image %s, because it has no "
                    "source images.",
                    problem_config.ref_image_name.c_str());
            }
            else {
                problems_.push_back(problem);
            }
        }

        LOG(INFO) << colmap::StringPrintf("Configuration has %d problems...",
            problems_.size());
    }
    void PatchMatch::Problem::Print() const {
        colmap::PrintHeading2("PatchMatch::Problem");

        PrintOption(ref_image_idx);

        LOG(INFO) << "src_image_idxs: ";
        if (!src_image_idxs.empty()) {
            for (size_t i = 0; i < src_image_idxs.size() - 1; ++i) {
                LOG(INFO) << src_image_idxs[i] << " ";
            }
            LOG(INFO) << src_image_idxs.back();
        }
        else {
        }
    }
    void PatchMatch::Run()
    {

        const auto& model = workspace_->GetModel();

        auto& problem = problems_.at(0);
        const int gpu_index = 0;
        CHECK_GE(gpu_index, -1);

        const std::string& stereo_folder = workspace_->GetOptions().stereo_folder;
        const std::string output_type = "geometric";
        const std::string image_name = model.GetImageName(problem.ref_image_idx);
        const std::string file_name =
            colmap::StringPrintf("%s.%s.bin", image_name.c_str(), output_type.c_str());
        const std::string depth_map_path =
            colmap::JoinPaths(options_.workingspace, stereo_folder, "depth_maps", file_name);
        const std::string normal_map_path =
            colmap::JoinPaths(options_.workingspace, stereo_folder, "normal_maps", file_name);
        const std::string consistency_graph_path = colmap::JoinPaths(
            options_.workingspace, stereo_folder, "consistency_graphs", file_name);

        if (colmap::ExistsFile(depth_map_path) && colmap::ExistsFile(normal_map_path)) {
            return;
        }

        const int size = 10;
        float* h_input, * h_output;  // Host arrays
        float* d_input, * d_output;  // Device arrays

        // Allocate memory on the host
        h_input = (float*)malloc(size * sizeof(float));
        h_output = (float*)malloc(size * sizeof(float));

        // Initialize input data
        for (int i = 0; i < size; i++) {
            h_input[i] = i;
        }

        // Allocate memory on the device
        cudaMalloc(&d_input, size * sizeof(float));
        cudaMalloc(&d_output, size * sizeof(float));

        // Copy input data from host to device
        cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

        // Launch the kernel
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        squareKernel << <numBlocks, blockSize >> > (d_input, d_output, size);

        // Copy the result back from device to host
        cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);

        // Print the result
        for (int i = 0; i < size; i++) {
            printf("%f ", h_output[i]);
        }

        // Free memory
        free(h_input);
        free(h_output);
        cudaFree(d_input);
        cudaFree(d_output);
    }

}