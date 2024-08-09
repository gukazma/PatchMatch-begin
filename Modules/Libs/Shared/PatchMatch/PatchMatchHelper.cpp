#include "PatchMatchHelper.h"
#include <iostream>
#include <unordered_set>
#include <Eigen/Core>
#include <colmap/math/math.h>
#include <colmap/util/misc.h>
namespace GU
{
	PatchMatchHelper::PatchMatchHelper(CudaPatchMatch::Options options)
		: m_options(options)
	{
		Init(m_options);
	}

	PatchMatchHelper::~PatchMatchHelper()
	{
	}

	void PatchMatchHelper::Start()
	{
        const auto& model = m_workspace->GetModel();
        auto& problem = m_problems.at(0);
        const int gpu_index = 0;
        CHECK_GE(gpu_index, -1);
        const std::string& stereo_folder = m_workspace->GetOptions().stereo_folder;
        const std::string output_type = "geometric";
        const std::string image_name = model.GetImageName(problem.ref_image_idx);
        const std::string file_name =
            colmap::StringPrintf("%s.%s.bin", image_name.c_str(), output_type.c_str());
        const std::string depth_map_path =
            colmap::JoinPaths(m_options.workingspace, stereo_folder, "depth_maps", file_name);
        const std::string normal_map_path =
            colmap::JoinPaths(m_options.workingspace, stereo_folder, "normal_maps", file_name);
        const std::string consistency_graph_path = colmap::JoinPaths(
            m_options.workingspace, stereo_folder, "consistency_graphs", file_name);

        if (colmap::ExistsFile(depth_map_path) && colmap::ExistsFile(normal_map_path)) {
            return;
        }

        colmap::PrintHeading1(colmap::StringPrintf("Processing view %d / %d for %s",
            0,
            m_problems.size(),
            image_name.c_str()));

        auto patch_match_options = m_options;

        if (patch_match_options.depth_min < 0 || patch_match_options.depth_max < 0) {
            patch_match_options.depth_min =
                m_depthRanges.at(problem.ref_image_idx).first;
            patch_match_options.depth_max =
                m_depthRanges.at(problem.ref_image_idx).second;
            CHECK(patch_match_options.depth_min > 0 &&
                patch_match_options.depth_max > 0)
                << " - You must manually set the minimum and maximum depth, since no "
                "sparse model is provided in the workspace.";
        }

        patch_match_options.gpu_index = std::to_string(gpu_index);

        if (patch_match_options.sigma_spatial <= 0.0f) {
            patch_match_options.sigma_spatial = patch_match_options.window_radius;
        }

        std::vector<colmap::mvs::Image> images = model.images;
        std::vector<colmap::mvs::DepthMap> depth_maps;
        std::vector<colmap::mvs::NormalMap> normal_maps;
        if (m_options.geom_consistency) {
            depth_maps.resize(model.images.size());
            normal_maps.resize(model.images.size());
        }

        problem.images = &images;
        problem.depth_maps = &depth_maps;
        problem.normal_maps = &normal_maps;

        {
            // Collect all used images in current problem.
            std::unordered_set<int> used_image_idxs(problem.src_image_idxs.begin(),
                problem.src_image_idxs.end());
            used_image_idxs.insert(problem.ref_image_idx);

            patch_match_options.filter_min_num_consistent =
                std::min(static_cast<int>(used_image_idxs.size()) - 1,
                    patch_match_options.filter_min_num_consistent);

            // Only access workspace from one thread at a time and only spawn resample
            // threads from one master thread at a time.
            //std::unique_lock<std::mutex> lock(workspace_mutex_);

            LOG(INFO) << "Reading inputs...";
            std::vector<int> src_image_idxs;
            for (const auto image_idx : used_image_idxs) {
                std::string image_path = m_workspace->GetBitmapPath(image_idx);
                std::string depth_path = m_workspace->GetDepthMapPath(image_idx);
                std::string normal_path = m_workspace->GetNormalMapPath(image_idx);

                if (!colmap::ExistsFile(image_path) ||
                    (m_options.geom_consistency && !colmap::ExistsFile(depth_path)) ||
                    (m_options.geom_consistency && !colmap::ExistsFile(normal_path))) {
                    if (m_options.allow_missing_files) {
                        LOG(WARNING) << colmap::StringPrintf(
                            "Skipping source image %d: %s for missing "
                            "image or depth/normal map",
                            image_idx,
                            model.GetImageName(image_idx).c_str());
                        continue;
                    }
                    else {
                        LOG(ERROR) << colmap::StringPrintf(
                            "Missing image or map dependency for image %d: %s",
                            image_idx,
                            model.GetImageName(image_idx).c_str());
                    }
                }

                if (image_idx != problem.ref_image_idx) {
                    src_image_idxs.push_back(image_idx);
                }
                images.at(image_idx).SetBitmap(m_workspace->GetBitmap(image_idx));
                if (m_options.geom_consistency) {
                    depth_maps.at(image_idx) = m_workspace->GetDepthMap(image_idx);
                    normal_maps.at(image_idx) = m_workspace->GetNormalMap(image_idx);
                }
            }
            problem.src_image_idxs = src_image_idxs;
        }

        problem.Print();
        patch_match_options.Print();
	}

	void PatchMatchHelper::Init(CudaPatchMatch::Options options_)
	{
		InitWorkspace(options_);
		InitProblems(options_);
	}

	void PatchMatchHelper::InitWorkspace(CudaPatchMatch::Options options_)
	{
		colmap::mvs::Workspace::Options workspace_options;
		workspace_options.max_image_size = 1000;
		workspace_options.image_as_rgb = false;
		workspace_options.cache_size = 32;
		workspace_options.workspace_path = options_.workingspace;
		workspace_options.workspace_format = "COLMAP";
		workspace_options.input_type = "photometric";
		m_workspace = std::make_unique<colmap::mvs::CachedWorkspace>(workspace_options);
		m_depthRanges = m_workspace->GetModel().ComputeDepthRanges();
	}

	void PatchMatchHelper::InitProblems(CudaPatchMatch::Options options_)
	{
        const auto& model = m_workspace->GetModel();

        const std::string config_path = options_.configpath.empty() ? colmap::JoinPaths(options_.workingspace,
            m_workspace->GetOptions().stereo_folder,
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
            CudaPatchMatch::Problem problem;

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
                m_problems.push_back(problem);
            }
        }

        LOG(INFO) << colmap::StringPrintf("Configuration has %d problems...",
            m_problems.size());
	}

}



