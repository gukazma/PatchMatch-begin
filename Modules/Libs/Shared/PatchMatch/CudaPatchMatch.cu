#include "CudaPatchMatch.h"
#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <device_launch_parameters.h>
#include <Eigen/Core>
#include <colmap/math/math.h>
#include <unordered_set>
#include <colmap/util/misc.h>

#define PrintOption(option) LOG(INFO) << #option ": " << option << std::endl
namespace GU
{
    void CudaPatchMatch::Problem::Print() const {
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
    CudaPatchMatch::CudaPatchMatch(Options options_, Problem problem_)
        : m_options(options_), m_problem(problem_)
    {}

    CudaPatchMatch::~CudaPatchMatch()
    {}

    void CudaPatchMatch::Run()
    {
        
    }
    void GU::CudaPatchMatch::Options::Print() const
    {
        colmap::PrintHeading2("PatchMatchOptions");
        PrintOption(max_image_size);
        PrintOption(gpu_index);
        PrintOption(depth_min);
        PrintOption(depth_max);
        PrintOption(window_radius);
        PrintOption(window_step);
        PrintOption(sigma_spatial);
        PrintOption(sigma_color);
        PrintOption(num_samples);
        PrintOption(ncc_sigma);
        PrintOption(min_triangulation_angle);
        PrintOption(incident_angle_sigma);
        PrintOption(num_iterations);
        PrintOption(geom_consistency);
        PrintOption(geom_consistency_regularizer);
        PrintOption(geom_consistency_max_cost);
        PrintOption(filter);
        PrintOption(filter_min_ncc);
        PrintOption(filter_min_triangulation_angle);
        PrintOption(filter_min_num_consistent);
        PrintOption(filter_geom_consistency_max_cost);
        PrintOption(write_consistency_graph);
        PrintOption(allow_missing_files);
    }

}