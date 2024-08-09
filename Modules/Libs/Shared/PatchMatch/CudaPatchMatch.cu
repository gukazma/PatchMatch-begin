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
    void CudaPatchMatch::Check() const {
        CHECK(m_options.Check());

        CHECK(!m_options.gpu_index.empty());
        const std::vector<int> gpu_indices = colmap::CSVToVector<int>(m_options.gpu_index);
        CHECK_EQ(gpu_indices.size(), 1);
        CHECK_GE(gpu_indices[0], -1);

        CHECK_NOTNULL(m_problem.images);
        if (m_options.geom_consistency) {
            CHECK_NOTNULL(m_problem.depth_maps);
            CHECK_NOTNULL(m_problem.normal_maps);
            CHECK_EQ(m_problem.depth_maps->size(), m_problem.images->size());
            CHECK_EQ(m_problem.normal_maps->size(), m_problem.images->size());
        }

        CHECK_GT(m_problem.src_image_idxs.size(), 0);

        // Check that there are no duplicate images and that the reference image
        // is not defined as a source image.
        std::set<int> unique_image_idxs(m_problem.src_image_idxs.begin(),
            m_problem.src_image_idxs.end());
        unique_image_idxs.insert(m_problem.ref_image_idx);
        CHECK_EQ(m_problem.src_image_idxs.size() + 1, unique_image_idxs.size());

        // Check that input data is well-formed.
        for (const int image_idx : unique_image_idxs) {
            CHECK_GE(image_idx, 0) << image_idx;
            CHECK_LT(image_idx, m_problem.images->size()) << image_idx;

            const colmap::mvs::Image& image = m_problem.images->at(image_idx);
            CHECK_GT(image.GetBitmap().Width(), 0) << image_idx;
            CHECK_GT(image.GetBitmap().Height(), 0) << image_idx;
            CHECK(image.GetBitmap().IsGrey()) << image_idx;
            CHECK_EQ(image.GetWidth(), image.GetBitmap().Width()) << image_idx;
            CHECK_EQ(image.GetHeight(), image.GetBitmap().Height()) << image_idx;

            // Make sure, the calibration matrix only contains fx, fy, cx, cy.
            CHECK_LT(std::abs(image.GetK()[1] - 0.0f), 1e-6f) << image_idx;
            CHECK_LT(std::abs(image.GetK()[3] - 0.0f), 1e-6f) << image_idx;
            CHECK_LT(std::abs(image.GetK()[6] - 0.0f), 1e-6f) << image_idx;
            CHECK_LT(std::abs(image.GetK()[7] - 0.0f), 1e-6f) << image_idx;
            CHECK_LT(std::abs(image.GetK()[8] - 1.0f), 1e-6f) << image_idx;

            if (m_options.geom_consistency) {
                CHECK_LT(image_idx, m_problem.depth_maps->size()) << image_idx;
                const colmap::mvs::DepthMap& depth_map = m_problem.depth_maps->at(image_idx);
                CHECK_EQ(image.GetWidth(), depth_map.GetWidth()) << image_idx;
                CHECK_EQ(image.GetHeight(), depth_map.GetHeight()) << image_idx;
            }
        }

        if (m_options.geom_consistency) {
            const colmap::mvs::Image& ref_image = m_problem.images->at(m_problem.ref_image_idx);
            const colmap::mvs::NormalMap& ref_normal_map =
                m_problem.normal_maps->at(m_problem.ref_image_idx);
            CHECK_EQ(ref_image.GetWidth(), ref_normal_map.GetWidth());
            CHECK_EQ(ref_image.GetHeight(), ref_normal_map.GetHeight());
        }
    }
    void CudaPatchMatch::Run()
    {
        colmap::PrintHeading2("PatchMatch::Run");
        Check();
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

    bool CudaPatchMatch::Options::Check() const
    {
        using namespace colmap;
        if (depth_min != -1.0f || depth_max != -1.0f) {
            CHECK_OPTION_LE(depth_min, depth_max);
            CHECK_OPTION_GE(depth_min, 0.0f);
        }
        CHECK_OPTION_LE(window_radius,
            static_cast<int>(kMaxPatchMatchWindowRadius));
        CHECK_OPTION_GT(sigma_color, 0.0f);
        CHECK_OPTION_GT(window_radius, 0);
        CHECK_OPTION_GT(window_step, 0);
        CHECK_OPTION_LE(window_step, 2);
        CHECK_OPTION_GT(num_samples, 0);
        CHECK_OPTION_GT(ncc_sigma, 0.0f);
        CHECK_OPTION_GE(min_triangulation_angle, 0.0f);
        CHECK_OPTION_LT(min_triangulation_angle, 180.0f);
        CHECK_OPTION_GT(incident_angle_sigma, 0.0f);
        CHECK_OPTION_GT(num_iterations, 0);
        CHECK_OPTION_GE(geom_consistency_regularizer, 0.0f);
        CHECK_OPTION_GE(geom_consistency_max_cost, 0.0f);
        CHECK_OPTION_GE(filter_min_ncc, -1.0f);
        CHECK_OPTION_LE(filter_min_ncc, 1.0f);
        CHECK_OPTION_GE(filter_min_triangulation_angle, 0.0f);
        CHECK_OPTION_LE(filter_min_triangulation_angle, 180.0f);
        CHECK_OPTION_GE(filter_min_num_consistent, 0);
        CHECK_OPTION_GE(filter_geom_consistency_max_cost, 0.0f);
        CHECK_OPTION_GT(cache_size, 0);
        return true;
    }

}