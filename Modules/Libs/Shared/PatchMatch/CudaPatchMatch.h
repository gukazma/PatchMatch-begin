#pragma once
#include <Common/Export.h>
#include <string>
#include <vector>
#include <colmap/mvs/image.h>
#include <colmap/mvs/depth_map.h>
#include <colmap/mvs/normal_map.h>
#include <colmap/mvs/workspace.h>
#include <colmap/util/logging.h>
#include <memory>
namespace GU
{
    class DLL_API CudaPatchMatch
    {
    public:
        struct Options
        {
            // Maximum image size in either dimension.
            int max_image_size = -1;

            // Index of the GPU used for patch match. For multi-GPU usage,
            // you should separate multiple GPU indices by comma, e.g., "0,1,2,3".
            std::string gpu_index = "-1";

            // Depth range in which to randomly sample depth hypotheses.
            double depth_min = -1.0f;
            double depth_max = -1.0f;

            // Half window size to compute NCC photo-consistency cost.
            int window_radius = 5;

            // Number of pixels to skip when computing NCC. For a value of 1, every
            // pixel is used to compute the NCC. For larger values, only every n-th row
            // and column is used and the computation speed thereby increases roughly by
            // a factor of window_step^2. Note that not all combinations of window sizes
            // and steps produce nice results, especially if the step is greather than 2.
            int window_step = 1;

            // Parameters for bilaterally weighted NCC.
            double sigma_spatial = -1;
            double sigma_color = 0.2f;

            // Number of random samples to draw in Monte Carlo sampling.
            int num_samples = 15;

            // Spread of the NCC likelihood function.
            double ncc_sigma = 0.6f;

            // Minimum triangulation angle in degrees.
            double min_triangulation_angle = 1.0f;

            // Spread of the incident angle likelihood function.
            double incident_angle_sigma = 0.9f;

            // Number of coordinate descent iterations. Each iteration consists
            // of four sweeps from left to right, top to bottom, and vice versa.
            int num_iterations = 5;

            // Whether to add a regularized geometric consistency term to the cost
            // function. If true, the `depth_maps` and `normal_maps` must not be null.
            bool geom_consistency = true;

            // The relative weight of the geometric consistency term w.r.t. to
            // the photo-consistency term.
            double geom_consistency_regularizer = 0.3f;

            // Maximum geometric consistency cost in terms of the forward-backward
            // reprojection error in pixels.
            double geom_consistency_max_cost = 3.0f;

            // Whether to enable filtering.
            bool filter = true;

            // Minimum NCC coefficient for pixel to be photo-consistent.
            double filter_min_ncc = 0.1f;

            // Minimum triangulation angle to be stable.
            double filter_min_triangulation_angle = 3.0f;

            // Minimum number of source images have to be consistent
            // for pixel not to be filtered.
            int filter_min_num_consistent = 2;

            // Maximum forward-backward reprojection error for pixel
            // to be geometrically consistent.
            double filter_geom_consistency_max_cost = 1.0f;

            // Cache size in gigabytes for patch match, which keeps the bitmaps, depth
            // maps, and normal maps of this number of images in memory. A higher value
            // leads to less disk access and faster computation, while a lower value
            // leads to reduced memory usage. Note that a single image can consume a lot
            // of memory, if the consistency graph is dense.
            double cache_size = 32.0;

            // Whether to tolerate missing images/maps in the problem setup
            bool allow_missing_files = false;

            // Whether to write the consistency graph.
            bool write_consistency_graph = false;

            void Print() const;
        };
        struct Problem {
            // Index of the reference image.
            int ref_image_idx = -1;

            // Indices of the source images.
            std::vector<int> src_image_idxs;

            // Input images for the photometric consistency term.
            std::vector<colmap::mvs::Image>* images = nullptr;

            // Input depth maps for the geometric consistency term.
            std::vector<colmap::mvs::DepthMap>* depth_maps = nullptr;

            // Input normal maps for the geometric consistency term.
            std::vector<colmap::mvs::NormalMap>* normal_maps = nullptr;

            // Print the configuration to stdout.
            void Print() const;
        };

        CudaPatchMatch(Options options_, Problem problem_);
        ~CudaPatchMatch();

        void Run();

        Options m_options;
        Problem m_problem;
    };
}
