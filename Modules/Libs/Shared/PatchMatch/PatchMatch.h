#pragma once
#include <Common/Export.h>
#include <string>
#include <vector>
#include <colmap/mvs/image.h>
#include <colmap/mvs/depth_map.h>
#include <colmap/mvs/normal_map.h>
#include <colmap/mvs/workspace.h>
#include <memory>
namespace GU
{
    class DLL_API PatchMatch
    {
    public:
        struct Options
        {
            std::string workingspace;
            std::string configpath;
            double min_triangulation_angle = 1.0f;
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
        void Init(Options options_);

        void Run();

        Options options_;
        std::unique_ptr<colmap::mvs::CachedWorkspace> workspace_;
        std::vector<Problem> problems_;
    };
}
