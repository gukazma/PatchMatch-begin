#include "CudaPatchMatch.h"
#include <iostream>
#include <unordered_set>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <Eigen/Core>
#include <colmap/math/math.h>
#include <colmap/util/misc.h>
#include <colmap/util/cuda.h>
#include <colmap/util/cudacc.h>
#include <colmap/mvs/gpu_mat_ref_image.h>
#include <colmap/mvs/gpu_mat.h>
#include <colmap/mvs/gpu_mat_prng.h>

#define PrintOption(option) LOG(INFO) << #option ": " << option << std::endl
namespace GU
{
    using namespace colmap;
    using namespace colmap::mvs;
    // Calibration of reference image as {fx, cx, fy, cy}.
    __constant__ float ref_K[4];
    // Calibration of reference image as {1/fx, -cx/fx, 1/fy, -cy/fy}.
    __constant__ float ref_inv_K[4];


    __device__ inline float DotProduct3(const float vec1[3], const float vec2[3]) {
        return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2];
    }

    __device__ inline void GenerateRandomNormal(const int row,
        const int col,
        curandState* rand_state,
        float normal[3]) {
        // Unbiased sampling of normal, according to George Marsaglia, "Choosing a
        // Point from the Surface of a Sphere", 1972.
        float v1 = 0.0f;
        float v2 = 0.0f;
        float s = 2.0f;
        while (s >= 1.0f) {
            v1 = 2.0f * curand_uniform(rand_state) - 1.0f;
            v2 = 2.0f * curand_uniform(rand_state) - 1.0f;
            s = v1 * v1 + v2 * v2;
        }

        const float s_norm = sqrt(1.0f - s);
        normal[0] = 2.0f * v1 * s_norm;
        normal[1] = 2.0f * v2 * s_norm;
        normal[2] = 1.0f - 2.0f * s;

        // Make sure normal is looking away from camera.
        const float view_ray[3] = { ref_inv_K[0] * col + ref_inv_K[1],
                                   ref_inv_K[2] * row + ref_inv_K[3],
                                   1.0f };
        if (DotProduct3(normal, view_ray) > 0) {
            normal[0] = -normal[0];
            normal[1] = -normal[1];
            normal[2] = -normal[2];
        }
    }

    // Rotate normals by 90deg around z-axis in counter-clockwise direction.
    __global__ void InitNormalMap(GpuMat<float> normal_map,
        GpuMat<curandState> rand_state_map) {
        const int row = blockDim.y * blockIdx.y + threadIdx.y;
        const int col = blockDim.x * blockIdx.x + threadIdx.x;
        if (col < normal_map.GetWidth() && row < normal_map.GetHeight()) {
            curandState rand_state = rand_state_map.Get(row, col);
            float normal[3];
            GenerateRandomNormal(row, col, &rand_state, normal);
            normal_map.SetSlice(row, col, normal);
            rand_state_map.Set(row, col, rand_state);
        }
    }

    void CudaPatchMatch::Problem::Print() const {
        PrintHeading2("PatchMatch::Problem");

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
        : options_(options_),
        problem_(problem_),
        ref_width_(0),
        ref_height_(0),
        rotation_in_half_pi_(0) {
        colmap::SetBestCudaDevice(std::stoi(options_.gpu_index));
        InitRefImage();
        InitSourceImages();
        InitTransforms();
        InitWorkspaceMemory();
    }

    CudaPatchMatch::~CudaPatchMatch()
    {}
    void CudaPatchMatch::Check() const {
        CHECK(options_.Check());

        CHECK(!options_.gpu_index.empty());
        const std::vector<int> gpu_indices = colmap::CSVToVector<int>(options_.gpu_index);
        CHECK_EQ(gpu_indices.size(), 1);
        CHECK_GE(gpu_indices[0], -1);

        CHECK_NOTNULL(problem_.images);
        if (options_.geom_consistency) {
            CHECK_NOTNULL(problem_.depth_maps);
            CHECK_NOTNULL(problem_.normal_maps);
            CHECK_EQ(problem_.depth_maps->size(), problem_.images->size());
            CHECK_EQ(problem_.normal_maps->size(), problem_.images->size());
        }

        CHECK_GT(problem_.src_image_idxs.size(), 0);

        // Check that there are no duplicate images and that the reference image
        // is not defined as a source image.
        std::set<int> unique_image_idxs(problem_.src_image_idxs.begin(),
            problem_.src_image_idxs.end());
        unique_image_idxs.insert(problem_.ref_image_idx);
        CHECK_EQ(problem_.src_image_idxs.size() + 1, unique_image_idxs.size());

        // Check that input data is well-formed.
        for (const int image_idx : unique_image_idxs) {
            CHECK_GE(image_idx, 0) << image_idx;
            CHECK_LT(image_idx, problem_.images->size()) << image_idx;

            const colmap::mvs::Image& image = problem_.images->at(image_idx);
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

            if (options_.geom_consistency) {
                CHECK_LT(image_idx, problem_.depth_maps->size()) << image_idx;
                const colmap::mvs::DepthMap& depth_map = problem_.depth_maps->at(image_idx);
                CHECK_EQ(image.GetWidth(), depth_map.GetWidth()) << image_idx;
                CHECK_EQ(image.GetHeight(), depth_map.GetHeight()) << image_idx;
            }
        }

        if (options_.geom_consistency) {
            const colmap::mvs::Image& ref_image = problem_.images->at(problem_.ref_image_idx);
            const colmap::mvs::NormalMap& ref_normal_map =
                problem_.normal_maps->at(problem_.ref_image_idx);
            CHECK_EQ(ref_image.GetWidth(), ref_normal_map.GetWidth());
            CHECK_EQ(ref_image.GetHeight(), ref_normal_map.GetHeight());
        }
    }
    void CudaPatchMatch::ComputeCudaConfig()
    {
    }
    void CudaPatchMatch::BindRefImageTexture()
    {
        cudaTextureDesc texture_desc;
        memset(&texture_desc, 0, sizeof(texture_desc));
        texture_desc.addressMode[0] = cudaAddressModeBorder;
        texture_desc.addressMode[1] = cudaAddressModeBorder;
        texture_desc.addressMode[2] = cudaAddressModeBorder;
        texture_desc.filterMode = cudaFilterModePoint;
        texture_desc.readMode = cudaReadModeNormalizedFloat;
        texture_desc.normalizedCoords = false;
        ref_image_texture_ = colmap::mvs::CudaArrayLayeredTexture<uint8_t>::FromGpuMat(
            texture_desc, *ref_image_->image);
    }
    void CudaPatchMatch::InitRefImage()
    {
        const colmap::mvs::Image& ref_image = problem_.images->at(problem_.ref_image_idx);

        ref_width_ = ref_image.GetWidth();
        ref_height_ = ref_image.GetHeight();

        // Upload to device and filter.
        ref_image_.reset(new colmap::mvs::GpuMatRefImage(ref_width_, ref_height_));
        const std::vector<uint8_t> ref_image_array =
            ref_image.GetBitmap().ConvertToRowMajorArray();
        ref_image_->Filter(ref_image_array.data(),
            options_.window_radius,
            options_.window_step,
            options_.sigma_spatial,
            options_.sigma_color);

        BindRefImageTexture();
    }
    void CudaPatchMatch::InitSourceImages()
    {
        // Determine maximum image size.
        size_t max_width = 0;
        size_t max_height = 0;
        for (const auto image_idx : problem_.src_image_idxs) {
            const colmap::mvs::Image& image = problem_.images->at(image_idx);
            if (image.GetWidth() > max_width) {
                max_width = image.GetWidth();
            }
            if (image.GetHeight() > max_height) {
                max_height = image.GetHeight();
            }
        }

        // Upload source images to device.
        {
            // Copy source images to contiguous memory block.
            const uint8_t kDefaultValue = 0;
            std::vector<uint8_t> src_images_host_data(
                static_cast<size_t>(max_width * max_height *
                    problem_.src_image_idxs.size()),
                kDefaultValue);
            for (size_t i = 0; i < problem_.src_image_idxs.size(); ++i) {
                const colmap::mvs::Image& image = problem_.images->at(problem_.src_image_idxs[i]);
                const colmap::Bitmap& bitmap = image.GetBitmap();
                uint8_t* dest = src_images_host_data.data() + max_width * max_height * i;
                for (size_t r = 0; r < image.GetHeight(); ++r) {
                    memcpy(dest, bitmap.GetScanline(r), image.GetWidth() * sizeof(uint8_t));
                    dest += max_width;
                }
            }

            // Create source images texture.
            cudaTextureDesc texture_desc;
            memset(&texture_desc, 0, sizeof(texture_desc));
            texture_desc.addressMode[0] = cudaAddressModeBorder;
            texture_desc.addressMode[1] = cudaAddressModeBorder;
            texture_desc.addressMode[2] = cudaAddressModeBorder;
            texture_desc.filterMode = cudaFilterModeLinear;
            texture_desc.readMode = cudaReadModeNormalizedFloat;
            texture_desc.normalizedCoords = false;
            src_images_texture_ = colmap::mvs::CudaArrayLayeredTexture<uint8_t>::FromHostArray(
                texture_desc,
                max_width,
                max_height,
                problem_.src_image_idxs.size(),
                src_images_host_data.data());
        }

        // Upload source depth maps to device.
        if (options_.geom_consistency) {
            const float kDefaultValue = 0.0f;
            std::vector<float> src_depth_maps_host_data(
                static_cast<size_t>(max_width * max_height *
                    problem_.src_image_idxs.size()),
                kDefaultValue);
            for (size_t i = 0; i < problem_.src_image_idxs.size(); ++i) {
                const colmap::mvs::DepthMap& depth_map =
                    problem_.depth_maps->at(problem_.src_image_idxs[i]);
                float* dest =
                    src_depth_maps_host_data.data() + max_width * max_height * i;
                for (size_t r = 0; r < depth_map.GetHeight(); ++r) {
                    memcpy(dest,
                        depth_map.GetPtr() + r * depth_map.GetWidth(),
                        depth_map.GetWidth() * sizeof(float));
                    dest += max_width;
                }
            }

            // Create source depth maps texture.
            cudaTextureDesc texture_desc;
            memset(&texture_desc, 0, sizeof(texture_desc));
            texture_desc.addressMode[0] = cudaAddressModeBorder;
            texture_desc.addressMode[1] = cudaAddressModeBorder;
            texture_desc.addressMode[2] = cudaAddressModeBorder;
            texture_desc.filterMode = cudaFilterModePoint;
            texture_desc.readMode = cudaReadModeElementType;
            texture_desc.normalizedCoords = false;
            src_depth_maps_texture_ = colmap::mvs::CudaArrayLayeredTexture<float>::FromHostArray(
                texture_desc,
                max_width,
                max_height,
                problem_.src_image_idxs.size(),
                src_depth_maps_host_data.data());
        }
    }
    void CudaPatchMatch::InitTransforms()
    {
        using namespace colmap;
        const colmap::mvs::Image& ref_image = problem_.images->at(problem_.ref_image_idx);

        //////////////////////////////////////////////////////////////////////////////
        // Generate rotated versions (counter-clockwise) of calibration matrix.
        //////////////////////////////////////////////////////////////////////////////

        for (size_t i = 0; i < 4; ++i) {
            ref_K_host_[i][0] = ref_image.GetK()[0];
            ref_K_host_[i][1] = ref_image.GetK()[2];
            ref_K_host_[i][2] = ref_image.GetK()[4];
            ref_K_host_[i][3] = ref_image.GetK()[5];
        }

        // Rotated by 90 degrees.
        std::swap(ref_K_host_[1][0], ref_K_host_[1][2]);
        std::swap(ref_K_host_[1][1], ref_K_host_[1][3]);
        ref_K_host_[1][3] = ref_width_ - 1 - ref_K_host_[1][3];

        // Rotated by 180 degrees.
        ref_K_host_[2][1] = ref_width_ - 1 - ref_K_host_[2][1];
        ref_K_host_[2][3] = ref_height_ - 1 - ref_K_host_[2][3];

        // Rotated by 270 degrees.
        std::swap(ref_K_host_[3][0], ref_K_host_[3][2]);
        std::swap(ref_K_host_[3][1], ref_K_host_[3][3]);
        ref_K_host_[3][1] = ref_height_ - 1 - ref_K_host_[3][1];

        // Extract 1/fx, -cx/fx, fy, -cy/fy.
        for (size_t i = 0; i < 4; ++i) {
            ref_inv_K_host_[i][0] = 1.0f / ref_K_host_[i][0];
            ref_inv_K_host_[i][1] = -ref_K_host_[i][1] / ref_K_host_[i][0];
            ref_inv_K_host_[i][2] = 1.0f / ref_K_host_[i][2];
            ref_inv_K_host_[i][3] = -ref_K_host_[i][3] / ref_K_host_[i][2];
        }

        // Bind 0 degrees version to constant global memory.
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(
            ref_K, ref_K_host_[0], sizeof(float) * 4, 0, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpyToSymbol(ref_inv_K,
            ref_inv_K_host_[0],
            sizeof(float) * 4,
            0,
            cudaMemcpyHostToDevice));

        //////////////////////////////////////////////////////////////////////////////
        // Generate rotated versions of camera poses.
        //////////////////////////////////////////////////////////////////////////////

        float rotated_R[9];
        memcpy(rotated_R, ref_image.GetR(), 9 * sizeof(float));

        float rotated_T[3];
        memcpy(rotated_T, ref_image.GetT(), 3 * sizeof(float));

        // Matrix for 90deg rotation around Z-axis in counter-clockwise direction.
        const float R_z90[9] = { 0, 1, 0, -1, 0, 0, 0, 0, 1 };

        cudaTextureDesc texture_desc;
        memset(&texture_desc, 0, sizeof(texture_desc));
        texture_desc.addressMode[0] = cudaAddressModeBorder;
        texture_desc.addressMode[1] = cudaAddressModeBorder;
        texture_desc.addressMode[2] = cudaAddressModeBorder;
        texture_desc.filterMode = cudaFilterModePoint;
        texture_desc.readMode = cudaReadModeElementType;
        texture_desc.normalizedCoords = false;

        for (size_t i = 0; i < 4; ++i) {
            const size_t kNumTformParams = 4 + 9 + 3 + 3 + 12 + 12;
            std::vector<float> poses_host_data(kNumTformParams *
                problem_.src_image_idxs.size());
            int offset = 0;
            for (const auto image_idx : problem_.src_image_idxs) {
                const colmap::mvs::Image& image = problem_.images->at(image_idx);

                const float K[4] = {
                    image.GetK()[0], image.GetK()[2], image.GetK()[4], image.GetK()[5] };
                memcpy(poses_host_data.data() + offset, K, 4 * sizeof(float));
                offset += 4;

                float rel_R[9];
                float rel_T[3];
                colmap::mvs::ComputeRelativePose(
                    rotated_R, rotated_T, image.GetR(), image.GetT(), rel_R, rel_T);
                memcpy(poses_host_data.data() + offset, rel_R, 9 * sizeof(float));
                offset += 9;
                memcpy(poses_host_data.data() + offset, rel_T, 3 * sizeof(float));
                offset += 3;

                float C[3];
                colmap::mvs::ComputeProjectionCenter(rel_R, rel_T, C);
                memcpy(poses_host_data.data() + offset, C, 3 * sizeof(float));
                offset += 3;

                float P[12];
                colmap::mvs::ComposeProjectionMatrix(image.GetK(), rel_R, rel_T, P);
                memcpy(poses_host_data.data() + offset, P, 12 * sizeof(float));
                offset += 12;

                float inv_P[12];
                colmap::mvs::ComposeInverseProjectionMatrix(image.GetK(), rel_R, rel_T, inv_P);
                memcpy(poses_host_data.data() + offset, inv_P, 12 * sizeof(float));
                offset += 12;
            }

            poses_texture_[i] = colmap::mvs::CudaArrayLayeredTexture<float>::FromHostArray(
                texture_desc,
                kNumTformParams,
                problem_.src_image_idxs.size(),
                1,
                poses_host_data.data());

            colmap::mvs::RotatePose(R_z90, rotated_R, rotated_T);
        }
    }
    void CudaPatchMatch::InitWorkspaceMemory()
    {
        using namespace colmap;
        using namespace colmap::mvs;
        rand_state_map_.reset(new GpuMatPRNG(ref_width_, ref_height_));

        depth_map_.reset(new GpuMat<float>(ref_width_, ref_height_));
        if (options_.geom_consistency) {
            const DepthMap& init_depth_map =
                problem_.depth_maps->at(problem_.ref_image_idx);
            depth_map_->CopyToDevice(init_depth_map.GetPtr(),
                init_depth_map.GetWidth() * sizeof(float));
        }
        else {
            depth_map_->FillWithRandomNumbers(
                options_.depth_min, options_.depth_max, *rand_state_map_);
        }

        normal_map_.reset(new GpuMat<float>(ref_width_, ref_height_, 3));

        // Note that it is not necessary to keep the selection probability map in
        // memory for all pixels. Theoretically, it is possible to incorporate
        // the temporary selection probabilities in the global_workspace_.
        // However, it is useful to keep the probabilities for the entire image
        // in memory, so that it can be exported.
        sel_prob_map_.reset(new GpuMat<float>(
            ref_width_, ref_height_, problem_.src_image_idxs.size()));
        prev_sel_prob_map_.reset(new GpuMat<float>(
            ref_width_, ref_height_, problem_.src_image_idxs.size()));
        prev_sel_prob_map_->FillWithScalar(0.5f);

        cost_map_.reset(new GpuMat<float>(
            ref_width_, ref_height_, problem_.src_image_idxs.size()));

        const int ref_max_dim = std::max(ref_width_, ref_height_);
        global_workspace_.reset(
            new GpuMat<float>(ref_max_dim, problem_.src_image_idxs.size(), 2));

        consistency_mask_.reset(new GpuMat<uint8_t>(0, 0, 0));

        ComputeCudaConfig();

        if (options_.geom_consistency) {
            const NormalMap& init_normal_map =
                problem_.normal_maps->at(problem_.ref_image_idx);
            normal_map_->CopyToDevice(init_normal_map.GetPtr(),
                init_normal_map.GetWidth() * sizeof(float));
        }
        else {
            InitNormalMap << <elem_wise_grid_size_, elem_wise_block_size_ >> > (
                *normal_map_, *rand_state_map_);
        }
    }
    void CudaPatchMatch::Rotate()
    {
    }
    void CudaPatchMatch::Run()
    {
        colmap::PrintHeading2("PatchMatch::Run");

        Check();
    }
    void CudaPatchMatch::Options::Print() const
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