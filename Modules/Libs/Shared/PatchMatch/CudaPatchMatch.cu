#include "CudaPatchMatch.h"
#include <iostream>
#include <unordered_set>
#define __CUDACC__
#include <cuda_runtime.h>
#include <texture_indirect_functions.h>
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

// The number of threads per Cuda thread. Warning: Do not change this value,
// since the templated window sizes rely on this value.
#define THREADS_PER_BLOCK 32

// We must not include "util/math.h" to avoid any Eigen includes here,
// since Visual Studio cannot compile some of the Eigen/Boost expressions.
#ifndef DEG2RAD
#define DEG2RAD(deg) deg * 0.0174532925199432
#endif
namespace GU
{
    using namespace colmap;
    using namespace colmap::mvs;
    // Calibration of reference image as {fx, cx, fy, cy}.
    __constant__ float ref_K[4];
    // Calibration of reference image as {1/fx, -cx/fx, 1/fy, -cy/fy}.
    __constant__ float ref_inv_K[4];

    __device__ inline void ComposeHomography(
        const cudaTextureObject_t poses_texture,
        const int image_idx,
        const int row,
        const int col,
        const float depth,
        const float normal[3],
        float H[9]) {
        // Calibration of source image.
        float K[4];
        for (int i = 0; i < 4; ++i) {
            K[i] = tex2D<float>(poses_texture, i, image_idx);
        }

        // Relative rotation between reference and source image.
        float R[9];
        for (int i = 0; i < 9; ++i) {
            R[i] = tex2D<float>(poses_texture, i + 4, image_idx);
        }

        // Relative translation between reference and source image.
        float T[3];
        for (int i = 0; i < 3; ++i) {
            T[i] = tex2D<float>(poses_texture, i + 13, image_idx);
        }

        // Distance to the plane.
        const float dist =
            depth * (normal[0] * (ref_inv_K[0] * col + ref_inv_K[1]) +
                normal[1] * (ref_inv_K[2] * row + ref_inv_K[3]) + normal[2]);
        const float inv_dist = 1.0f / dist;

        const float inv_dist_N0 = inv_dist * normal[0];
        const float inv_dist_N1 = inv_dist * normal[1];
        const float inv_dist_N2 = inv_dist * normal[2];

        // Homography as H = K * (R - T * n' / d) * Kref^-1.
        H[0] = ref_inv_K[0] * (K[0] * (R[0] + inv_dist_N0 * T[0]) +
            K[1] * (R[6] + inv_dist_N0 * T[2]));
        H[1] = ref_inv_K[2] * (K[0] * (R[1] + inv_dist_N1 * T[0]) +
            K[1] * (R[7] + inv_dist_N1 * T[2]));
        H[2] = K[0] * (R[2] + inv_dist_N2 * T[0]) +
            K[1] * (R[8] + inv_dist_N2 * T[2]) +
            ref_inv_K[1] * (K[0] * (R[0] + inv_dist_N0 * T[0]) +
                K[1] * (R[6] + inv_dist_N0 * T[2])) +
            ref_inv_K[3] * (K[0] * (R[1] + inv_dist_N1 * T[0]) +
                K[1] * (R[7] + inv_dist_N1 * T[2]));
        H[3] = ref_inv_K[0] * (K[2] * (R[3] + inv_dist_N0 * T[1]) +
            K[3] * (R[6] + inv_dist_N0 * T[2]));
        H[4] = ref_inv_K[2] * (K[2] * (R[4] + inv_dist_N1 * T[1]) +
            K[3] * (R[7] + inv_dist_N1 * T[2]));
        H[5] = K[2] * (R[5] + inv_dist_N2 * T[1]) +
            K[3] * (R[8] + inv_dist_N2 * T[2]) +
            ref_inv_K[1] * (K[2] * (R[3] + inv_dist_N0 * T[1]) +
                K[3] * (R[6] + inv_dist_N0 * T[2])) +
            ref_inv_K[3] * (K[2] * (R[4] + inv_dist_N1 * T[1]) +
                K[3] * (R[7] + inv_dist_N1 * T[2]));
        H[6] = ref_inv_K[0] * (R[6] + inv_dist_N0 * T[2]);
        H[7] = ref_inv_K[2] * (R[7] + inv_dist_N1 * T[2]);
        H[8] = R[8] + ref_inv_K[1] * (R[6] + inv_dist_N0 * T[2]) +
            ref_inv_K[3] * (R[7] + inv_dist_N1 * T[2]) + inv_dist_N2 * T[2];
    }


    // Each thread in the current warp / thread block reads in 3 columns of the
    // reference image. The shared memory holds 3 * THREADS_PER_BLOCK columns and
    // kWindowSize rows of the reference image. Each thread copies every
    // THREADS_PER_BLOCK-th column from global to shared memory offset by its ID.
    // For example, if THREADS_PER_BLOCK = 32, then thread 0 reads columns 0, 32, 64
    // and thread 1 columns 1, 33, 65. When computing the photoconsistency, which is
    // shared among each thread block, each thread can then read the reference image
    // colors from shared memory. Note that this limits the window radius to a
    // maximum of THREADS_PER_BLOCK.
    template <int kWindowSize>
    struct LocalRefImage {
        const static int kWindowRadius = kWindowSize / 2;
        const static int kThreadBlockRadius = 1;
        const static int kThreadBlockSize = 2 * kThreadBlockRadius + 1;
        const static int kNumRows = kWindowSize;
        const static int kNumColumns = kThreadBlockSize * THREADS_PER_BLOCK;
        const static int kDataSize = kNumRows * kNumColumns;

        __device__ explicit LocalRefImage(const cudaTextureObject_t ref_image_texture)
            : ref_image_texture_(ref_image_texture) {}

        float* data = nullptr;

        __device__ inline void Read(const int row) {
            // For the first row, read the entire block into shared memory. For all
            // consecutive rows, it is only necessary to shift the rows in shared memory
            // up by one element and then read in a new row at the bottom of the shared
            // memory. Note that this assumes that the calling loop starts with the
            // first row and then consecutively reads in the next row.

            const int thread_id = threadIdx.x;
            const int thread_block_first_id = blockDim.x * blockIdx.x;

            const int local_col_start = thread_id;
            const int global_col_start = thread_block_first_id -
                kThreadBlockRadius * THREADS_PER_BLOCK +
                thread_id;

            if (row == 0) {
                int global_row = row - kWindowRadius;
                for (int local_row = 0; local_row < kNumRows; ++local_row, ++global_row) {
                    int local_col = local_col_start;
                    int global_col = global_col_start;
#pragma unroll
                    for (int block = 0; block < kThreadBlockSize; ++block) {
                        data[local_row * kNumColumns + local_col] =
                            tex2D<float>(ref_image_texture_, global_col, global_row);
                        local_col += THREADS_PER_BLOCK;
                        global_col += THREADS_PER_BLOCK;
                    }
                }
            }
            else {
                // Move rows in shared memory up by one row.
                for (int local_row = 1; local_row < kNumRows; ++local_row) {
                    int local_col = local_col_start;
#pragma unroll
                    for (int block = 0; block < kThreadBlockSize; ++block) {
                        data[(local_row - 1) * kNumColumns + local_col] =
                            data[local_row * kNumColumns + local_col];
                        local_col += THREADS_PER_BLOCK;
                    }
                }

                // Read next row into the last row of shared memory.
                const int local_row = kNumRows - 1;
                const int global_row = row + kWindowRadius;
                int local_col = local_col_start;
                int global_col = global_col_start;
#pragma unroll
                for (int block = 0; block < kThreadBlockSize; ++block) {
                    data[local_row * kNumColumns + local_col] =
                        tex2D<float>(ref_image_texture_, global_col, global_row);
                    local_col += THREADS_PER_BLOCK;
                    global_col += THREADS_PER_BLOCK;
                }
            }
        }

    private:
        const cudaTextureObject_t ref_image_texture_;
    };

    template <int kWindowSize, int kWindowStep>
    struct PhotoConsistencyCostComputer {
        const static int kWindowRadius = kWindowSize / 2;

        __device__ PhotoConsistencyCostComputer(
            const cudaTextureObject_t ref_image_texture,
            const cudaTextureObject_t src_images_texture,
            const cudaTextureObject_t poses_texture,
            const float sigma_spatial,
            const float sigma_color)
            : local_ref_image(ref_image_texture),
            src_images_texture_(src_images_texture),
            poses_texture_(poses_texture),
            bilateral_weight_computer_(sigma_spatial, sigma_color) {}

        // Maximum photo consistency cost as 1 - min(NCC).
        const float kMaxCost = 2.0f;

        // Thread warp local reference image data around current patch.
        typedef LocalRefImage<kWindowSize> LocalRefImageType;
        LocalRefImageType local_ref_image;

        // Precomputed sum of raw and squared image intensities.
        float local_ref_sum = 0.0f;
        float local_ref_squared_sum = 0.0f;

        // Index of source image.
        int src_image_idx = -1;

        // Center position of patch in reference image.
        int row = -1;
        int col = -1;

        // Depth and normal for which to warp patch.
        float depth = 0.0f;
        const float* normal = nullptr;

        __device__ inline void Read(const int row) {
            local_ref_image.Read(row);
            __syncthreads();
        }

        __device__ inline float Compute() const {
            float tform[9];
            ComposeHomography(
                poses_texture_, src_image_idx, row, col, depth, normal, tform);

            float tform_step[8];
            for (int i = 0; i < 8; ++i) {
                tform_step[i] = kWindowStep * tform[i];
            }

            const int thread_id = threadIdx.x;
            const int row_start = row - kWindowRadius;
            const int col_start = col - kWindowRadius;

            float col_src = tform[0] * col_start + tform[1] * row_start + tform[2];
            float row_src = tform[3] * col_start + tform[4] * row_start + tform[5];
            float z = tform[6] * col_start + tform[7] * row_start + tform[8];
            float base_col_src = col_src;
            float base_row_src = row_src;
            float base_z = z;

            int ref_image_idx = THREADS_PER_BLOCK - kWindowRadius + thread_id;
            int ref_image_base_idx = ref_image_idx;

            const float ref_center_color =
                local_ref_image
                .data[ref_image_idx + kWindowRadius * 3 * THREADS_PER_BLOCK +
                kWindowRadius];
            const float ref_color_sum = local_ref_sum;
            const float ref_color_squared_sum = local_ref_squared_sum;
            float src_color_sum = 0.0f;
            float src_color_squared_sum = 0.0f;
            float src_ref_color_sum = 0.0f;
            float bilateral_weight_sum = 0.0f;

            for (int row = -kWindowRadius; row <= kWindowRadius; row += kWindowStep) {
                for (int col = -kWindowRadius; col <= kWindowRadius; col += kWindowStep) {
                    const float inv_z = 1.0f / z;
                    const float norm_col_src = inv_z * col_src + 0.5f;
                    const float norm_row_src = inv_z * row_src + 0.5f;
                    const float ref_color = local_ref_image.data[ref_image_idx];
                    const float src_color = tex2DLayered<float>(
                        src_images_texture_, norm_col_src, norm_row_src, src_image_idx);

                    const float bilateral_weight = bilateral_weight_computer_.Compute(
                        row, col, ref_center_color, ref_color);

                    const float bilateral_weight_src = bilateral_weight * src_color;

                    src_color_sum += bilateral_weight_src;
                    src_color_squared_sum += bilateral_weight_src * src_color;
                    src_ref_color_sum += bilateral_weight_src * ref_color;
                    bilateral_weight_sum += bilateral_weight;

                    ref_image_idx += kWindowStep;

                    // Accumulate warped source coordinates per row to reduce numerical
                    // errors. Note that this is necessary since coordinates usually are in
                    // the order of 1000s as opposed to the color values which are
                    // normalized to the range [0, 1].
                    col_src += tform_step[0];
                    row_src += tform_step[3];
                    z += tform_step[6];
                }

                ref_image_base_idx += kWindowStep * 3 * THREADS_PER_BLOCK;
                ref_image_idx = ref_image_base_idx;

                base_col_src += tform_step[1];
                base_row_src += tform_step[4];
                base_z += tform_step[7];

                col_src = base_col_src;
                row_src = base_row_src;
                z = base_z;
            }

            const float inv_bilateral_weight_sum = 1.0f / bilateral_weight_sum;
            src_color_sum *= inv_bilateral_weight_sum;
            src_color_squared_sum *= inv_bilateral_weight_sum;
            src_ref_color_sum *= inv_bilateral_weight_sum;

            const float ref_color_var =
                ref_color_squared_sum - ref_color_sum * ref_color_sum;
            const float src_color_var =
                src_color_squared_sum - src_color_sum * src_color_sum;

            // Based on Jensen's Inequality for convex functions, the variance
            // should always be larger than 0. Do not make this threshold smaller.
            constexpr float kMinVar = 1e-5f;
            if (ref_color_var < kMinVar || src_color_var < kMinVar) {
                return kMaxCost;
            }
            else {
                const float src_ref_color_covar =
                    src_ref_color_sum - ref_color_sum * src_color_sum;
                const float src_ref_color_var = sqrt(ref_color_var * src_color_var);
                return max(0.0f,
                    min(kMaxCost, 1.0f - src_ref_color_covar / src_ref_color_var));
            }
        }

    private:
        const cudaTextureObject_t src_images_texture_;
        const cudaTextureObject_t poses_texture_;
        const BilateralWeightComputer bilateral_weight_computer_;
    };


    template <int kWindowSize, int kWindowStep>
    __global__ void ComputeInitialCost(GpuMat<float> cost_map,
        const GpuMat<float> depth_map,
        const GpuMat<float> normal_map,
        const cudaTextureObject_t ref_image_texture,
        const GpuMat<float> ref_sum_image,
        const GpuMat<float> ref_squared_sum_image,
        const cudaTextureObject_t src_images_texture,
        const cudaTextureObject_t poses_texture,
        const float sigma_spatial,
        const float sigma_color) {
        const int col = blockDim.x * blockIdx.x + threadIdx.x;

        typedef PhotoConsistencyCostComputer<kWindowSize, kWindowStep>
            PhotoConsistencyCostComputerType;
        PhotoConsistencyCostComputerType pcc_computer(ref_image_texture,
            src_images_texture,
            poses_texture,
            sigma_spatial,
            sigma_color);
        pcc_computer.col = col;

        __shared__ float local_ref_image_data
            [PhotoConsistencyCostComputerType::LocalRefImageType::kDataSize];
        pcc_computer.local_ref_image.data = &local_ref_image_data[0];

        float normal[3] = { 0 };
        pcc_computer.normal = normal;

        for (int row = 0; row < cost_map.GetHeight(); ++row) {
            // Note that this must be executed even for pixels outside the borders,
            // since pixels are used in the local neighborhood of the current pixel.
            pcc_computer.Read(row);

            if (col < cost_map.GetWidth()) {
                pcc_computer.depth = depth_map.Get(row, col);
                normal_map.GetSlice(row, col, normal);

                pcc_computer.row = row;
                pcc_computer.local_ref_sum = ref_sum_image.Get(row, col);
                pcc_computer.local_ref_squared_sum = ref_squared_sum_image.Get(row, col);

                for (int image_idx = 0; image_idx < cost_map.GetDepth(); ++image_idx) {
                    pcc_computer.src_image_idx = image_idx;
                    cost_map.Set(row, col, image_idx, pcc_computer.Compute());
                }
            }
        }
    }
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
        sweep_block_size_.x = THREADS_PER_BLOCK;
        sweep_block_size_.y = 1;
        sweep_block_size_.z = 1;
        sweep_grid_size_.x = (depth_map_->GetWidth() - 1) / THREADS_PER_BLOCK + 1;
        sweep_grid_size_.y = 1;
        sweep_grid_size_.z = 1;

        elem_wise_block_size_.x = THREADS_PER_BLOCK;
        elem_wise_block_size_.y = THREADS_PER_BLOCK;
        elem_wise_block_size_.z = 1;
        elem_wise_grid_size_.x = (depth_map_->GetWidth() - 1) / THREADS_PER_BLOCK + 1;
        elem_wise_grid_size_.y =
            (depth_map_->GetHeight() - 1) / THREADS_PER_BLOCK + 1;
        elem_wise_grid_size_.z = 1;
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

#define CASE_WINDOW_RADIUS(window_radius, window_step)              \
  case window_radius:                                               \
    RunWithWindowSizeAndStep<2 * window_radius + 1, window_step>(); \
    break;

#define CASE_WINDOW_STEP(window_step)                          \
  case window_step:                                            \
    switch (options_.window_radius) {                          \
      CASE_WINDOW_RADIUS(1, window_step)                       \
      CASE_WINDOW_RADIUS(2, window_step)                       \
      CASE_WINDOW_RADIUS(3, window_step)                       \
      CASE_WINDOW_RADIUS(4, window_step)                       \
      CASE_WINDOW_RADIUS(5, window_step)                       \
      CASE_WINDOW_RADIUS(6, window_step)                       \
      CASE_WINDOW_RADIUS(7, window_step)                       \
      CASE_WINDOW_RADIUS(8, window_step)                       \
      CASE_WINDOW_RADIUS(9, window_step)                       \
      CASE_WINDOW_RADIUS(10, window_step)                      \
      CASE_WINDOW_RADIUS(11, window_step)                      \
      CASE_WINDOW_RADIUS(12, window_step)                      \
      CASE_WINDOW_RADIUS(13, window_step)                      \
      CASE_WINDOW_RADIUS(14, window_step)                      \
      CASE_WINDOW_RADIUS(15, window_step)                      \
      CASE_WINDOW_RADIUS(16, window_step)                      \
      CASE_WINDOW_RADIUS(17, window_step)                      \
      CASE_WINDOW_RADIUS(18, window_step)                      \
      CASE_WINDOW_RADIUS(19, window_step)                      \
      CASE_WINDOW_RADIUS(20, window_step)                      \
      default: {                                               \
        LOG(ERROR) << "Window size " << options_.window_radius \
                   << " not supported";                        \
        break;                                                 \
      }                                                        \
    }                                                          \
    break;

        switch (options_.window_step) {
            CASE_WINDOW_STEP(1)
                CASE_WINDOW_STEP(2)
        default: {
                LOG(ERROR) << "Window step " << options_.window_step << " not supported";
                break;
            }
        }

#undef SWITCH_WINDOW_RADIUS
#undef CALL_RUN_FUNC
    }


    struct SweepOptions {
        float perturbation = 1.0f;
        float depth_min = 0.0f;
        float depth_max = 1.0f;
        int num_samples = 15;
        float sigma_spatial = 3.0f;
        float sigma_color = 0.3f;
        float ncc_sigma = 0.6f;
        float min_triangulation_angle = 0.5f;
        float incident_angle_sigma = 0.9f;
        float prev_sel_prob_weight = 0.0f;
        float geom_consistency_regularizer = 0.1f;
        float geom_consistency_max_cost = 5.0f;
        float filter_min_ncc = 0.1f;
        float filter_min_triangulation_angle = 3.0f;
        int filter_min_num_consistent = 2;
        float filter_geom_consistency_max_cost = 1.0f;
    };

    template <int kWindowSize, int kWindowStep>
    void CudaPatchMatch::RunWithWindowSizeAndStep() {
        // Wait for all initializations to finish.
        CUDA_SYNC_AND_CHECK();

        CudaTimer total_timer;
        CudaTimer init_timer;

        ComputeCudaConfig();
        ComputeInitialCost<kWindowSize, kWindowStep>
            << <sweep_grid_size_, sweep_block_size_ >> > (*cost_map_,
                *depth_map_,
                *normal_map_,
                ref_image_texture_->GetObj(),
                *ref_image_->sum_image,
                *ref_image_->squared_sum_image,
                src_images_texture_->GetObj(),
                poses_texture_[0]->GetObj(),
                options_.sigma_spatial,
                options_.sigma_color);
        CUDA_SYNC_AND_CHECK();

        init_timer.Print("Initialization");

        const float total_num_steps = options_.num_iterations * 4;

        SweepOptions sweep_options;
        sweep_options.depth_min = options_.depth_min;
        sweep_options.depth_max = options_.depth_max;
        sweep_options.sigma_spatial = options_.sigma_spatial;
        sweep_options.sigma_color = options_.sigma_color;
        sweep_options.num_samples = options_.num_samples;
        sweep_options.ncc_sigma = options_.ncc_sigma;
        sweep_options.min_triangulation_angle =
            DEG2RAD(options_.min_triangulation_angle);
        sweep_options.incident_angle_sigma = options_.incident_angle_sigma;
        sweep_options.geom_consistency_regularizer =
            options_.geom_consistency_regularizer;
        sweep_options.geom_consistency_max_cost = options_.geom_consistency_max_cost;
        sweep_options.filter_min_ncc = options_.filter_min_ncc;
        sweep_options.filter_min_triangulation_angle =
            DEG2RAD(options_.filter_min_triangulation_angle);
        sweep_options.filter_min_num_consistent = options_.filter_min_num_consistent;
        sweep_options.filter_geom_consistency_max_cost =
            options_.filter_geom_consistency_max_cost;

        for (int iter = 0; iter < options_.num_iterations; ++iter) {
            CudaTimer iter_timer;

            for (int sweep = 0; sweep < 4; ++sweep) {
                CudaTimer sweep_timer;

                // Expenentially reduce amount of perturbation during the optimization.
                sweep_options.perturbation = 1.0f / std::pow(2.0f, iter + sweep / 4.0f);

                // Linearly increase the influence of previous selection probabilities.
                sweep_options.prev_sel_prob_weight =
                    static_cast<float>(iter * 4 + sweep) / total_num_steps;

                const bool last_sweep = iter == options_.num_iterations - 1 && sweep == 3;

#define CALL_SWEEP_FUNC                                   /*\*/
  /*SweepFromTopToBottom<kWindowSize,                       \
                       kWindowStep,                       \
                       kGeomConsistencyTerm,              \
                       kFilterPhotoConsistency,           \
                       kFilterGeomConsistency>            \
      <<<sweep_grid_size_, sweep_block_size_>>>(          \
          *global_workspace_,                             \
          *rand_state_map_,                               \
          *cost_map_,                                     \
          *depth_map_,                                    \
          *normal_map_,                                   \
          *consistency_mask_,                             \
          *sel_prob_map_,                                 \
          *prev_sel_prob_map_,                            \
          ref_image_texture_->GetObj(),                   \
          *ref_image_->sum_image,                         \
          *ref_image_->squared_sum_image,                 \
          src_images_texture_->GetObj(),                  \
          src_depth_maps_texture_ == nullptr              \
              ? 0                                         \
              : src_depth_maps_texture_->GetObj(),        \
          poses_texture_[rotation_in_half_pi_]->GetObj(), \
          sweep_options);*/

                if (last_sweep) {
                    if (options_.filter) {
                        consistency_mask_.reset(new GpuMat<uint8_t>(cost_map_->GetWidth(),
                            cost_map_->GetHeight(),
                            cost_map_->GetDepth()));
                        consistency_mask_->FillWithScalar(0);
                    }
                    if (options_.geom_consistency) {
                        const bool kGeomConsistencyTerm = true;
                        if (options_.filter) {
                            const bool kFilterPhotoConsistency = true;
                            const bool kFilterGeomConsistency = true;
                            CALL_SWEEP_FUNC
                        }
                        else {
                            const bool kFilterPhotoConsistency = false;
                            const bool kFilterGeomConsistency = false;
                            CALL_SWEEP_FUNC
                        }
                    }
                    else {
                        const bool kGeomConsistencyTerm = false;
                        if (options_.filter) {
                            const bool kFilterPhotoConsistency = true;
                            const bool kFilterGeomConsistency = false;
                            CALL_SWEEP_FUNC
                        }
                        else {
                            const bool kFilterPhotoConsistency = false;
                            const bool kFilterGeomConsistency = false;
                            CALL_SWEEP_FUNC
                        }
                    }
                }
                else {
                    const bool kFilterPhotoConsistency = false;
                    const bool kFilterGeomConsistency = false;
                    if (options_.geom_consistency) {
                        const bool kGeomConsistencyTerm = true;
                        CALL_SWEEP_FUNC
                    }
                    else {
                        const bool kGeomConsistencyTerm = false;
                        CALL_SWEEP_FUNC
                    }
                }

#undef CALL_SWEEP_FUNC

                CUDA_SYNC_AND_CHECK();

                Rotate();

                // Rotate selected image map.
                if (last_sweep && options_.filter) {
                    std::unique_ptr<GpuMat<uint8_t>> rot_consistency_mask_(
                        new GpuMat<uint8_t>(cost_map_->GetWidth(),
                            cost_map_->GetHeight(),
                            cost_map_->GetDepth()));
                    consistency_mask_->Rotate(rot_consistency_mask_.get());
                    consistency_mask_.swap(rot_consistency_mask_);
                }

                sweep_timer.Print(" Sweep " + std::to_string(sweep + 1));
            }

            iter_timer.Print("Iteration " + std::to_string(iter + 1));
        }

        total_timer.Print("Total");
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