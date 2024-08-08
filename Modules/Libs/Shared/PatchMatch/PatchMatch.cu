#include "PatchMatch.h"
#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <device_launch_parameters.h>
#include <colmap/mvs/workspace.h>
__global__ void squareKernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        output[idx] = input[idx] * input[idx];
    }
}


void PatchMatch::Init(Options options_)
{
    colmap::mvs::Workspace::Options workspace_options;


    workspace_options.max_image_size = 1000;
    workspace_options.image_as_rgb = false;
    workspace_options.cache_size = 32;
     workspace_options.workspace_path = options_.workingspace;
    workspace_options.workspace_format = "COLMAP";
    workspace_options.input_type = "photometric";

    auto workspace_ = std::make_unique<colmap::mvs::CachedWorkspace>(workspace_options);

    auto depth_ranges_ = workspace_->GetModel().ComputeDepthRanges();
}

void PatchMatch::Run()
{
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