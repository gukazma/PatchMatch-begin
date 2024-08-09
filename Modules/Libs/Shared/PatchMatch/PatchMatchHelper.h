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
#include "CudaPatchMatch.h"
namespace GU
{
	class DLL_API PatchMatchHelper
	{
	public:
		PatchMatchHelper(CudaPatchMatch::Options options_);
		~PatchMatchHelper();
		void Run();
	private:
		void Init(CudaPatchMatch::Options options_);
		void InitWorkspace(CudaPatchMatch::Options options_);
		void InitProblems(CudaPatchMatch::Options options_);
		void InitGpuIndices(CudaPatchMatch::Options options_);

		CudaPatchMatch::Options m_options;
		std::unique_ptr<colmap::mvs::CachedWorkspace> m_workspace;
		std::vector<CudaPatchMatch::Problem> m_problems;
		std::vector<std::pair<float, float>> m_depthRanges;
		std::vector<int> m_gpuIndices;
	};
}