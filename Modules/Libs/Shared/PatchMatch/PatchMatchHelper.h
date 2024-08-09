#pragma once
#include <string>
#include <vector>
#include <memory>

#include <colmap/mvs/image.h>
#include <colmap/mvs/depth_map.h>
#include <colmap/mvs/normal_map.h>
#include <colmap/mvs/workspace.h>
#include <colmap/util/logging.h>
#include <colmap/util/threading.h>

#include <Common/Export.h>
#include "CudaPatchMatch.h"

namespace GU
{
	class DLL_API PatchMatchHelper
	{
	public:
		PatchMatchHelper(CudaPatchMatch::Options options_,
			const std::string& workspace_path,
			const std::string& workspace_format,
			const std::string& pmvs_option_name,
			const std::string& config_path = "");
		~PatchMatchHelper();
		void Run();
	private:
		void Init(CudaPatchMatch::Options options_);
		void InitWorkspace(CudaPatchMatch::Options options_);
		void InitProblems(CudaPatchMatch::Options options_);
		void InitGpuIndices(CudaPatchMatch::Options options_);
		void ProcessProblem(const CudaPatchMatch::Options& options,
			const size_t problem_idx);

	private:

		CudaPatchMatch::Options m_options;
		const std::string m_workspacePath;
		const std::string m_workspaceFormat;
		const std::string m_pmvsOptionName;
		const std::string m_configPath;

		std::unique_ptr<colmap::mvs::CachedWorkspace> m_workspace;
		std::unique_ptr<colmap::ThreadPool> m_threadpool;
		std::mutex m_workspaceMutex;
		std::vector<CudaPatchMatch::Problem> m_problems;
		std::vector<std::pair<float, float>> m_depthRanges;
		std::vector<int> m_gpuIndices;
	};
}