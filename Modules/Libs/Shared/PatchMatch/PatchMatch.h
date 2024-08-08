#pragma once
#include <Common/Export.h>
#include <string>
class DLL_API PatchMatch
{
public:
	struct Options
	{
		std::string workingspace;
	};

	void Init(Options options_);

	void Run();

	Options options;
};