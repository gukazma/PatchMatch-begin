#pragma once
#ifdef WIN32
#ifdef DLL_EXPORT
#define DLL_API _declspec(dllexport)
#else
#define DLL_API _declspec(dllimport)
#endif
#else
#define DLL_API
#endif   // WIN32
