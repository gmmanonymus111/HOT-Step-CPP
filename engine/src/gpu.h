#pragma once

#if defined (GGML_USE_CUDA)
  #if defined (__HIP_PLATFORM_AMD__)
     #include <hip/hip_runtime.h>
     #define cudaMemGetInfo hipMemGetInfo
     #define cudaGetErrorString hipGetErrorString
     #define cudaError_t hipError_t
     #define cudaSuccess hipSuccess
  #else
     #include <cuda_runtime.h>
  #endif
#endif
