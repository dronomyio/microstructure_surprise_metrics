#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>

namespace surprise_metrics {
namespace cuda {

// Constants for kernel configuration
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int MAX_BLOCKS = 65535;

// Kernel for GARCH volatility estimation
__global__ void garch_kernel(
    const float* __restrict__ returns,
    float* __restrict__ sigma_squared,
    const int n,
    const float omega,
    const float alpha, 
    const float beta
);

// Kernel for Lee-Mykland jump detection
__global__ void lee_mykland_kernel(
    const float* __restrict__ returns,
    float* __restrict__ local_vol,
    float* __restrict__ test_stats,
    bool* __restrict__ jump_flags,
    const int n,
    const int window_size,
    const float threshold
);

// Kernel for BNS test statistic
__global__ void bns_kernel(
    const float* __restrict__ returns,
    float* __restrict__ rv,
    float* __restrict__ bv,
    float* __restrict__ tq,
    float* __restrict__ test_stats,
    const int n,
    const int window_size
);

// Kernel for Hawkes process intensity
__global__ void hawkes_intensity_kernel(
    const float* __restrict__ timestamps,
    float* __restrict__ intensity,
    const int n,
    const float mu,
    const float phi,
    const float kappa,
    const float dt
);

// Kernel for parallel standardized returns
__global__ void standardized_returns_kernel(
    const float* __restrict__ returns,
    const float* __restrict__ sigma,
    float* __restrict__ z_scores,
    const int n
);

// Multi-GPU management
class MultiGPUContext {
public:
    MultiGPUContext(int num_gpus);
    ~MultiGPUContext();
    
    void distribute_data(const float* host_data, size_t total_size);
    void gather_results(float* host_results, size_t total_size);
    
private:
    int num_gpus_;
    std::vector<cudaStream_t> streams_;
    std::vector<float*> device_buffers_;
    std::vector<size_t> chunk_sizes_;
};

}} // namespace surprise_metrics::cuda
