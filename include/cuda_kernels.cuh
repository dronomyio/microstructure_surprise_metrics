#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

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

// Host functions for launching kernels
void launch_garch_estimation(float* returns, float* sigma, int n, 
                             float omega, float alpha, float beta,
                             cudaStream_t stream = 0);

void launch_jump_detection(float* returns, float* local_vol, 
                           bool* jump_flags, int n, float threshold,
                           cudaStream_t stream = 0);

void launch_hawkes_intensity(float* timestamps, float* intensity, 
                             int n, float mu, float phi, float kappa,
                             cudaStream_t stream = 0);

}} // namespace surprise_metrics::cuda
