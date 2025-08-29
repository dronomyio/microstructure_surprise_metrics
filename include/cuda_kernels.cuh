#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace surprise_metrics {
namespace cuda {

// Constants for kernel configuration
constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int MAX_BLOCKS = 65535;

// ==================== KERNEL DECLARATIONS ====================

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
    char* __restrict__ jump_flags,  // Changed bool* to char*
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

// Enhanced Hawkes process intensity with branching ratio and endogeneity
__global__ void hawkes_intensity_kernel(
    const float* __restrict__ timestamps,
    float* __restrict__ intensity,
    float* __restrict__ branching_ratio,
    float* __restrict__ endogeneity,
    const int n,
    const float mu,
    const float phi,
    const float kappa
);

// Poisson baseline model for trade arrival
__global__ void poisson_intensity_kernel(
    const float* __restrict__ timestamps,
    const float* __restrict__ prices,
    float* __restrict__ intensity,
    float* __restrict__ surprise_score,
    const int n,
    const int window_size,
    const float time_window_seconds
);

// Composite burst detection
__global__ void burst_detection_kernel(
    const float* __restrict__ hawkes_intensity,
    const float* __restrict__ poisson_surprise,
    const float* __restrict__ branching_ratio,
    char* __restrict__ burst_flags,  // Changed bool* to char*
    float* __restrict__ burst_scores,
    const int n,
    const float hawkes_threshold,
    const float poisson_threshold,
    const float composite_threshold
);

// Kernel for parallel standardized returns
__global__ void standardized_returns_kernel(
    const float* __restrict__ returns,
    const float* __restrict__ sigma,
    float* __restrict__ z_scores,
    const int n
);

// ==================== LAUNCHER FUNCTION DECLARATIONS ====================
// These can be called from .cpp files

void launch_garch_estimation(
    float* returns, float* sigma, int n, 
    float omega, float alpha, float beta,
    cudaStream_t stream = 0
);

void launch_jump_detection(
    float* returns, float* local_vol, 
    char* jump_flags, int n, float threshold,  // Changed bool* to char*
    cudaStream_t stream = 0
);

void launch_bns_computation(
    float* returns, float* rv, float* bv, float* tq,
    float* stats, int n, int window, 
    cudaStream_t stream = 0
);

void launch_lee_mykland_computation(
    float* returns, float* local_vol, float* test_stats,
    char* jump_flags, int n, int window, float threshold,  // Changed bool* to char*
    cudaStream_t stream = 0
);

void launch_hawkes_computation(
    float* timestamps, float* intensity, float* branching,
    float* endogeneity, int n, float mu, float phi, float kappa,
    cudaStream_t stream = 0
);

void launch_poisson_computation(
    float* timestamps, float* prices, float* intensity,
    float* surprise, int n, int window, float time_window,
    cudaStream_t stream = 0
);

void launch_burst_computation(
    float* hawkes_int, float* poisson_surp, float* branching,
    char* flags, float* scores, int n, float h_thresh,  // Changed bool* to char*
    float p_thresh, float c_thresh, 
    cudaStream_t stream = 0
);

void launch_standardized_returns(
    float* returns, float* sigma, float* z_scores, int n,
    cudaStream_t stream = 0
);

}} // namespace surprise_metrics::cuda
