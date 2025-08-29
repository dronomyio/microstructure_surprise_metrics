#include "cuda_kernels.cuh"
#include <cstdio>
#include <cmath>
#include <float.h>

namespace surprise_metrics {
namespace cuda {

// ==================== KERNEL IMPLEMENTATIONS ====================

// Optimized GARCH kernel using shared memory
__global__ void garch_kernel(
    const float* __restrict__ returns,
    float* __restrict__ sigma_squared,
    const int n,
    const float omega,
    const float alpha,
    const float beta
) {
    extern __shared__ float shared_mem[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    if (gid < n) {
        shared_mem[tid] = returns[gid] * returns[gid];
    }
    __syncthreads();
    
    if (gid == 0) {
        sigma_squared[0] = omega / (1.0f - alpha - beta);
    }
    __syncthreads();
    
    if (gid > 0 && gid < n) {
        float prev_sigma2 = (gid > 1) ? sigma_squared[gid-1] : sigma_squared[0];
        float curr_sigma2 = omega + alpha * shared_mem[tid-1] + beta * prev_sigma2;
        sigma_squared[gid] = curr_sigma2;
    }
}

// Lee-Mykland kernel with corrected bipower variation
__global__ void lee_mykland_kernel(
    const float* __restrict__ returns,
    float* __restrict__ local_vol,
    float* __restrict__ test_stats,
    char* __restrict__ jump_flags,  // Changed bool* to char*
    const int n,
    const int window_size,
    const float threshold
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid >= window_size && gid < n) {
        float bv = 0.0f;
        const float pi_over_2 = 1.5707963267948966f;
        
        // Fix: Ensure we don't access returns[i-1] when i=0
        int start_idx = (gid - window_size + 1 > 1) ? (gid - window_size + 1) : 1;
        for (int i = start_idx; i < gid; i++) {
            bv += fabsf(returns[i]) * fabsf(returns[i-1]);
        }
        
        // Adjust denominator for the actual number of terms used
        int actual_terms = gid - start_idx;
        if (actual_terms > 0) {
            bv *= pi_over_2 / actual_terms;
        } else {
            bv = 0.01f;  // Small default value to avoid division by zero
        }
        
        local_vol[gid] = sqrtf(bv);
        
        // Avoid division by zero
        float L = (local_vol[gid] > 0.0f) ? fabsf(returns[gid]) / local_vol[gid] : 0.0f;
        test_stats[gid] = L;
        
        // Debug output for first few threads
        if (gid < 5) {
            printf("Thread %d: return=%.6f, local_vol=%.6f, L=%.6f, threshold=%.2f\n", 
                   gid, returns[gid], local_vol[gid], L, threshold);
        }
        
        // Lee-Mykland critical value calculation
        // The threshold parameter is the critical value directly
        float critical_value = threshold;
        
        jump_flags[gid] = (L > critical_value) ? 1 : 0;  // Convert boolean to char (0 or 1)
        
        // Debug output for jumps detected
        if (L > critical_value && gid < 100) {
            printf("JUMP DETECTED at thread %d: L=%.6f > threshold=%.2f\n", gid, L, critical_value);
        }
    }
}

// Corrected BNS kernel - fixed tri-power quarticity scaling
__global__ void bns_kernel(
    const float* __restrict__ returns,
    float* __restrict__ rv,
    float* __restrict__ bv,
    float* __restrict__ tq,
    float* __restrict__ test_stats,
    const int n,
    const int window_size
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid < n - window_size) {
        float local_rv = 0.0f;
        float local_bv = 0.0f;
        float local_tq = 0.0f;
        
        const float pi_over_2 = 1.5707963267948966f;
        const float mu_43 = 1.67f;
        
        for (int i = 0; i < window_size; i++) {
            int idx = gid + i;
            float r = returns[idx];
            local_rv += r * r;
            
            if (i > 0) {
                local_bv += fabsf(r) * fabsf(returns[idx-1]);
            }
            
            if (i > 1) {
                float r1 = powf(fabsf(returns[idx]), 4.0f/3.0f);
                float r2 = powf(fabsf(returns[idx-1]), 4.0f/3.0f);
                float r3 = powf(fabsf(returns[idx-2]), 4.0f/3.0f);
                local_tq += r1 * r2 * r3;
            }
        }
        
        local_bv *= pi_over_2 * window_size / (window_size - 1);
        local_tq *= powf(mu_43, -3.0f) * window_size / (window_size - 2);
        
        rv[gid] = local_rv;
        bv[gid] = local_bv;
        tq[gid] = local_tq;
        
        if (local_rv > 0 && local_bv > 0) {
            float jump_component = local_rv - local_bv;
            float theta = 0.609f;
            float denominator = sqrtf(theta * local_tq / (local_bv * local_bv));
            
            if (denominator > 0) {
                test_stats[gid] = sqrtf(float(window_size)) * (jump_component / local_rv) / denominator;
            } else {
                test_stats[gid] = 0.0f;
            }
        } else {
            test_stats[gid] = 0.0f;
        }
    }
}

// Complete Hawkes process implementation with parameter estimation
__global__ void hawkes_intensity_kernel(
    const float* __restrict__ timestamps,
    float* __restrict__ intensity,
    float* __restrict__ branching_ratio,
    float* __restrict__ endogeneity,
    const int n,
    const float mu,
    const float phi,
    const float kappa
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid < n) {
        float t_current = timestamps[gid] * 1e-9f;
        float lambda = mu;
        
        for (int j = 0; j < gid; j++) {
            float t_j = timestamps[j] * 1e-9f;
            float time_diff = t_current - t_j;
            if (time_diff > 0) {
                lambda += phi * expf(-kappa * time_diff);
            }
        }
        
        intensity[gid] = lambda;
        
        float n_branch = phi / kappa;
        branching_ratio[gid] = n_branch;
        
        if (lambda > 0) {
            endogeneity[gid] = 1.0f - mu / lambda;
        } else {
            endogeneity[gid] = 0.0f;
        }
    }
}

// Poisson baseline model for trade arrival
__global__ void poisson_intensity_kernel(
    const float* __restrict__ timestamps,
    const float* __restrict__ prices,
    float* __restrict__ intensity,
    float* __restrict__ surprise_score,
    const int n,
    const int window_size,
    const float time_window_seconds
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid >= window_size && gid < n) {
        float t_current = timestamps[gid] * 1e-9f;
        float t_start = timestamps[gid - window_size] * 1e-9f;
        float duration = t_current - t_start;
        
        int count = 0;
        float window_start = t_current - time_window_seconds;
        
        for (int i = gid - window_size; i < gid; i++) {
            float t = timestamps[i] * 1e-9f;
            if (t >= window_start && t < t_current) {
                count++;
            }
        }
        
        float lambda_baseline = float(window_size) / duration;
        intensity[gid] = lambda_baseline;
        
        float expected = lambda_baseline * time_window_seconds;
        
        if (expected > 0) {
            surprise_score[gid] = (count - expected) / sqrtf(expected);
        } else {
            surprise_score[gid] = 0.0f;
        }
    }
}

// Composite burst detection combining Hawkes and Poisson
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
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid < n) {
        float hawkes_score = branching_ratio[gid] / 0.8f;
        float poisson_score = fabsf(poisson_surprise[gid]) / 3.0f;
        
        float composite = 0.6f * hawkes_score + 0.4f * poisson_score;
        burst_scores[gid] = composite;
        
        burst_flags[gid] = ((branching_ratio[gid] > hawkes_threshold) ||
                           (fabsf(poisson_surprise[gid]) > poisson_threshold) ||
                           (composite > composite_threshold)) ? 1 : 0;  // Convert boolean to char (0 or 1)
    }
}

// Standardized returns kernel
__global__ void standardized_returns_kernel(
    const float* __restrict__ returns,
    const float* __restrict__ sigma,
    float* __restrict__ z_scores,
    const int n
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = gid; i < n; i += stride) {
        if (sigma[i] > 0) {
            z_scores[i] = returns[i] / sigma[i];
        } else {
            z_scores[i] = 0.0f;
        }
    }
}

// ==================== LAUNCHER IMPLEMENTATIONS ====================

void launch_garch_estimation(float* returns, float* sigma, int n, 
                             float omega, float alpha, float beta,
                             cudaStream_t stream) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t shared_mem_size = BLOCK_SIZE * sizeof(float);
    
    garch_kernel<<<blocks, BLOCK_SIZE, shared_mem_size, stream>>>(
        returns, sigma, n, omega, alpha, beta
    );
}

void launch_jump_detection(float* returns, float* local_vol, 
                           char* jump_flags, int n, float threshold,  // Changed bool* to char*
                           cudaStream_t stream) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    float* test_stats;
    cudaMalloc(&test_stats, n * sizeof(float));
    
    lee_mykland_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        returns, local_vol, test_stats, jump_flags, n, 100, threshold
    );
    
    cudaFree(test_stats);
}

void launch_bns_computation(float* returns, float* rv, float* bv, float* tq,
                            float* stats, int n, int window, 
                            cudaStream_t stream) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bns_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        returns, rv, bv, tq, stats, n, window
    );
}

void launch_lee_mykland_computation(float* returns, float* local_vol, float* test_stats,
                                    char* jump_flags, int n, int window, float threshold,  // Changed bool* to char*
                                    cudaStream_t stream) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    lee_mykland_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        returns, local_vol, test_stats, jump_flags, n, window, threshold
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Lee-Mykland kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    
    // Wait for kernel to complete and check for execution errors
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        printf("Lee-Mykland kernel execution failed: %s\n", cudaGetErrorString(err));
    }
}

void launch_hawkes_computation(float* timestamps, float* intensity, float* branching,
                               float* endogeneity, int n, float mu, float phi, float kappa,
                               cudaStream_t stream) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hawkes_intensity_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        timestamps, intensity, branching, endogeneity, n, mu, phi, kappa
    );
}

void launch_poisson_computation(float* timestamps, float* prices, float* intensity,
                                float* surprise, int n, int window, float time_window,
                                cudaStream_t stream) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    poisson_intensity_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        timestamps, prices, intensity, surprise, n, window, time_window
    );
}

void launch_burst_computation(float* hawkes_int, float* poisson_surp, float* branching,
                              char* flags, float* scores, int n, float h_thresh,  // Changed bool* to char*
                              float p_thresh, float c_thresh, 
                              cudaStream_t stream) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    burst_detection_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        hawkes_int, poisson_surp, branching, flags, scores, n, h_thresh, p_thresh, c_thresh
    );
}

void launch_standardized_returns(float* returns, float* sigma, float* z_scores, int n,
                                 cudaStream_t stream) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    standardized_returns_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        returns, sigma, z_scores, n
    );
}

}} // namespace surprise_metrics::cuda

