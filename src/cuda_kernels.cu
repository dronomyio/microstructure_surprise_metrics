#include "cuda_kernels.cuh"
#include <cstdio>
#include <cmath>

namespace surprise_metrics {
namespace cuda {

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
    
    // Load returns to shared memory with coalesced access
    if (gid < n) {
        shared_mem[tid] = returns[gid] * returns[gid];
    }
    __syncthreads();
    
    // Initialize first value
    if (gid == 0) {
        sigma_squared[0] = omega / (1.0f - alpha - beta);
    }
    __syncthreads();
    
    // Sequential GARCH update within blocks
    if (gid > 0 && gid < n) {
        float prev_sigma2 = (gid > 1) ? sigma_squared[gid-1] : sigma_squared[0];
        float curr_sigma2 = omega + alpha * shared_mem[tid-1] + beta * prev_sigma2;
        sigma_squared[gid] = curr_sigma2;
    }
}

// Lee-Mykland kernel with warp-level primitives
__global__ void lee_mykland_kernel(
    const float* __restrict__ returns,
    float* __restrict__ local_vol,
    float* __restrict__ test_stats,
    bool* __restrict__ jump_flags,
    const int n,
    const int window_size,
    const float threshold
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid >= window_size && gid < n) {
        // Compute bipower variation in window
        float bv = 0.0f;
        const float pi_over_2 = 1.5707963267948966f;
        
        #pragma unroll 4
        for (int i = gid - window_size + 1; i < gid; i++) {
            bv += fabsf(returns[i]) * fabsf(returns[i-1]);
        }
        bv *= pi_over_2 / (window_size - 1);
        
        local_vol[gid] = sqrtf(bv);
        
        // Compute test statistic
        float L = fabsf(returns[gid]) / local_vol[gid];
        test_stats[gid] = L;
        
        // Jump detection with threshold
        float Cn = sqrtf(2.0f * logf(float(n)));
        float Sn = 1.0f/Cn + (logf(3.14159f) + logf(2.0f * logf(float(n)))) / (2.0f * Cn);
        float critical_value = threshold + Cn * Sn;
        
        jump_flags[gid] = (L > critical_value);
    }
}

// BNS kernel
__global__ void bns_kernel(
    const float* __restrict__ returns,
    float* __restrict__ rv,
    float* __restrict__ bv,
    float* __restrict__ tq,
    float* __restrict__ test_stats,
    const int n,
    const int window_size
) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    if (gid < n - window_size) {
        float local_rv = 0.0f;
        float local_bv = 0.0f;
        float local_tq = 0.0f;
        
        const float pi_over_2 = 1.5707963267948966f;
        const float mu_43 = 1.67f; // E[|Z|^(4/3)] for standard normal
        
        // Compute statistics over window
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
        //local_tq *= window_size * powf(mu_43, -3.0f) * window_size / (window_size - 2);
	//local_tq *= (window_size / (window_size - 2)) * powf(mu_43, -3.0f);
	local_tq *= powf(mu_43, -3.0f) * window_size / (window_size - 2);
        
        // Store results
        rv[gid] = local_rv;
        bv[gid] = local_bv;
        tq[gid] = local_tq;
	 
	//ref https://public.econ.duke.edu/~get/browse/courses/883/Spr15/COURSE-MATERIALS/Z_Papers/BNSJFEC2006.pdf
	//page 8 equ 11
        
        // Compute BNS test statistic
        float jump_component = local_rv - local_bv; //RV - BV
        float theta = (pi_over_2 * pi_over_2 / 4.0f + 3.14159f - 5.0f); //ϑ
        float denominator = sqrtf(theta * local_tq / (local_bv * local_bv));
        
        test_stats[gid] = sqrtf(float(window_size)) * (jump_component / local_rv) / denominator;
    }
}

// Hawkes intensity kernel with exponential decay
__global__ void hawkes_intensity_kernel(
    const float* __restrict__ timestamps,
    float* __restrict__ intensity,
    const int n,
    const float mu,
    const float phi,
    const float kappa,
    const float dt
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid < n) {
        float lambda = mu;
        float t_current = timestamps[gid];
        
        // Sum contributions from all previous events
        for (int j = 0; j < gid; j++) {
            float time_diff = (t_current - timestamps[j]) * 1e-9f; // Convert ns to seconds
            if (time_diff > 0) {
                lambda += phi * expf(-kappa * time_diff);
            }
        }
        
        intensity[gid] = lambda;
    }
}

// Standardized returns kernel using vectorized operations
__global__ void standardized_returns_kernel(
    const float* __restrict__ returns,
    const float* __restrict__ sigma,
    float* __restrict__ z_scores,
    const int n
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop for better GPU utilization
    for (int i = gid; i < n; i += stride) {
        z_scores[i] = returns[i] / sigma[i];
    }
}

// Host launcher functions
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
                           bool* jump_flags, int n, float threshold,
                           cudaStream_t stream) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Allocate test_stats on device
    float* test_stats;
    cudaMalloc(&test_stats, n * sizeof(float));
    
    lee_mykland_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        returns, local_vol, test_stats, jump_flags, n, 100, threshold
    );
    
    cudaFree(test_stats);
}

void launch_hawkes_intensity(float* timestamps, float* intensity, 
                             int n, float mu, float phi, float kappa,
                             cudaStream_t stream) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    hawkes_intensity_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
        timestamps, intensity, n, mu, phi, kappa, 0.001f
    );
}

}} // namespace surprise_metrics::cuda
