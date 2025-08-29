#include "simd_ops.h"
#include <cmath>
#include <algorithm>
#include <iostream>

// Only include immintrin if we have AVX2
#ifdef HAS_AVX2
#include <immintrin.h>
#endif

namespace surprise_metrics {
namespace simd {

// AVX2 implementation for computing returns
void compute_returns_avx512(const float* prices, float* returns, size_t n) {
    #ifdef HAS_AVX2
    // Use compile-time check, not runtime
    std::cout << "Using AVX2 optimized path\n";
    
    size_t simd_end = (n - 1) & ~7; // Process 8 elements at a time
    
    for (size_t i = 0; i < simd_end; i += 8) {
        __m256 curr_prices = _mm256_loadu_ps(&prices[i + 1]);
        __m256 prev_prices = _mm256_loadu_ps(&prices[i]);
        
        // Compute simple returns: (p[i+1] - p[i]) / p[i]
        __m256 diff = _mm256_sub_ps(curr_prices, prev_prices);
        __m256 returns_vec = _mm256_div_ps(diff, prev_prices);
        
        _mm256_storeu_ps(&returns[i], returns_vec);
    }
    
    // Handle remaining elements with scalar log returns
    for (size_t i = simd_end; i < n - 1; ++i) {
        returns[i] = std::log(prices[i + 1] / prices[i]);
    }
    #else
    // Scalar fallback
    std::cout << "Using scalar path (no AVX2)\n";
    for (size_t i = 0; i < n - 1; ++i) {
        returns[i] = std::log(prices[i + 1] / prices[i]);
    }
    #endif
}

// AVX2 implementation for realized variance
void compute_realized_variance_avx512(const float* returns, float* rv, size_t n, int window) {
    #ifdef HAS_AVX2
    for (size_t i = 0; i < n - window; ++i) {
        __m256 sum = _mm256_setzero_ps();
        
        size_t j = 0;
        // Process 8 elements at a time
        for (; j + 8 <= window; j += 8) {
            __m256 r = _mm256_loadu_ps(&returns[i + j]);
            sum = _mm256_fmadd_ps(r, r, sum); // r * r + sum
        }
        
        // Horizontal sum for AVX2
        __m128 low = _mm256_castps256_ps128(sum);
        __m128 high = _mm256_extractf128_ps(sum, 1);
        __m128 sum128 = _mm_add_ps(low, high);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float result = _mm_cvtss_f32(sum128);
        
        // Handle remaining elements
        for (; j < window; ++j) {
            result += returns[i + j] * returns[i + j];
        }
        
        rv[i] = result;
    }
    #else
    // Scalar fallback
    for (size_t i = 0; i < n - window; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < window; ++j) {
            sum += returns[i + j] * returns[i + j];
        }
        rv[i] = sum;
    }
    #endif
}

// AVX2 implementation for bipower variation
void compute_bipower_variation_avx512(const float* returns, float* bv, size_t n) {
    const float pi_over_2 = 1.5707963267948966f;
    
    #ifdef HAS_AVX2
    const __m256 pi_over_2_vec = _mm256_set1_ps(pi_over_2);
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    
    size_t simd_end = (n - 1) & ~7;
    
    for (size_t i = 0; i < simd_end; i += 8) {
        __m256 curr = _mm256_loadu_ps(&returns[i + 1]);
        __m256 prev = _mm256_loadu_ps(&returns[i]);
        
        // Compute |r[i]| * |r[i-1]|
        curr = _mm256_andnot_ps(sign_mask, curr); // abs
        prev = _mm256_andnot_ps(sign_mask, prev); // abs
        
        __m256 prod = _mm256_mul_ps(curr, prev);
        prod = _mm256_mul_ps(prod, pi_over_2_vec);
        
        _mm256_storeu_ps(&bv[i], prod);
    }
    
    // Handle remaining elements
    for (size_t i = simd_end; i < n - 1; ++i) {
        bv[i] = std::abs(returns[i + 1]) * std::abs(returns[i]) * pi_over_2;
    }
    #else
    // Scalar fallback
    for (size_t i = 0; i < n - 1; ++i) {
        bv[i] = std::abs(returns[i + 1]) * std::abs(returns[i]) * pi_over_2;
    }
    #endif
}

}} // namespace surprise_metrics::simd
