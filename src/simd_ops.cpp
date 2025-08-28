#include "simd_ops.h"
#include <immintrin.h>
#include <cmath>

namespace surprise_metrics {
namespace simd {

// AVX-512 implementation for computing returns
void compute_returns_avx512(const float* prices, float* returns, size_t n) {
    const __m512 zero = _mm512_setzero_ps();
    
    size_t simd_end = (n - 1) & ~15; // Process 16 elements at a time
    
    for (size_t i = 0; i < simd_end; i += 16) {
        __m512 curr_prices = _mm512_loadu_ps(&prices[i + 1]);
        __m512 prev_prices = _mm512_loadu_ps(&prices[i]);
        
        // Compute log returns: log(p[i+1]/p[i]) = log(p[i+1]) - log(p[i])
        __m512 ratio = _mm512_div_ps(curr_prices, prev_prices);
        
        // Custom fast log approximation for financial data
        // Using polynomial approximation for better accuracy
        __m512 log_returns = _mm512_log_ps(ratio);
        
        _mm512_storeu_ps(&returns[i], log_returns);
    }
    
    // Handle remaining elements
    for (size_t i = simd_end; i < n - 1; ++i) {
        returns[i] = std::log(prices[i + 1] / prices[i]);
    }
}

// AVX-512 implementation for realized variance
void compute_realized_variance_avx512(const float* returns, float* rv, size_t n, int window) {
    const __m512 zero = _mm512_setzero_ps();
    
    for (size_t i = 0; i < n - window; ++i) {
        __m512 sum = zero;
        
        size_t j = 0;
        for (; j + 16 <= window; j += 16) {
            __m512 r = _mm512_loadu_ps(&returns[i + j]);
            sum = _mm512_fmadd_ps(r, r, sum); // r * r + sum
        }
        
        // Horizontal sum of vector
        float result = _mm512_reduce_add_ps(sum);
        
        // Handle remaining elements
        for (; j < window; ++j) {
            result += returns[i + j] * returns[i + j];
        }
        
        rv[i] = result;
    }
}

// AVX-512 implementation for bipower variation
void compute_bipower_variation_avx512(const float* returns, float* bv, size_t n) {
    const __m512 pi_over_2 = _mm512_set1_ps(1.5707963267948966f);
    const __m512 sign_mask = _mm512_set1_ps(-0.0f);
    
    size_t simd_end = (n - 1) & ~15;
    
    for (size_t i = 0; i < simd_end; i += 16) {
        __m512 curr = _mm512_loadu_ps(&returns[i + 1]);
        __m512 prev = _mm512_loadu_ps(&returns[i]);
        
        // Compute |r[i]| * |r[i-1]|
        curr = _mm512_andnot_ps(sign_mask, curr); // abs
        prev = _mm512_andnot_ps(sign_mask, prev); // abs
        
        __m512 prod = _mm512_mul_ps(curr, prev);
        prod = _mm512_mul_ps(prod, pi_over_2);
        
        _mm512_storeu_ps(&bv[i], prod);
    }
    
    // Handle remaining elements
    for (size_t i = simd_end; i < n - 1; ++i) {
        bv[i] = std::abs(returns[i + 1]) * std::abs(returns[i]) * 1.5707963267948966f;
    }
}

}} // namespace surprise_metrics::simd
