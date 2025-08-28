#pragma once

#include <cstddef>

namespace surprise_metrics {
namespace simd {

// Aligned allocator for SIMD operations
template<typename T, std::size_t Alignment>
class aligned_allocator {
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    
    template<typename U>
    struct rebind {
        using other = aligned_allocator<U, Alignment>;
    };
    
    aligned_allocator() noexcept = default;
    
    template<typename U>
    aligned_allocator(const aligned_allocator<U, Alignment>&) noexcept {}
    
    T* allocate(size_type n) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(ptr);
    }
    
    void deallocate(T* ptr, size_type) noexcept {
        free(ptr);
    }
};

template<typename T>
using aligned_vector = std::vector<T, aligned_allocator<T, 64>>;

// Function declarations
void compute_returns_avx512(const float* prices, float* returns, size_t n);
void compute_realized_variance_avx512(const float* returns, float* rv, size_t n, int window);
void compute_bipower_variation_avx512(const float* returns, float* bv, size_t n);

} // namespace simd
} // namespace surprise_metrics
