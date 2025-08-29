#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <cstddef>

namespace surprise_metrics {
namespace cuda {

// Multi-GPU management class (separated from CUDA kernels)
class MultiGPUContext {
public:
    MultiGPUContext(int num_gpus);
    ~MultiGPUContext();
    
    void distribute_data(const float* host_data, size_t total_size);
    void gather_results(float* host_results, size_t total_size);
    
    int get_num_gpus() const { return num_gpus_; }
    cudaStream_t get_stream(int gpu_id) { return streams_[gpu_id]; }
    float* get_device_buffer(int gpu_id) { return device_buffers_[gpu_id]; }
    size_t get_chunk_size(int gpu_id) { return chunk_sizes_[gpu_id]; }
    
private:
    int num_gpus_;
    std::vector<cudaStream_t> streams_;
    std::vector<float*> device_buffers_;
    std::vector<size_t> chunk_sizes_;
};

}} // namespace surprise_metrics::cuda
