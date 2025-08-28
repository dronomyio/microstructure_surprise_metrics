#include "multi_gpu.h"
#include <algorithm>
#include <cuda_runtime.h>

namespace surprise_metrics {
namespace cuda {

MultiGPUContext::MultiGPUContext(int num_gpus) : num_gpus_(num_gpus) {
    streams_.resize(num_gpus);
    device_buffers_.resize(num_gpus);
    chunk_sizes_.resize(num_gpus);
    
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&streams_[i]);
    }
}

MultiGPUContext::~MultiGPUContext() {
    for (int i = 0; i < num_gpus_; i++) {
        cudaSetDevice(i);
        if (device_buffers_[i]) {
            cudaFree(device_buffers_[i]);
        }
        cudaStreamDestroy(streams_[i]);
    }
}

void MultiGPUContext::distribute_data(const float* host_data, size_t total_size) {
    size_t chunk_size = (total_size + num_gpus_ - 1) / num_gpus_;
    
    for (int i = 0; i < num_gpus_; i++) {
        cudaSetDevice(i);
        size_t offset = i * chunk_size;
        size_t actual_size = std::min(chunk_size, total_size - offset);
        chunk_sizes_[i] = actual_size;
        
        cudaMalloc(&device_buffers_[i], actual_size * sizeof(float));
        cudaMemcpyAsync(device_buffers_[i], host_data + offset, 
                       actual_size * sizeof(float),
                       cudaMemcpyHostToDevice, streams_[i]);
    }
}

void MultiGPUContext::gather_results(float* host_results, size_t total_size) {
    for (int i = 0; i < num_gpus_; i++) {
        cudaSetDevice(i);
        size_t offset = i * chunk_sizes_[i];
        cudaMemcpyAsync(host_results + offset, device_buffers_[i],
                       chunk_sizes_[i] * sizeof(float),
                       cudaMemcpyDeviceToHost, streams_[i]);
    }
    
    // Synchronize all devices
    for (int i = 0; i < num_gpus_; i++) {
        cudaSetDevice(i);
        cudaStreamSynchronize(streams_[i]);
    }
}

}} // namespace surprise_metrics::cuda
