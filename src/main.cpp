#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cstring>
#include "surprise_metrics.h"

using namespace surprise_metrics;

int main(int argc, char* argv[]) {
    std::cout << "SurpriseMetrics Runner v0.1.0\n";
    std::cout << "=============================\n\n";
    
    // Check CUDA availability
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        device_count = 0;
    }
    
    std::cout << "CUDA Devices Found: " << device_count << std::endl;
    
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "  Device " << i << ": " << prop.name 
                  << " (SM " << prop.major << "." << prop.minor << ")"
                  << " Memory: " << prop.totalGlobalMem / (1024*1024) << " MB\n";
    }
    
    std::cout << "\nInitializing MetricsCalculator...\n";
    
    try {
        // Initialize calculator with NO GPUs for now to avoid CUDA allocation issues
        //MetricsCalculator calculator(0, 10000);  // 0 GPUs, smaller buffer
        MetricsCalculator calculator(device_count, 10000);  // 4 GPUs, smaller buffer
        
        // Set parameters
        calculator.set_garch_params(0.00001, 0.05, 0.94);
        calculator.set_jump_threshold(4.0);
        calculator.set_window_size(100);
        
        std::cout << "Generating test data...\n";
        
        // Generate smaller test dataset
        std::vector<Trade> trades;
        trades.reserve(1000);  // Only 1000 trades
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> price_dist(100.0, 0.5);
        std::poisson_distribution<> size_dist(100);
        
        auto start_time = std::chrono::steady_clock::now();
        
        for (int i = 0; i < 1000; ++i) {
            Trade trade;
            trade.timestamp = std::chrono::nanoseconds(
                start_time.time_since_epoch().count() + i * 1000000
            );
            trade.price = price_dist(gen);
            trade.size = size_dist(gen);
            trade.exchange = 'N';  // NYSE
            std::memset(trade.conditions, 0, sizeof(trade.conditions));
            trades.push_back(trade);
        }
        
        std::cout << "Generated " << trades.size() << " trades\n";
        
        // Process trades
        std::cout << "Processing trades...\n";
        auto process_start = std::chrono::high_resolution_clock::now();
        
        calculator.process_trades(trades);
        
        auto process_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(process_end - process_start);
        
        std::cout << "Processing completed in " << duration.count() << " ms\n";
        
        // Get metrics
        auto metrics = calculator.get_metrics();
        
        // Print summary
        int jump_count = 0;
        float max_zscore = 0.0f;
        for (const auto& m : metrics) {
            if (m.jump_detected) jump_count++;
            max_zscore = std::max(max_zscore, std::abs(m.standardized_return));
        }
        
        std::cout << "\nResults Summary:\n";
        std::cout << "  Total Metrics: " << metrics.size() << "\n";
        std::cout << "  Jumps Detected: " << jump_count << "\n";
        std::cout << "  Max Z-Score: " << max_zscore << "\n";
        
        if (duration.count() > 0) {
            std::cout << "  Throughput: " << (trades.size() * 1000.0 / duration.count()) 
                      << " trades/sec\n";
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
