#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cstring>
#include "surprise_metrics.h"

using namespace surprise_metrics;

std::vector<Trade> generate_test_trades_with_jumps() {
    std::vector<Trade> trades;
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Much smaller normal price movements - typical stock volatility
    std::normal_distribution<> normal_return_dist(0.0, 0.001); // 0.1% daily vol
    std::poisson_distribution<> size_dist(100);
    
    double current_price = 100.0;
    auto start_time = std::chrono::steady_clock::now().time_since_epoch().count();
    
    // Generate realistic price series with actual detectable jumps
    for (int i = 0; i < 1000; ++i) {
        Trade trade;
        trade.timestamp = std::chrono::nanoseconds(start_time + i * 1000000LL);
        
        // Add large jumps at specific intervals
        double return_shock = 0.0;
        if (i > 50 && (i % 100 == 23 || i % 137 == 7)) {
            // Add significant jumps: 1-5% moves (much larger than normal 0.1% vol)
            std::uniform_real_distribution<> jump_dist(-0.05, 0.05);
            return_shock = jump_dist(gen);
            std::cout << "Added jump at trade " << i << " with size " << return_shock << std::endl;
        }
        
        // Normal small price movement plus any jump
        double normal_return = normal_return_dist(gen);
        double total_return = normal_return + return_shock;
        
        current_price *= (1.0 + total_return);
        trade.price = current_price;
        trade.size = size_dist(gen);
        trade.exchange = 'N';
        std::memset(trade.conditions, 0, sizeof(trade.conditions));
        
        trades.push_back(trade);
    }
    
    return trades;
}

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
        // Use fewer GPUs or 0 to test CPU path first
        MetricsCalculator calculator(0, 10000);  // CPU-only for testing
        
        // Set more sensitive parameters for jump detection
        calculator.set_garch_params(0.00001, 0.05, 0.94);
        calculator.set_jump_threshold(3.0);  // Lower threshold
        calculator.set_window_size(50);      // Smaller window
        
        std::cout << "Generating test data with artificial jumps...\n";
        auto trades = generate_test_trades_with_jumps();
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
        
        // Detailed analysis
        int jump_count = 0;
        float max_zscore = 0.0f;
        float max_lm_stat = 0.0f;
        float max_bns_stat = 0.0f;
        
        std::cout << "\nDetailed Analysis:\n";
        std::cout << "First 10 metrics:\n";
        for (size_t i = 0; i < std::min(size_t(10), metrics.size()); ++i) {
            const auto& m = metrics[i];
            std::cout << "  [" << i << "] Return: " << m.standardized_return 
                      << ", LM: " << m.lee_mykland_stat 
                      << ", BNS: " << m.bns_stat 
                      << ", Jump: " << (m.jump_detected ? "YES" : "NO") << "\n";
        }
        
        for (const auto& m : metrics) {
            if (m.jump_detected) jump_count++;
            max_zscore = std::max(max_zscore, std::abs(m.standardized_return));
            max_lm_stat = std::max(max_lm_stat, m.lee_mykland_stat);
            max_bns_stat = std::max(max_bns_stat, std::abs(m.bns_stat));
        }
        
        std::cout << "\nResults Summary:\n";
        std::cout << "  Total Metrics: " << metrics.size() << "\n";
        std::cout << "  Jumps Detected: " << jump_count << "\n";
        std::cout << "  Max Z-Score: " << max_zscore << "\n";
        std::cout << "  Max LM Statistic: " << max_lm_stat << "\n";
        std::cout << "  Max BNS Statistic: " << max_bns_stat << "\n";
        
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
