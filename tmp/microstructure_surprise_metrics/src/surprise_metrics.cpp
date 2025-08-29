#include "surprise_metrics.h"
#include "simd_ops.h"
#include "cuda_kernels.cuh"
#include "multi_gpu.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>
#include <cuda_runtime.h>

namespace surprise_metrics {

class MetricsCalculator::Impl {
public:
    Impl(int num_gpus, size_t buffer_size) 
        : num_gpus_(num_gpus), buffer_size_(buffer_size) {
        
        // Initialize CUDA if GPUs available
        if (num_gpus > 0) {
            init_cuda();
        }
    }
    
    ~Impl() {
        cleanup_cuda();
    }

    void process_quotes(const std::vector<Quote>& quotes) {
        // Process quote data for microstructure metrics
        // Could add GPU processing for spread, order imbalance, etc.
    }
    
    void process_trades(const std::vector<Trade>& trades) {
        if (trades.empty()) {
            std::cerr << "No trades to process\n";
            return;
        }
        
        // Extract prices and timestamps
        price_buffer_.clear();
        price_buffer_.reserve(trades.size());
        timestamp_buffer_.clear();
        timestamp_buffer_.reserve(trades.size());
        
        for (const auto& trade : trades) {
            price_buffer_.push_back(static_cast<float>(trade.price));
            timestamp_buffer_.push_back(static_cast<float>(trade.timestamp.count()));
        }
        
        std::cout << "Extracted " << price_buffer_.size() << " prices\n";
        
        // Compute returns using SIMD
        if (price_buffer_.size() < 2) {
            std::cerr << "Not enough prices for returns\n";
            return;
        }
        
        return_buffer_.resize(price_buffer_.size() - 1);
        simd::compute_returns_avx512(
            price_buffer_.data(),
            return_buffer_.data(),
            price_buffer_.size()
        );
        
        std::cout << "Computed " << return_buffer_.size() << " returns\n";
        
        // Process on GPU if available, otherwise CPU
	std::cout << "Debug: num_gpus_=" << num_gpus_ << ", cuda_available_=" << cuda_available_ << std::endl;
        if (num_gpus_ > 0 && cuda_available_) {
	    std::cout << "Using GPU processing" << std::endl;
            process_gpu(trades);
        } else {
	    std::cout << "Using CPU processing (GPUs: " << num_gpus_ << ", CUDA available: " << cuda_available_ << ")" << std::endl;
            process_cpu(trades);
        }
    }
    
    std::vector<SurpriseMetrics> get_metrics() const {
        return metrics_buffer_;
    }
    
    void set_garch_params(double omega, double alpha, double beta) {
        garch_omega_ = omega;
        garch_alpha_ = alpha;
        garch_beta_ = beta;
    }
    
    void set_jump_threshold(double threshold) {
        jump_threshold_ = threshold;
    }
    
    void set_window_size(int window) {
        window_size_ = window;
    }
    
private:
    int num_gpus_;
    size_t buffer_size_;
    int window_size_ = 100;
    double garch_omega_ = 0.00001;
    double garch_alpha_ = 0.05;
    double garch_beta_ = 0.94;
    //double jump_threshold_ = 4.6055;  // Corrected from paper
    double jump_threshold_ = 2; //4.6055;  // Corrected from paper
    
    // Hawkes parameters
    float hawkes_mu_ = 0.1f;
    float hawkes_phi_ = 0.3f;
    float hawkes_kappa_ = 0.8f;
    
    // Buffers
    std::vector<float> price_buffer_;
    std::vector<float> return_buffer_;
    std::vector<float> sigma_buffer_;
    std::vector<float> timestamp_buffer_;
    std::vector<SurpriseMetrics> metrics_buffer_;
    
    // CUDA resources
    bool cuda_available_ = false;
    float* d_returns_ = nullptr;
    float* d_sigma_ = nullptr;
    float* d_timestamps_ = nullptr;
    float* d_prices_ = nullptr;
    float* d_rv_ = nullptr;
    float* d_bv_ = nullptr;
    float* d_tq_ = nullptr;
    float* d_bns_stats_ = nullptr;
    char* d_jump_flags_ = nullptr;  // Use char instead of bool for CUDA compatibility
    float* d_hawkes_intensity_ = nullptr;
    float* d_branching_ratio_ = nullptr;
    float* d_endogeneity_ = nullptr;
    float* d_poisson_intensity_ = nullptr;
    float* d_poisson_surprise_ = nullptr;
    char* d_burst_flags_ = nullptr;  // Use char instead of bool for CUDA compatibility
    float* d_burst_scores_ = nullptr;
    
    void init_cuda() {
        cudaError_t err = cudaGetDeviceCount(&num_gpus_);
        if (err != cudaSuccess || num_gpus_ == 0) {
            cuda_available_ = false;
            return;
        }
        
        cuda_available_ = true;
        size_t alloc_size = buffer_size_ * sizeof(float);
        
        // Allocate GPU memory
        cudaMalloc(&d_returns_, alloc_size);
        cudaMalloc(&d_sigma_, alloc_size);
        cudaMalloc(&d_timestamps_, alloc_size);
        cudaMalloc(&d_prices_, alloc_size);
        cudaMalloc(&d_rv_, alloc_size);
        cudaMalloc(&d_bv_, alloc_size);
        cudaMalloc(&d_tq_, alloc_size);
        cudaMalloc(&d_bns_stats_, alloc_size);
        cudaMalloc(&d_jump_flags_, buffer_size_ * sizeof(char));
        cudaMalloc(&d_hawkes_intensity_, alloc_size);
        cudaMalloc(&d_branching_ratio_, alloc_size);
        cudaMalloc(&d_endogeneity_, alloc_size);
        cudaMalloc(&d_poisson_intensity_, alloc_size);
        cudaMalloc(&d_poisson_surprise_, alloc_size);
        cudaMalloc(&d_burst_flags_, buffer_size_ * sizeof(char));
        cudaMalloc(&d_burst_scores_, alloc_size);
        
        std::cout << "Initialized CUDA with " << num_gpus_ << " GPUs\n";
    }
    
    void cleanup_cuda() {
        if (!cuda_available_) return;
        
        cudaFree(d_returns_);
        cudaFree(d_sigma_);
        cudaFree(d_timestamps_);
        cudaFree(d_prices_);
        cudaFree(d_rv_);
        cudaFree(d_bv_);
        cudaFree(d_tq_);
        cudaFree(d_bns_stats_);
        cudaFree(d_jump_flags_);
        cudaFree(d_hawkes_intensity_);
        cudaFree(d_branching_ratio_);
        cudaFree(d_endogeneity_);
        cudaFree(d_poisson_intensity_);
        cudaFree(d_poisson_surprise_);
        cudaFree(d_burst_flags_);
        cudaFree(d_burst_scores_);
    }
    
    void process_gpu(const std::vector<Trade>& trades) {
        std::cout << "Processing on GPU...\n";
        
        size_t n_returns = return_buffer_.size();
        size_t n_trades = trades.size();
        
        // Copy data to GPU
        cudaError_t err = cudaMemcpy(d_returns_, return_buffer_.data(), n_returns * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) throw std::runtime_error("Failed to copy returns to GPU");
        
        err = cudaMemcpy(d_timestamps_, timestamp_buffer_.data(), n_trades * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) throw std::runtime_error("Failed to copy timestamps to GPU");
        
        err = cudaMemcpy(d_prices_, price_buffer_.data(), n_trades * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) throw std::runtime_error("Failed to copy prices to GPU");
        
        // 1. GARCH volatility
        sigma_buffer_.resize(n_returns);
        cuda::launch_garch_estimation(d_returns_, d_sigma_, n_returns, 
                                     garch_omega_, garch_alpha_, garch_beta_);
        
        // 2. BNS jump detection
        cuda::launch_bns_computation(d_returns_, d_rv_, d_bv_, d_tq_, 
                                    d_bns_stats_, n_returns, window_size_);
        
        // 3. Lee-Mykland jump detection
        float* d_local_vol;
        float* d_lm_stats;
        cudaMalloc(&d_local_vol, n_returns * sizeof(float));
        cudaMalloc(&d_lm_stats, n_returns * sizeof(float));
        
        cuda::launch_lee_mykland_computation(
            d_returns_, d_local_vol, d_lm_stats, d_jump_flags_,
            n_returns, window_size_, jump_threshold_
        );

	//cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
    		std::cerr << "Kernel failed: " << cudaGetErrorString(err) << std::endl;
    	// Fall back to CPU processing
    		process_cpu(trades);
    		return;
	}
        
        // 4. Hawkes intensity with branching ratio
        cuda::launch_hawkes_computation(
            d_timestamps_, d_hawkes_intensity_, d_branching_ratio_, d_endogeneity_,
            n_trades, hawkes_mu_, hawkes_phi_, hawkes_kappa_
        );
        
        // 5. Poisson baseline
        cuda::launch_poisson_computation(
            d_timestamps_, d_prices_, d_poisson_intensity_, d_poisson_surprise_,
            n_trades, window_size_, 60.0f
        );
        
        // 6. Burst detection
        cuda::launch_burst_computation(
            d_hawkes_intensity_, d_poisson_surprise_, d_branching_ratio_,
            d_burst_flags_, d_burst_scores_ ,
            n_trades, 0.8f, 3.0f, 4.0f
        );
        
        // Copy results back from GPU
        err = cudaMemcpy(sigma_buffer_.data(), d_sigma_, n_returns * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) throw std::runtime_error("Failed to copy sigma buffer from GPU");
        
        std::vector<float> bns_stats(n_returns);
        std::vector<float> lm_stats(n_returns);
        std::vector<char> jump_flags(n_returns);  // Use char instead of bool for CUDA compatibility
        std::vector<float> burst_scores(n_trades);
        std::vector<char> burst_flags(n_trades);  // Use char instead of bool for CUDA compatibility
        std::vector<float> branching_ratio(n_trades);
        std::vector<float> endogeneity(n_trades);
        
        err = cudaMemcpy(bns_stats.data(), d_bns_stats_, n_returns * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) throw std::runtime_error("Failed to copy BNS stats from GPU");
        
        err = cudaMemcpy(lm_stats.data(), d_lm_stats, n_returns * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) throw std::runtime_error("Failed to copy LM stats from GPU");
        
        err = cudaMemcpy(jump_flags.data(), d_jump_flags_, n_returns * sizeof(char), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) throw std::runtime_error("Failed to copy jump flags from GPU");
        
        err = cudaMemcpy(burst_scores.data(), d_burst_scores_, n_trades * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) throw std::runtime_error("Failed to copy burst scores from GPU");
        
        err = cudaMemcpy(burst_flags.data(), d_burst_flags_, n_trades * sizeof(char), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) throw std::runtime_error("Failed to copy burst flags from GPU");
        
        err = cudaMemcpy(branching_ratio.data(), d_branching_ratio_, n_trades * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) throw std::runtime_error("Failed to copy branching ratio from GPU");
        
        err = cudaMemcpy(endogeneity.data(), d_endogeneity_, n_trades * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) throw std::runtime_error("Failed to copy endogeneity from GPU");
        
        // Clean up temp allocations
        cudaFree(d_local_vol);
        cudaFree(d_lm_stats);
        
        // Generate final metrics
        metrics_buffer_.clear();
        for (size_t i = window_size_; i < n_returns && i < trades.size(); ++i) {
            SurpriseMetrics metric;
            metric.timestamp = trades[i].timestamp;
            
            // Price metrics
            metric.standardized_return = return_buffer_[i] / (sigma_buffer_[i] > 0 ? sigma_buffer_[i] : 1.0f);
            metric.lee_mykland_stat = lm_stats[i];
            metric.bns_stat = bns_stats[i];
            metric.jump_detected = jump_flags[i];
            
            // Trade intensity from burst detection
            metric.trade_intensity_zscore = burst_scores[i];
            
            metrics_buffer_.push_back(metric);
        }
        
        // Summary statistics
        int total_jumps = std::count(jump_flags.begin(), jump_flags.end(), 1);  // Use 1 instead of true for char
        int total_bursts = std::count(burst_flags.begin(), burst_flags.end(), 1);  // Use 1 instead of true for char
        float avg_branching = std::accumulate(branching_ratio.begin(), branching_ratio.end(), 0.0f) / branching_ratio.size();
        
        std::cout << "GPU Processing Complete:\n";
        std::cout << "  Jumps detected: " << total_jumps << " (" << (100.0f * total_jumps / n_returns) << "%)\n";
        std::cout << "  Trade bursts: " << total_bursts << " (" << (100.0f * total_bursts / n_trades) << "%)\n";
        std::cout << "  Avg branching ratio: " << avg_branching << "\n";
        std::cout << "  Computed " << metrics_buffer_.size() << " metrics\n";
    }
    
    void process_cpu(const std::vector<Trade>& trades) {
        std::cout << "Processing on CPU...\n";
        
        // Compute volatility
        sigma_buffer_.resize(return_buffer_.size());
        compute_volatility();
        
        // Compute RV and BV using SIMD
        std::vector<float> rv(return_buffer_.size(), 0.0f);
        simd::compute_realized_variance_avx512(
            return_buffer_.data(), rv.data(), return_buffer_.size(), window_size_
        );
        
        std::vector<float> bv(return_buffer_.size(), 0.0f);
        simd::compute_bipower_variation_avx512(
            return_buffer_.data(), bv.data(), return_buffer_.size()
        );
        
        // Generate metrics
        metrics_buffer_.clear();
        for (size_t i = window_size_; i < return_buffer_.size() && i < trades.size(); ++i) {
            SurpriseMetrics metric;
            metric.timestamp = trades[i].timestamp;
            
            // Standardized return
            metric.standardized_return = return_buffer_[i] / (sigma_buffer_[i] > 0 ? sigma_buffer_[i] : 1.0f);
            
            // Lee-Mykland statistic
            float local_vol = std::sqrt(std::max(0.0f, bv[i]));
            metric.lee_mykland_stat = std::abs(return_buffer_[i]) / (local_vol > 0 ? local_vol : 1.0f);
            
            // BNS statistic
            /*if (rv[i] > 0) {
                float jump_component = rv[i] - bv[i];
                metric.bns_stat = std::sqrt(window_size_) * jump_component / rv[i];
            } else {
                metric.bns_stat = 0;
            }*/
	    // Need to compute tri-power quarticity (TQ) for proper BNS
	    if (rv[i] > 0 && bv[i] > 0) {
		    float jump_component = std::max(0.0f, rv[i] - bv[i]);
		    // This needs TQ calculation - currently missing
		    float theta = 0.609f;
		    // Proper formula needs: sqrt(window) * (jump_component/RV) / sqrt(theta * TQ/BV^2)
		    metric.bns_stat = std::sqrt(window_size_) * jump_component / rv[i]; // Incomplete formula
	    } else {
		    metric.bns_stat = 0;
	    }
            
            // Jump detection
            /*float Cn = std::sqrt(2.0 * std::log(return_buffer_.size()));
            float Sn = 1.0f/Cn + (std::log(M_PI) + std::log(2.0 * std::log(return_buffer_.size()))) / (2.0f * Cn);
            float critical_value = jump_threshold_ + Cn * Sn;
	    */
	    //float Cn = sqrtf(2.0f * logf(float(n)));
	    float Cn = sqrtf(2.0f * logf(float(return_buffer_.size())));
	    float beta_star = 0.49; // From Lee-Mykland paper
	    float critical_value = beta_star * Cn;

            metric.jump_detected = (metric.lee_mykland_stat > critical_value);
            
            // Trade intensity (placeholder for CPU version)
            metric.trade_intensity_zscore = 0.0f;
            
            metrics_buffer_.push_back(metric);
        }
        
        std::cout << "Computed " << metrics_buffer_.size() << " metrics\n";
    }
    
    void compute_volatility() {
        if (sigma_buffer_.empty()) return;
        
        sigma_buffer_[0] = std::sqrt(garch_omega_ / (1.0 - garch_alpha_ - garch_beta_));
        
        for (size_t i = 1; i < return_buffer_.size(); ++i) {
            float r2 = return_buffer_[i-1] * return_buffer_[i-1];
            float sigma2 = garch_omega_ + garch_alpha_ * r2 + 
                          garch_beta_ * sigma_buffer_[i-1] * sigma_buffer_[i-1];
            sigma_buffer_[i] = std::sqrt(sigma2);
        }
    }
};

// MetricsCalculator implementation
MetricsCalculator::MetricsCalculator(int num_gpus, size_t buffer_size)
    : pImpl(std::make_unique<Impl>(num_gpus, buffer_size)) {}

MetricsCalculator::~MetricsCalculator() = default;

void MetricsCalculator::process_trades(const std::vector<Trade>& trades) {
    pImpl->process_trades(trades);
}

void MetricsCalculator::process_quotes(const std::vector<Quote>& quotes) {
    pImpl->process_quotes(quotes);
}

std::vector<SurpriseMetrics> MetricsCalculator::get_metrics() const {
    return pImpl->get_metrics();
}

void MetricsCalculator::set_garch_params(double omega, double alpha, double beta) {
    pImpl->set_garch_params(omega, alpha, beta);
}

void MetricsCalculator::set_jump_threshold(double threshold) {
    pImpl->set_jump_threshold(threshold);
}

void MetricsCalculator::set_window_size(int window) {
    pImpl->set_window_size(window);
}

} // namespace surprise_metrics
