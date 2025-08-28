#include "surprise_metrics.h"
#include "simd_ops.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>

#ifdef HAS_EIGEN
#include <Eigen/Dense>
#endif

namespace surprise_metrics {

class MetricsCalculator::Impl {
public:
    Impl(int num_gpus, size_t buffer_size) 
        : num_gpus_(num_gpus), buffer_size_(buffer_size) {
        
        // Don't pre-allocate, let vectors grow naturally
        // This avoids bad_alloc from aligned allocator
        
        // Skip CUDA initialization for now
        // if (num_gpus > 0) {
        //     init_cuda();
        // }
    }
    
    ~Impl() {
        cleanup_cuda();
    }
    
    void process_trades(const std::vector<Trade>& trades) {
        if (trades.empty()) {
            std::cerr << "No trades to process\n";
            return;
        }
        
        // Extract prices
        price_buffer_.clear();
        price_buffer_.reserve(trades.size());
        
        for (const auto& trade : trades) {
            price_buffer_.push_back(static_cast<float>(trade.price));
        }
        
        std::cout << "Extracted " << price_buffer_.size() << " prices\n";
        
        // Compute returns
        if (price_buffer_.size() < 2) {
            std::cerr << "Not enough prices for returns\n";
            return;
        }
        
        return_buffer_.resize(price_buffer_.size() - 1);
        compute_returns();
        
        std::cout << "Computed " << return_buffer_.size() << " returns\n";
        
        // Compute volatility
        sigma_buffer_.resize(return_buffer_.size());
        compute_volatility();
        
        std::cout << "Computed volatility\n";
        
        // Compute metrics
        compute_all_metrics(trades);
        
        std::cout << "Computed " << metrics_buffer_.size() << " metrics\n";
    }
    
    void process_quotes(const std::vector<Quote>& quotes) {
        // Process quote data for microstructure metrics
        // Implementation here...
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
    double jump_threshold_ = 4.0;
    
    // Use regular vectors instead of aligned_vector to avoid allocation issues
    std::vector<float> price_buffer_;
    std::vector<float> return_buffer_;
    std::vector<float> sigma_buffer_;
    std::vector<SurpriseMetrics> metrics_buffer_;
    
    // CUDA resources (unused for now)
    void* cuda_context_ = nullptr;
    float* d_prices_ = nullptr;
    float* d_returns_ = nullptr;
    float* d_sigma_ = nullptr;
    
    void init_cuda() {
        // Skip CUDA allocation for now
    }
    
    void cleanup_cuda() {
        // Skip CUDA cleanup for now
    }
    
    void compute_returns() {
        simd::compute_returns_avx512(
            price_buffer_.data(),
            return_buffer_.data(),
            price_buffer_.size()
        );
    }
    
    void compute_volatility() {
        // EWMA volatility estimation (simplified)
        if (sigma_buffer_.empty()) return;
        
        sigma_buffer_[0] = std::sqrt(garch_omega_ / (1.0 - garch_alpha_ - garch_beta_));
        
        for (size_t i = 1; i < return_buffer_.size(); ++i) {
            float r2 = return_buffer_[i-1] * return_buffer_[i-1];
            float sigma2 = garch_omega_ + garch_alpha_ * r2 + 
                          garch_beta_ * sigma_buffer_[i-1] * sigma_buffer_[i-1];
            sigma_buffer_[i] = std::sqrt(sigma2);
        }
    }
    
    void compute_all_metrics(const std::vector<Trade>& trades) {
        metrics_buffer_.clear();
        
        if (return_buffer_.size() < window_size_) {
            std::cerr << "Not enough returns for window size " << window_size_ << "\n";
            return;
        }
        
        // Use regular vectors
        std::vector<float> rv(return_buffer_.size(), 0.0f);
        simd::compute_realized_variance_avx512(
            return_buffer_.data(),
            rv.data(),
            return_buffer_.size(),
            window_size_
        );
        
        std::vector<float> bv(return_buffer_.size(), 0.0f);
        simd::compute_bipower_variation_avx512(
            return_buffer_.data(),
            bv.data(),
            return_buffer_.size()
        );
        
        // Generate metrics starting from window_size_
        for (size_t i = window_size_; i < return_buffer_.size() && i < trades.size(); ++i) {
            SurpriseMetrics metric;
            metric.timestamp = trades[i].timestamp;
            
            // Standardized return
            if (sigma_buffer_[i] > 0) {
                metric.standardized_return = return_buffer_[i] / sigma_buffer_[i];
            } else {
                metric.standardized_return = 0;
            }
            
            // Lee-Mykland statistic
            float local_vol = std::sqrt(std::max(0.0f, bv[i] / window_size_));
            if (local_vol > 0) {
                metric.lee_mykland_stat = std::abs(return_buffer_[i]) / local_vol;
            } else {
                metric.lee_mykland_stat = 0;
            }
            
            // BNS statistic
            if (rv[i] > 0) {
                float jump_component = rv[i] - bv[i];
                metric.bns_stat = std::sqrt(window_size_) * jump_component / rv[i];
            } else {
                metric.bns_stat = 0;
            }
            
            // Trade intensity (simplified)
            metric.trade_intensity_zscore = 0.0; // Placeholder
            
            // Jump detection
            float Cn = std::sqrt(2.0 * std::log(return_buffer_.size()));
            float Sn = 1.0/Cn + (std::log(M_PI) + std::log(2.0 * std::log(return_buffer_.size()))) / (2.0 * Cn);
            float critical_value = jump_threshold_ + Cn * Sn;
            metric.jump_detected = (metric.lee_mykland_stat > critical_value);
            
            metrics_buffer_.push_back(metric);
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
