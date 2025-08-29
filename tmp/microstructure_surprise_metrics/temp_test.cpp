#include <stdexcept>
#include <vector>
typedef int cudaError_t;
#define cudaSuccess 0
#define cudaMemcpyDeviceToHost 0
#define cudaMemcpyHostToDevice 0
cudaError_t cudaMemcpy(void*, const void*, size_t, int) { return 0; }
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
        std::vector<bool> jump_flags(n_returns);
        std::vector<float> burst_scores(n_trades);
        std::vector<bool> burst_flags(n_trades);
        std::vector<float> branching_ratio(n_trades);
        std::vector<float> endogeneity(n_trades);
        
        err = cudaMemcpy(bns_stats.data(), d_bns_stats_, n_returns * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) throw std::runtime_error("Failed to copy BNS stats from GPU");
        
        err = cudaMemcpy(lm_stats.data(), d_lm_stats, n_returns * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) throw std::runtime_error("Failed to copy LM stats from GPU");
        
        err = cudaMemcpy(jump_flags.data(), d_jump_flags_, n_returns * sizeof(bool), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) throw std::runtime_error("Failed to copy jump flags from GPU");
        
        err = cudaMemcpy(burst_scores.data(), d_burst_scores_, n_trades * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) throw std::runtime_error("Failed to copy burst scores from GPU");
        
        err = cudaMemcpy(burst_flags.data(), d_burst_flags_, n_trades * sizeof(bool), cudaMemcpyDeviceToHost);
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
