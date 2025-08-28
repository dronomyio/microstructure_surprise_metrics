# intuition 

 intuition behind these surprise metrics is to detect **when something unusual is happening** in financial markets that warrants immediate attention. Let me break down each component:

## Standardized Returns (zₜ = rₜ/σ̂ₜ)

This normalizes returns by their predicted volatility. The intuition is simple: a 2% move when volatility is low (say σ = 0.5%) is much more surprising than a 2% move when volatility is high (σ = 2%). By dividing returns by estimated volatility, you get a measure of "how many standard deviations" the move represents - making surprises comparable across different volatility regimes.

## Jump Detection (Lee-Mykland/BNS Statistics)

These statistics identify price jumps that are too large to be explained by normal diffusive volatility. The core insight is that in high-frequency data, if you look at very short intervals, the continuous component of price movement becomes negligible while jumps remain prominent. 

Check some videos
a) https://www.youtube.com/watch?v=xc_X9GFVuVU

b) ref: https://galton.uchicago.edu/~mykland/paperlinks/LeeMykland-2535.pdf

## The Lee-Mykland Jump Detection Test Explained

The paper addresses a critical problem: how do you detect "jumps" (sudden price movements from news/events) in high-frequency trading data without being fooled by normal volatility changes?

### Core Concept Visualization

```
Normal Trading vs Jump Event:

Normal:  ──╱╲─╱╲──╱╲───  (small fluctuations around trend)
         
Jump:    ────╱╲──│────   (sudden discontinuous move)
                  │
                  └── JUMP! (earnings, news, etc.)
```

### The Mathematical Framework

**1. The Model:**
```
Price = Continuous Part + Jump Part
dp(t) = μdt + σdW(t) + dJ(t)
        ↑       ↑         ↑
     drift  volatility  jumps
```

**2. The Test Statistic:**
```
L(i) = |return(i)| / local_volatility(i)

where local_volatility uses Bipower Variation:
BV = (π/2) × Σ|r(i)| × |r(i-1)|  ← robust to jumps!
```

### Why Bipower Variation?

```
Regular Variance:  Σr²  ← contaminated by jumps
Bipower Variation: Σ|r(i)||r(i-1)| ← ignores jumps

Example with jump at t=3:
r = [0.01, -0.02, 0.15, 0.01, -0.01]  ← 0.15 is jump
                    ↑
Variance includes this²
Bipower uses |0.02|×|0.15| and |0.15|×|0.01| ← diluted
```

### The Detection Process

```
Step 1: Calculate returns over small intervals (5-min)
Step 2: Compute local volatility using K-period window
Step 3: Standardize current return by local volatility
Step 4: Compare to threshold adjusted for multiple testing

      If L(i) > β* → JUMP DETECTED
      
      where β* ≈ -log(-log(1-α)) + Cn×Sn
            Cn = (2log n)^0.5
            Sn ≈ 1/Cn
```

### Your Implementation Issues

Looking at your code in `src/surprise_metrics.cpp`:

```cpp
// Line 165-169 - YOUR CODE:
float local_vol = std::sqrt(bv[i] / window_size);  // ← WRONG!
```

**The Problem:** You're dividing bipower variation by window_size, but BV is already a variance estimate. Should be:

```cpp
// CORRECTED:
float local_vol = std::sqrt(bv[i]);  // No division needed
```

Also, your threshold calculation:
```cpp
// Line 172-175 - YOUR CODE:
float Cn = std::sqrt(2.0 * std::log(return_buffer_.size()));
float critical_value = jump_threshold_ + Cn * Sn;  // threshold=4.0
```

The paper recommends `β* ≈ 4.6055` for α=0.01. Your value of 4.0 is too low.

### Visual Example of Your Problem

```
Your Current Detection:
Returns: ──╱╲─╱╲──╱╲─╱╲─╱╲──
           ↑  ↑  ↑  ↑  ↑
         jump jump jump jump  ← 60% false positives!

Correct Detection:
Returns: ──╱╲─╱╲──╱╲─╱╲─╱╲──
                    ↑
                 real jump  ← 1-5% detection rate
```

### Fix Required

In `src/surprise_metrics.cpp`, change:
1. Line 165: Remove `/window_size` from local_vol calculation
2. Line 174: Change `jump_threshold_` from 4.0 to 4.6055
3. Verify bipower variation calculation in `simd_ops.cpp` is summing over the window

This should reduce your jump detection from 60% to a more realistic 1-5%.

./surprise_metrics_runner 
SurpriseMetrics Runner v0.1.0
=============================

CUDA Devices Found: 4
  Device 0: NVIDIA GeForce RTX 3070 (SM 8.6) Memory: 7958 MB
  Device 1: NVIDIA GeForce RTX 3070 (SM 8.6) Memory: 7966 MB
  Device 2: NVIDIA GeForce RTX 3070 (SM 8.6) Memory: 7966 MB
  Device 3: NVIDIA GeForce RTX 3070 (SM 8.6) Memory: 7966 MB

Initializing MetricsCalculator...
Generating test data...
Generated 1000 trades
Processing trades...
Extracted 1000 prices
Using AVX2 optimized path
Computed 999 returns
Computed volatility
Computed 899 metrics
Processing completed in 0 ms

Results Summary:
  Total Metrics: 899
  Jumps Detected: 8  with window false detection 540.
  Max Z-Score: 1.59997


The **Lee-Mykland statistic** specifically compares each return to a local volatility estimate from surrounding returns. When this ratio exceeds a threshold (derived from extreme value theory), it flags a jump. This helps separate genuine news-driven discontinuities from normal market noise.

The **BNS (Barndorff-Nielsen & Shephard)** approach uses bipower variation to estimate continuous volatility robust to jumps, then identifies jumps as deviations from this baseline.

## Abnormal Trade-Arrival Bursts

This monitors the **intensity of trading activity** relative to recent patterns. The intuition is that informed traders or significant news events create clustering in trade arrivals - suddenly everyone wants to trade at once. By modeling the normal arrival rate (often as a Poisson or Hawkes process) and detecting deviations, you can identify periods of abnormal information flow or liquidity demand.

## Core Alerting Philosophy

Together, these metrics implement a multi-dimensional surprise detection system:
- **Standardized returns** catch price surprises (adjusted for current volatility)
- **Jump flags** identify discontinuous moves (potential news or liquidity events)  
- **Trade bursts** detect activity surprises (information arrival or urgency)

The combination is powerful because different market events leave different "fingerprints" - a earnings surprise might show as a jump with high trade intensity, while a liquidity crisis might show abnormal returns without jumps but with trade clustering. By monitoring all three dimensions, you get robust detection of market stress, information events, and regime changes that simple threshold-based systems would miss.

# Mathematical framework behind these surprise metrics:

## 1. Standardized Returns

The standardized return is:
$$z_t = \frac{r_t}{\hat{\sigma}_t}$$

Where $\hat{\sigma}_t$ is typically estimated using:

**GARCH(1,1):**
$$\hat{\sigma}^2_t = \omega + \alpha r^2_{t-1} + \beta \hat{\sigma}^2_{t-1}$$

**Realized Volatility (RV):**
$$\hat{\sigma}^2_t = \sum_{i=1}^{n} r^2_{t-i,\Delta}$$
where $r_{t-i,\Delta}$ are intraday returns at frequency $\Delta$

**EWMA:**
$$\hat{\sigma}^2_t = \lambda \hat{\sigma}^2_{t-1} + (1-\lambda) r^2_{t-1}$$

Under the null hypothesis of no surprise: $z_t \sim \mathcal{N}(0,1)$

## 2. Jump Detection

### Lee-Mykland Statistic

For log-prices following: $dp_t = \mu_t dt + \sigma_t dW_t + dJ_t$

The test statistic is:
$$\mathcal{L}_i = \frac{|r_i|}{\hat{\sigma}_i}$$

where $\hat{\sigma}_i$ is the local bipower variation:
$$\hat{\sigma}^2_i = \frac{1}{K-2} \frac{\pi}{2} \sum_{j=i-K+2}^{i-1} |r_j||r_{j-1}|$$

The maximum statistic over window $[t-1,t]$ with $n$ observations:
$$\xi_t = \max_{i \in [t-1,t]} \mathcal{L}_i$$

Jump detected when:
$$\xi_t > g_n + C_n S_n$$

where:
- $C_n = (2\log n)^{1/2}$
- $S_n = \frac{1}{C_n} + \frac{\log(\pi) + \log(2\log n)}{2C_n}$
- $g_n = -\log(-\log(1-\alpha))$ (for significance level $\alpha$)

### BNS (Barndorff-Nielsen & Shephard) Test

**Realized Variance:**
$$RV_t = \sum_{i=1}^{n} r^2_{i,t}$$

**Bipower Variation** (robust to jumps):
$$BV_t = \frac{\pi}{2} \frac{n}{n-1} \sum_{i=2}^{n} |r_{i,t}||r_{i-1,t}|$$

**Jump component:**
$$J_t = RV_t - BV_t$$

**Test statistic:**
$$\mathcal{Z}_{BNS} = \sqrt{n} \frac{J_t/RV_t}{\sqrt{\Theta \cdot TQ_t/BV^2_t}}$$

where:
- $\Theta = (\pi^2/4 + \pi - 5)$ 
- $TQ_t = n \frac{\pi}{2} \frac{n}{n-2} \sum_{i=3}^{n} |r_{i,t}|^{4/3}|r_{i-1,t}|^{4/3}|r_{i-2,t}|^{4/3}$ (Tri-power Quarticity)

Under null (no jumps): $\mathcal{Z}_{BNS} \sim \mathcal{N}(0,1)$

## 3. Abnormal Trade-Arrival Bursts

### Poisson Model
Trade counts $N_t$ in interval $[t, t+\Delta]$:
$$N_t \sim \text{Poisson}(\lambda_t \Delta)$$

**Time-varying intensity:**
$$\lambda_t = \lambda_0 \cdot s_t \cdot \exp(\beta' X_t)$$

where:
- $s_t$ = seasonal component (intraday pattern)
- $X_t$ = covariates (volatility, spread, etc.)

**Surprise metric:**
$$z_{N,t} = \frac{N_t - \hat{\lambda}_t\Delta}{\sqrt{\hat{\lambda}_t\Delta}}$$

### Hawkes Process
Self-exciting intensity:
$$\lambda_t = \mu + \sum_{t_i < t} \phi \cdot e^{-\kappa(t-t_i)}$$

where:
- $\mu$ = baseline intensity
- $\phi$ = jump size in intensity per trade
- $\kappa$ = decay rate

**Branching ratio:** $n = \phi/\kappa$ (must be < 1 for stationarity)

**Abnormal burst detection:**
$$\mathcal{A}_t = \frac{\lambda_t - \mathbb{E}[\lambda]}{\sqrt{\text{Var}[\lambda]}}$$

where $\mathbb{E}[\lambda] = \frac{\mu}{1-n}$

### ACD (Autoregressive Conditional Duration) Model
For inter-trade durations $x_i$:
$$x_i = \psi_i \cdot \epsilon_i$$

**Conditional expected duration:**
$$\psi_i = \omega + \sum_{j=1}^{p} \alpha_j x_{i-j} + \sum_{j=1}^{q} \beta_j \psi_{i-j}$$

**Surprise in trade intensity:**
$$z_{ACD,i} = \frac{x_i - \psi_i}{\psi_i}$$

## 4. Core Alerting Framework

**Composite Score:**
$$\mathcal{S}_t = w_1 \cdot \mathbb{1}_{|z_t| > k_1} + w_2 \cdot \mathbb{1}_{\text{jump}_t} + w_3 \cdot \mathbb{1}_{z_{N,t} > k_2}$$

**Mahalanobis Distance** (multivariate surprise):
$$D_t = \sqrt{(X_t - \mu_X)' \Sigma^{-1}_X (X_t - \mu_X)}$$

where $X_t = [z_t, \mathcal{L}_t, z_{N,t}]'$

**Sequential Detection** (CUSUM):
$$S_t = \max(0, S_{t-1} + z_t - k)$$
Alert when $S_t > h$ (threshold)

**False Discovery Rate Control:**
For multiple testing across assets/timepoints, use Benjamini-Hochberg:
Order p-values: $p_{(1)} \leq p_{(2)} \leq ... \leq p_{(m)}$
Find largest $k$ such that: $p_{(k)} \leq \frac{k}{m} \cdot \alpha_{FDR}$

This mathematical framework provides rigorous statistical testing for market surprises while controlling for multiple testing and maintaining specified false positive rates.

## Project Structure
```
surprise_metrics/
├── CMakeLists.txt
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── setup.py
├── include/
│   ├── surprise_metrics.h
│   ├── simd_ops.h
│   ├── cuda_kernels.cuh
│   └── polygon_parser.h
├── src/
│   ├── surprise_metrics.cpp
│   ├── simd_ops.cpp
│   ├── cuda_kernels.cu
│   ├── polygon_parser.cpp
│   └── main.cpp
├── python/
│   ├── __init__.py
│   ├── api.py
│   └── surprise_metrics.pyx
├── tests/
│   └── test_metrics.py
└── scripts/
    ├── download_polygon_data.sh
    └── run_analysis.py
```

Here's a visualization of the architecture and relationships between the files:

```
                           main.cpp
                              |
                              | creates & uses
                              ↓
                      MetricsCalculator
                    (surprise_metrics.cpp)
                              |
                              | contains Impl class that coordinates
                              ↓
        ┌──────────────┬──────────────┬──────────────┬──────────────┐
        |              |              |              |              |
        ↓              ↓              ↓              ↓              ↓
  simd_ops.cpp   cuda_kernels.cu  multi_gpu.cpp  polygon_parser.cpp  (buffers)
        |              |              |              |
        |              |              |              |
  AVX2/AVX512    GPU kernels    Multi-GPU      Data loading
  compute_       - GARCH        management      from Polygon.io
  returns        - Lee-Mykland  - distribute    CSV files
  - RV           - BNS          - gather        
  - BV           - Hawkes       - streams       

```

**Key relationships:**

1. **main.cpp** → Entry point, creates MetricsCalculator instance, generates test data

2. **surprise_metrics.cpp** → Core orchestrator containing the Impl class that:
   - Manages data buffers (price, return, sigma, metrics)
   - Calls SIMD operations for CPU preprocessing
   - Would call CUDA kernels for GPU processing (currently #ifdef'd out)
   - Computes final surprise metrics

3. **simd_ops.cpp** → CPU vectorization for:
   - Log returns calculation
   - Realized variance (RV)
   - Bipower variation (BV)

4. **cuda_kernels.cu** → GPU implementations of:
   - GARCH volatility
   - Jump detection algorithms
   - Hawkes process intensity
   - Currently compiled but not called from surprise_metrics.cpp

5. **multi_gpu.cpp** → Manages multiple GPUs:
   - Data distribution across GPUs
   - Stream management
   - Result gathering
   - Currently separate from main flow

6. **polygon_parser.cpp** → Data ingestion:
   - Parse CSV trade/quote data
   - Handle nanosecond timestamps
   - Currently not used by main.cpp

**Note:** There is no "simd.cpp" file - only simd_ops.cpp.

The current issue is that these components aren't fully integrated - CUDA kernels are compiled but not called, multi_gpu exists but isn't used, and polygon_parser isn't connected to the main flow.
