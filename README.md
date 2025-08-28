# intuition 

 intuition behind these surprise metrics is to detect **when something unusual is happening** in financial markets that warrants immediate attention. Let me break down each component:

## Standardized Returns (zₜ = rₜ/σ̂ₜ)

This normalizes returns by their predicted volatility. The intuition is simple: a 2% move when volatility is low (say σ = 0.5%) is much more surprising than a 2% move when volatility is high (σ = 2%). By dividing returns by estimated volatility, you get a measure of "how many standard deviations" the move represents - making surprises comparable across different volatility regimes.

## Jump Detection (Lee-Mykland/BNS Statistics)

These statistics identify price jumps that are too large to be explained by normal diffusive volatility. The core insight is that in high-frequency data, if you look at very short intervals, the continuous component of price movement becomes negligible while jumps remain prominent. 

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
