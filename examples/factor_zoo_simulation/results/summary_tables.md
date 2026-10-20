### D1_sparse (100 replications)

| Estimator and tuning | OOS rel. MSE | OOS 1-corr² | # selected | mkt only | family TPR | family FPR | factor TPR | factor FDR | median λ₁ |
|---|---|---|---|---|---|---|---|---|---|
| GrOWL, paper CV (min) | 0.711 ± 0.011 | 0.296 | 1.0 | 100% | 0.14 | 0.00 | 0.14 | 0.00 | 10.000 |
| GrOWL, paper CV (min, converged) | 0.711 ± 0.011 | 0.296 | 1.0 | 100% | 0.14 | 0.00 | 0.14 | 0.00 | 10.000 |
| GrOWL, CV with OLS projection | 0.319 ± 0.007 | 0.203 | 21.5 | 0% | 0.74 | 0.19 | 0.73 | 0.59 | 3.162 |
| GrOWL, CV with OLS projection, 1-SE | 0.318 ± 0.007 | 0.203 | 8.4 | 0% | 0.72 | 0.14 | 0.72 | 0.38 | 3.162 |
| GrOWL, best grid cell (infeasible) | 0.316 ± 0.007 | 0.203 | 19.2 | 0% | 0.73 | 0.15 | 0.72 | 0.59 | 3.162 |
| MT-LASSO, paper CV | 0.987 ± 0.003 | 0.647 | 0.5 | 50% | 0.07 | 0.00 | 0.07 | 0.00 | 31.623 |
| MT-LASSO, CV with OLS projection | 0.319 ± 0.007 | 0.203 | 9.9 | 0% | 0.72 | 0.15 | 0.72 | 0.48 | 3.162 |
| MT-ENet, paper CV | 0.886 ± 0.002 | 0.483 | 17.0 | 0% | 0.29 | 0.00 | 0.29 | 0.88 |  |
| MT-ENet, CV with OLS projection | 0.354 ± 0.008 | 0.225 | 90.0 | 0% | 0.98 | 0.94 | 0.85 | 0.93 |  |
| OLS on mkt only | 0.466 ± 0.008 | 0.296 | 1.0 | 100% | 0.14 | 0.00 | 0.14 | 0.00 |  |
| OLS on all 154 | 0.561 ± 0.014 | 0.273 | 154.0 | 0% | 1.00 | 1.00 | 1.00 | 0.95 |  |
| OLS on true support | 0.427 ± 0.009 | 0.224 | 7.0 | 0% | 1.00 | 0.00 | 1.00 | 0.00 |  |
| GrOWL, true target, CV (min) | 0.000 ± 0.000 | 0.000 | 28.5 | 0% | 1.00 | 0.42 | 1.00 | 0.75 | 0.010 |
| MT-LASSO, true target, CV (min) | 0.000 ± 0.000 | 0.000 | 28.5 | 0% | 1.00 | 0.42 | 1.00 | 0.75 | 0.010 |
| MT-ENet, true target, CV (min) | 0.000 ± 0.000 | 0.000 | 141.8 | 0% | 1.00 | 1.00 | 1.00 | 0.95 |  |

### D2_proxy (100 replications)

| Estimator and tuning | OOS rel. MSE | OOS 1-corr² | # selected | mkt only | family TPR | family FPR | factor TPR | factor FDR | median λ₁ |
|---|---|---|---|---|---|---|---|---|---|
| GrOWL, paper CV (min) | 0.711 ± 0.011 | 0.296 | 1.0 | 100% | 0.14 | 0.00 | 0.01 | 0.00 | 10.000 |
| GrOWL, paper CV (min, converged) | 0.711 ± 0.011 | 0.296 | 1.0 | 100% | 0.14 | 0.00 | 0.01 | 0.00 | 10.000 |
| GrOWL, CV with OLS projection | 0.322 ± 0.007 | 0.207 | 57.5 | 0% | 0.77 | 0.34 | 0.53 | 0.18 | 3.162 |
| GrOWL, CV with OLS projection, 1-SE | 0.322 ± 0.007 | 0.208 | 28.5 | 0% | 0.72 | 0.15 | 0.29 | 0.14 | 3.162 |
| GrOWL, best grid cell (infeasible) | 0.319 ± 0.007 | 0.205 | 47.9 | 0% | 0.72 | 0.26 | 0.48 | 0.14 | 3.162 |
| MT-LASSO, paper CV | 0.987 ± 0.003 | 0.647 | 0.5 | 50% | 0.07 | 0.00 | 0.01 | 0.00 | 31.623 |
| MT-LASSO, CV with OLS projection | 0.322 ± 0.007 | 0.208 | 28.5 | 0% | 0.72 | 0.15 | 0.29 | 0.14 | 3.162 |
| MT-ENet, paper CV | 0.887 ± 0.002 | 0.482 | 17.0 | 0% | 0.29 | 0.00 | 0.20 | 0.00 |  |
| MT-ENet, CV with OLS projection | 0.357 ± 0.008 | 0.228 | 91.2 | 0% | 0.98 | 0.93 | 0.70 | 0.34 |  |
| OLS on mkt only | 0.466 ± 0.008 | 0.296 | 1.0 | 100% | 0.14 | 0.00 | 0.01 | 0.00 |  |
| OLS on all 154 | 0.566 ± 0.014 | 0.283 | 154.0 | 0% | 1.00 | 1.00 | 1.00 | 0.45 |  |
| OLS on true support | 0.488 ± 0.011 | 0.257 | 85.0 | 0% | 1.00 | 0.00 | 1.00 | 0.00 |  |
| GrOWL, true target, CV (min) | 0.017 ± 0.000 | 0.017 | 145.9 | 0% | 1.00 | 1.00 | 1.00 | 0.42 | 0.100 |
| MT-LASSO, true target, CV (min) | 0.017 ± 0.000 | 0.016 | 140.4 | 0% | 1.00 | 0.99 | 0.99 | 0.40 | 0.100 |
| MT-ENet, true target, CV (min) | 0.017 ± 0.000 | 0.017 | 146.7 | 0% | 1.00 | 1.00 | 1.00 | 0.42 |  |

### D2_proxy_strong (50 replications)

| Estimator and tuning | OOS rel. MSE | OOS 1-corr² | # selected | mkt only | family TPR | family FPR | factor TPR | factor FDR | median λ₁ |
|---|---|---|---|---|---|---|---|---|---|
| GrOWL, paper CV (min) | 0.763 ± 0.013 | 0.430 | 30.9 | 44% | 0.41 | 0.08 | 0.30 | 0.05 | 3.162 |
| GrOWL, paper CV (min, converged) | 0.763 ± 0.013 | 0.430 | 30.9 | 44% | 0.41 | 0.08 | 0.30 | 0.05 | 3.162 |
| GrOWL, CV with OLS projection | 0.347 ± 0.007 | 0.233 | 135.0 | 0% | 1.00 | 0.97 | 0.96 | 0.39 | 1.000 |
| GrOWL, CV with OLS projection, 1-SE | 0.348 ± 0.007 | 0.238 | 37.5 | 0% | 0.98 | 0.17 | 0.39 | 0.12 | 3.162 |
| GrOWL, best grid cell (infeasible) | 0.345 ± 0.007 | 0.234 | 120.5 | 0% | 1.00 | 0.83 | 0.89 | 0.34 | 1.000 |
| MT-LASSO, paper CV | 1.000 ± 0.000 | 1.000 | 0.0 | 0% | 0.00 | 0.00 | 0.00 | 0.00 | 31.623 |
| MT-LASSO, CV with OLS projection | 0.352 ± 0.008 | 0.236 | 50.8 | 0% | 0.99 | 0.35 | 0.48 | 0.16 | 3.162 |
| MT-ENet, paper CV | 0.909 ± 0.002 | 0.564 | 57.0 | 0% | 0.68 | 0.11 | 0.60 | 0.08 |  |
| MT-ENet, CV with OLS projection | 0.360 ± 0.007 | 0.235 | 118.2 | 0% | 1.00 | 1.00 | 0.90 | 0.36 |  |
| OLS on mkt only | 0.608 ± 0.008 | 0.450 | 1.0 | 100% | 0.14 | 0.00 | 0.01 | 0.00 |  |
| OLS on all 154 | 0.586 ± 0.013 | 0.292 | 154.0 | 0% | 1.00 | 1.00 | 1.00 | 0.45 |  |
| OLS on true support | 0.494 ± 0.010 | 0.263 | 85.0 | 0% | 1.00 | 0.00 | 1.00 | 0.00 |  |
| GrOWL, true target, CV (min) | 0.027 ± 0.000 | 0.026 | 138.2 | 0% | 1.00 | 1.00 | 0.98 | 0.39 | 0.100 |
| MT-LASSO, true target, CV (min) | 0.027 ± 0.000 | 0.027 | 121.9 | 0% | 1.00 | 1.00 | 0.97 | 0.32 | 0.316 |
| MT-ENet, true target, CV (min) | 0.027 ± 0.001 | 0.026 | 145.8 | 0% | 1.00 | 1.00 | 1.00 | 0.42 |  |

### D2_proxy_weak (50 replications)

| Estimator and tuning | OOS rel. MSE | OOS 1-corr² | # selected | mkt only | family TPR | family FPR | factor TPR | factor FDR | median λ₁ |
|---|---|---|---|---|---|---|---|---|---|
| GrOWL, paper CV (min) | 0.653 ± 0.006 | 0.107 | 1.0 | 100% | 0.14 | 0.00 | 0.01 | 0.00 | 10.000 |
| GrOWL, paper CV (min, converged) | 0.653 ± 0.006 | 0.107 | 1.0 | 100% | 0.14 | 0.00 | 0.01 | 0.00 | 10.000 |
| GrOWL, CV with OLS projection | 0.339 ± 0.008 | 0.147 | 33.4 | 0% | 0.65 | 0.17 | 0.34 | 0.12 | 3.162 |
| GrOWL, CV with OLS projection, 1-SE | 0.345 ± 0.009 | 0.150 | 15.5 | 2% | 0.62 | 0.13 | 0.17 | 0.08 | 3.162 |
| GrOWL, best grid cell (infeasible) | 0.334 ± 0.008 | 0.144 | 45.3 | 0% | 0.74 | 0.15 | 0.46 | 0.12 | 3.162 |
| MT-LASSO, paper CV | 0.886 ± 0.005 | 0.107 | 1.0 | 100% | 0.14 | 0.00 | 0.01 | 0.00 | 31.623 |
| MT-LASSO, CV with OLS projection | 0.346 ± 0.008 | 0.153 | 16.3 | 0% | 0.65 | 0.15 | 0.17 | 0.10 | 3.162 |
| MT-ENet, paper CV | 0.882 ± 0.002 | 0.414 | 16.8 | 0% | 0.29 | 0.00 | 0.20 | 0.00 |  |
| MT-ENet, CV with OLS projection | 0.383 ± 0.009 | 0.198 | 78.7 | 0% | 0.98 | 0.98 | 0.55 | 0.40 |  |
| OLS on mkt only | 0.374 ± 0.010 | 0.107 | 1.0 | 100% | 0.14 | 0.00 | 0.01 | 0.00 |  |
| OLS on all 154 | 0.662 ± 0.018 | 0.304 | 154.0 | 0% | 1.00 | 1.00 | 1.00 | 0.45 |  |
| OLS on true support | 0.534 ± 0.013 | 0.259 | 85.0 | 0% | 1.00 | 0.00 | 1.00 | 0.00 |  |
| GrOWL, true target, CV (min) | 0.006 ± 0.000 | 0.006 | 128.8 | 0% | 1.00 | 1.00 | 0.98 | 0.35 | 0.100 |
| MT-LASSO, true target, CV (min) | 0.006 ± 0.000 | 0.006 | 128.5 | 0% | 1.00 | 1.00 | 0.98 | 0.35 | 0.100 |
| MT-ENet, true target, CV (min) | 0.006 ± 0.000 | 0.006 | 135.8 | 0% | 1.00 | 1.00 | 0.99 | 0.38 |  |

### D2_proxy_balanced (50 replications)

| Estimator and tuning | OOS rel. MSE | OOS 1-corr² | # selected | mkt only | family TPR | family FPR | factor TPR | factor FDR | median λ₁ |
|---|---|---|---|---|---|---|---|---|---|
| GrOWL, paper CV (min) | 0.073 ± 0.003 | 0.067 | 144.6 | 0% | 1.00 | 1.00 | 0.97 | 0.43 | 0.100 |
| GrOWL, paper CV (min, converged) | 0.073 ± 0.003 | 0.067 | 144.6 | 0% | 1.00 | 1.00 | 0.97 | 0.43 | 0.100 |
| GrOWL, CV with OLS projection | 0.073 ± 0.003 | 0.067 | 144.6 | 0% | 1.00 | 1.00 | 0.97 | 0.43 | 0.100 |
| GrOWL, CV with OLS projection, 1-SE | 0.074 ± 0.003 | 0.068 | 99.7 | 0% | 0.99 | 0.97 | 0.81 | 0.31 | 0.316 |
| GrOWL, best grid cell (infeasible) | 0.073 ± 0.003 | 0.067 | 142.1 | 0% | 1.00 | 1.00 | 0.96 | 0.42 | 0.100 |
| MT-LASSO, paper CV | 0.073 ± 0.003 | 0.067 | 110.2 | 0% | 1.00 | 0.99 | 0.85 | 0.34 | 0.316 |
| MT-LASSO, CV with OLS projection | 0.073 ± 0.003 | 0.067 | 110.2 | 0% | 1.00 | 0.99 | 0.85 | 0.34 | 0.316 |
| MT-ENet, paper CV | 0.074 ± 0.003 | 0.068 | 145.1 | 0% | 1.00 | 1.00 | 0.96 | 0.44 |  |
| MT-ENet, CV with OLS projection | 0.074 ± 0.003 | 0.068 | 145.1 | 0% | 1.00 | 1.00 | 0.96 | 0.44 |  |
| OLS on mkt only | 0.303 ± 0.003 | 0.296 | 1.0 | 100% | 0.14 | 0.00 | 0.01 | 0.00 |  |
| OLS on all 154 | 0.080 ± 0.003 | 0.072 | 154.0 | 0% | 1.00 | 1.00 | 1.00 | 0.45 |  |
| OLS on true support | 0.077 ± 0.003 | 0.070 | 85.0 | 0% | 1.00 | 0.00 | 1.00 | 0.00 |  |
| GrOWL, true target, CV (min) | 0.017 ± 0.000 | 0.017 | 146.1 | 0% | 1.00 | 1.00 | 1.00 | 0.42 | 0.100 |
| MT-LASSO, true target, CV (min) | 0.017 ± 0.000 | 0.017 | 140.9 | 0% | 1.00 | 0.99 | 0.99 | 0.40 | 0.100 |
| MT-ENet, true target, CV (min) | 0.017 ± 0.000 | 0.017 | 146.9 | 0% | 1.00 | 1.00 | 1.00 | 0.42 |  |

### D0_market_only (50 replications)

| Estimator and tuning | OOS rel. MSE | OOS 1-corr² | # selected | mkt only | family TPR | family FPR | factor TPR | factor FDR | median λ₁ |
|---|---|---|---|---|---|---|---|---|---|
| GrOWL, paper CV (min) | 0.609 ± 0.006 | -0.000 | 1.0 | 100% | 1.00 | 0.00 | 1.00 | 0.00 | 10.000 |
| GrOWL, paper CV (min, converged) | 0.609 ± 0.006 | -0.000 | 1.0 | 100% | 1.00 | 0.00 | 1.00 | 0.00 | 10.000 |
| GrOWL, CV with OLS projection | 0.305 ± 0.009 | 0.001 | 7.4 | 80% | 1.00 | 0.06 | 1.00 | 0.14 | 1.000 |
| GrOWL, CV with OLS projection, 1-SE | 0.305 ± 0.009 | 0.000 | 1.0 | 100% | 1.00 | 0.00 | 1.00 | 0.00 | 1.000 |
| GrOWL, best grid cell (infeasible) | 0.289 ± 0.007 | 0.000 | 3.4 | 98% | 1.00 | 0.02 | 1.00 | 0.02 | 3.162 |
| MT-LASSO, paper CV | 0.860 ± 0.005 | -0.000 | 1.0 | 100% | 1.00 | 0.00 | 1.00 | 0.00 | 31.623 |
| MT-LASSO, CV with OLS projection | 0.301 ± 0.009 | -0.000 | 1.0 | 100% | 1.00 | 0.00 | 1.00 | 0.00 | 3.162 |
| MT-ENet, paper CV | 0.891 ± 0.005 | 0.315 | 17.0 | 0% | 1.00 | 0.09 | 1.00 | 0.94 |  |
| MT-ENet, CV with OLS projection | 0.310 ± 0.008 | 0.067 | 53.5 | 0% | 1.00 | 0.96 | 1.00 | 0.98 |  |
| OLS on mkt only | 0.331 ± 0.010 | 0.000 | 1.0 | 100% | 1.00 | 0.00 | 1.00 | 0.00 |  |
| OLS on all 154 | 0.667 ± 0.018 | 0.266 | 154.0 | 0% | 1.00 | 1.00 | 1.00 | 0.99 |  |
| OLS on true support | 0.331 ± 0.010 | 0.000 | 1.0 | 100% | 1.00 | 0.00 | 1.00 | 0.00 |  |
| GrOWL, true target, CV (min) | 0.000 ± 0.000 | 0.000 | 1.5 | 72% | 1.00 | 0.02 | 1.00 | 0.15 | 0.010 |
| MT-LASSO, true target, CV (min) | 0.000 ± 0.000 | 0.000 | 1.5 | 72% | 1.00 | 0.02 | 1.00 | 0.15 | 0.010 |
| MT-ENet, true target, CV (min) | 0.000 ± 0.000 | 0.000 | 12.4 | 0% | 1.00 | 0.35 | 1.00 | 0.92 |  |

### D2_break (50 replications)

| Estimator and tuning | OOS rel. MSE | OOS 1-corr² | # selected | mkt only | family TPR | family FPR | factor TPR | factor FDR | median λ₁ |
|---|---|---|---|---|---|---|---|---|---|
| GrOWL, paper CV (min) | 0.719 ± 0.015 | 0.321 | 1.0 | 100% | 0.14 | 0.00 | 0.01 | 0.00 | 10.000 |
| GrOWL, paper CV (min, converged) | 0.719 ± 0.015 | 0.321 | 1.0 | 100% | 0.14 | 0.00 | 0.01 | 0.00 | 10.000 |
| GrOWL, CV with OLS projection | 0.515 ± 0.009 | 0.370 | 51.0 | 0% | 0.75 | 0.25 | 0.50 | 0.15 | 3.162 |
| GrOWL, CV with OLS projection, 1-SE | 0.525 ± 0.012 | 0.363 | 20.0 | 0% | 0.63 | 0.18 | 0.20 | 0.12 | 3.162 |
| GrOWL, best grid cell (infeasible) | 0.493 ± 0.006 | 0.345 | 9.0 | 0% | 0.47 | 0.06 | 0.10 | 0.05 | 10.000 |
| MT-LASSO, paper CV | 0.997 ± 0.003 | 0.905 | 0.1 | 14% | 0.02 | 0.00 | 0.00 | 0.00 | 31.623 |
| MT-LASSO, CV with OLS projection | 0.551 ± 0.012 | 0.379 | 31.4 | 0% | 0.78 | 0.27 | 0.30 | 0.18 | 3.162 |
| MT-ENet, paper CV | 0.913 ± 0.003 | 0.543 | 17.5 | 0% | 0.29 | 0.00 | 0.21 | 0.00 |  |
| MT-ENet, CV with OLS projection | 0.628 ± 0.017 | 0.413 | 75.2 | 0% | 0.95 | 0.66 | 0.61 | 0.28 |  |
| OLS on mkt only | 0.497 ± 0.009 | 0.321 | 1.0 | 100% | 0.14 | 0.00 | 0.01 | 0.00 |  |
| OLS on all 154 | 1.012 ± 0.027 | 0.472 | 154.0 | 0% | 1.00 | 1.00 | 1.00 | 0.45 |  |
| OLS on true support | 0.889 ± 0.022 | 0.450 | 85.0 | 0% | 1.00 | 0.00 | 1.00 | 0.00 |  |
| GrOWL, true target, CV (min) | 0.173 ± 0.009 | 0.166 | 86.6 | 0% | 0.95 | 0.73 | 0.73 | 0.26 | 1.000 |
| MT-LASSO, true target, CV (min) | 0.170 ± 0.008 | 0.164 | 69.6 | 0% | 0.94 | 0.69 | 0.59 | 0.25 | 1.000 |
| MT-ENet, true target, CV (min) | 0.144 ± 0.004 | 0.138 | 134.0 | 0% | 1.00 | 0.99 | 0.92 | 0.41 |  |

### D2_break_themes (50 replications)

| Estimator and tuning | OOS rel. MSE | OOS 1-corr² | # selected | mkt only | family TPR | family FPR | factor TPR | factor FDR | median λ₁ |
|---|---|---|---|---|---|---|---|---|---|
| GrOWL, paper CV (min) | 0.763 ± 0.007 | 0.430 | 1.0 | 100% | 0.10 | 0.00 | 0.01 | 0.00 | 3.162 |
| GrOWL, paper CV (min, converged) | 0.763 ± 0.007 | 0.430 | 1.0 | 100% | 0.10 | 0.00 | 0.01 | 0.00 | 3.162 |
| GrOWL, CV with OLS projection | 0.622 ± 0.008 | 0.439 | 50.6 | 0% | 0.64 | 0.19 | 0.43 | 0.05 | 3.162 |
| GrOWL, CV with OLS projection, 1-SE | 0.609 ± 0.010 | 0.435 | 10.8 | 0% | 0.39 | 0.04 | 0.10 | 0.01 | 10.000 |
| GrOWL, best grid cell (infeasible) | 0.590 ± 0.005 | 0.434 | 8.9 | 0% | 0.40 | 0.01 | 0.08 | 0.00 | 10.000 |
| MT-LASSO, paper CV | 1.000 ± 0.000 | 1.000 | 0.0 | 0% | 0.00 | 0.00 | 0.00 | 0.00 | 31.623 |
| MT-LASSO, CV with OLS projection | 0.646 ± 0.015 | 0.440 | 24.6 | 0% | 0.59 | 0.18 | 0.21 | 0.06 | 3.162 |
| MT-ENet, paper CV | 0.953 ± 0.001 | 0.593 | 16.0 | 0% | 0.25 | 0.00 | 0.15 | 0.00 |  |
| MT-ENet, CV with OLS projection | 0.725 ± 0.012 | 0.474 | 67.1 | 0% | 0.89 | 0.42 | 0.54 | 0.11 |  |
| OLS on mkt only | 0.600 ± 0.008 | 0.430 | 1.0 | 100% | 0.10 | 0.00 | 0.01 | 0.00 |  |
| OLS on all 154 | 1.337 ± 0.020 | 0.530 | 154.0 | 0% | 1.00 | 1.00 | 1.00 | 0.29 |  |
| OLS on true support | 1.229 ± 0.016 | 0.513 | 110.0 | 0% | 1.00 | 0.00 | 1.00 | 0.00 |  |
| GrOWL, true target, CV (min) | 0.178 ± 0.009 | 0.173 | 98.8 | 0% | 0.97 | 0.84 | 0.69 | 0.21 | 1.000 |
| MT-LASSO, true target, CV (min) | 0.181 ± 0.008 | 0.175 | 78.1 | 0% | 0.95 | 0.78 | 0.56 | 0.20 | 1.000 |
| MT-ENet, true target, CV (min) | 0.161 ± 0.004 | 0.157 | 137.2 | 0% | 1.00 | 1.00 | 0.90 | 0.28 |  |

### First stage (means over replications)

| Scenario | Bai-Ng k (full) | Bai-Ng k (folds) | EM iter. (full) | missing | firm R² (7 f.) | firm R² (mkt) | sv² share PC1 of Y | R²(Y on mkt) | R²(Y on all X) | min cancorr(F̂, X) top 7 | corr²(ĉ, c) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| data (CRSP / GFD) | 7 | 3, 4, 5, 7, 7 | 390 | 0.68 | 0.318 | 0.230 | 0.406 | 0.357 | 0.857 | 0.800 | n/a |
| D1_sparse | 6.41 | 3.59 | 470 | 0.71 | 0.327 | 0.221 | 0.540 | 0.512 | 0.850 | 0.500 | 0.661 |
| D2_proxy | 6.41 | 3.59 | 470 | 0.71 | 0.327 | 0.221 | 0.540 | 0.512 | 0.838 | 0.499 | 0.661 |
| D2_proxy_strong | 8.72 | 5.17 | 469 | 0.71 | 0.394 | 0.213 | 0.430 | 0.387 | 0.825 | 0.590 | 0.658 |
| D2_proxy_weak | 5.28 | 3.03 | 496 | 0.71 | 0.275 | 0.228 | 0.609 | 0.595 | 0.758 | 0.482 | 0.599 |
| D2_proxy_balanced | 4.80 | 2.18 | 1 | 0.00 | 0.314 | 0.221 | 0.739 | 0.706 | 0.985 | 0.934 | 0.940 |
| D0_market_only | 5.16 | 2.98 | 498 | 0.71 | 0.222 | 0.222 | 0.612 | 0.606 | 0.706 | 0.452 | 0.606 |
| D2_break | 6.06 | 3.56 | 469 | 0.71 | 0.336 | 0.219 | 0.503 | 0.476 | 0.806 | 0.565 | 0.535 |
| D2_break_themes | 7.84 | 4.52 | 488 | 0.71 | 0.356 | 0.191 | 0.376 | 0.347 | 0.767 | 0.580 | 0.494 |

### CV landscape: does the CV loss rank cells like the true loss?

| Scenario | CV variant | validation-target variance | Spearman(CV MSE, OOS MSE) | regret of CV choice | CV rank of best cell | cells within 1 SE | cells within 1 paired SE |
|---|---|---|---|---|---|---|---|
| D1_sparse | Paper CV (projection z·Λ/N) | 0.027 | -0.73 | 0.395 | 72 | 2 | 2 |
| D1_sparse | CV with OLS projection | 0.739 | 0.98 | 0.002 | 6 | 16 | 5 |
| D1_sparse | CV on true common component | 1.000 | 1.00 | 0.000 | 1 | 1 | 1 |
| D2_proxy | Paper CV (projection z·Λ/N) | 0.027 | -0.73 | 0.393 | 72 | 2 | 2 |
| D2_proxy | CV with OLS projection | 0.739 | 0.98 | 0.003 | 5 | 17 | 5 |
| D2_proxy | CV on true common component | 1.000 | 1.00 | 0.000 | 2 | 5 | 3 |
| D2_proxy_strong | Paper CV (projection z·Λ/N) | 0.027 | -0.75 | 0.418 | 79 | 2 | 4 |
| D2_proxy_strong | CV with OLS projection | 0.725 | 0.99 | 0.003 | 2 | 18 | 3 |
| D2_proxy_strong | CV on true common component | 0.997 | 1.00 | 0.001 | 6 | 5 | 3 |
| D2_proxy_weak | Paper CV (projection z·Λ/N) | 0.026 | -0.61 | 0.319 | 71 | 1 | 2 |
| D2_proxy_weak | CV with OLS projection | 0.744 | 0.95 | 0.005 | 7 | 21 | 6 |
| D2_proxy_weak | CV on true common component | 0.998 | 1.00 | 0.000 | 1 | 2 | 1 |
| D2_proxy_balanced | Paper CV (projection z·Λ/N) | 0.960 | 1.00 | 0.000 | 1 | 10 | 2 |
| D2_proxy_balanced | CV with OLS projection | 0.960 | 1.00 | 0.000 | 1 | 10 | 2 |
| D2_proxy_balanced | CV on true common component | 0.998 | 1.00 | 0.000 | 2 | 5 | 3 |
| D0_market_only | Paper CV (projection z·Λ/N) | 0.026 | -0.59 | 0.320 | 70 | 1 | 2 |
| D0_market_only | CV with OLS projection | 0.734 | 0.95 | 0.016 | 25 | 28 | 6 |
| D0_market_only | CV on true common component | 0.983 | 1.00 | 0.000 | 1 | 1 | 1 |
| D2_break | Paper CV (projection z·Λ/N) | 0.024 | -0.41 | 0.226 | 69 | 2 | 2 |
| D2_break | CV with OLS projection | 0.655 | 0.88 | 0.022 | 10 | 10 | 2 |
| D2_break | CV on true common component | 1.096 | 0.90 | 0.041 | 25 | 59 | 22 |
| D2_break_themes | Paper CV (projection z·Λ/N) | 0.020 | 0.07 | 0.173 | 65 | 1 | 1 |
| D2_break_themes | CV with OLS projection | 0.552 | 0.73 | 0.032 | 8 | 16 | 7 |
| D2_break_themes | CV on true common component | 1.119 | 0.92 | 0.023 | 10 | 67 | 28 |

### Split-half stability of the selected mapping

GrOWL at the CV-selected cell (CV with OLS projection; true target: CV on the true common component) is fitted on each half of the sample and scored on both halves (R² of the half-centered target). In the regime-change scenarios the halves are the two regimes. Means over replications.

| Scenario | target | within-half R² | across-half R² | mkt-only within | mkt-only across | support Jaccard | row cosine | # selected per half |
|---|---|---|---|---|---|---|---|---|
| D2_proxy | PCA target | 0.768 | 0.603 | 0.546 | 0.411 | 0.66 | 0.85 | 59.8 |
| D2_proxy | true target | 0.990 | 0.981 | 0.704 | 0.702 | 0.93 | 0.58 | 147.8 |
| D2_break | PCA target | 0.807 | 0.190 | 0.539 | 0.292 | 0.44 | 0.07 | 42.3 |
| D2_break | true target | 0.951 | 0.483 | 0.707 | 0.588 | 0.53 | 0.11 | 76.4 |
| D2_break_themes | PCA target | 0.746 | 0.058 | 0.395 | 0.209 | 0.29 | 0.14 | 36.8 |
| D2_break_themes | true target | 0.971 | 0.259 | 0.604 | 0.495 | 0.46 | 0.11 | 83.2 |

### Grouping inside the six style families

Members per selected style family, mean cosine between the selected rows of one family, and number of factors in exact row-norm ties.

| Scenario | Tuning | Method | # selected | members / selected family | within-family cosine | tied factors | tie pairs in same family | same-family share of all selected pairs (chance) |
|---|---|---|---|---|---|---|---|---|
| D1_sparse | CV, OLS projection | growl | 21.5 | 3.29 | 0.99 | 11.9 | 21% | 16% |
| D1_sparse | CV, OLS projection | mtl | 9.9 | 1.28 | 0.99 | 0.0 |  | 14% |
| D1_sparse | CV, OLS projection | enet | 90.0 | 9.87 | 0.87 | 0.0 |  | 11% |
| D1_sparse | best grid cell | growl | 19.2 | 3.38 | 0.99 | 10.9 | 25% | 16% |
| D1_sparse | best grid cell | mtl | 9.9 | 1.28 | 0.99 | 0.0 |  | 14% |
| D1_sparse | best grid cell | enet | 58.5 | 9.39 | 0.97 | 0.0 |  | 18% |
| D1_sparse | true target, CV | growl | 28.5 | 2.71 | 0.99 | 0.0 |  | 11% |
| D1_sparse | true target, CV | mtl | 28.5 | 2.71 | 0.99 | 0.0 |  | 11% |
| D1_sparse | true target, CV | enet | 141.8 | 13.93 | 0.85 | 0.0 |  | 9% |
| D2_proxy | CV, OLS projection | growl | 57.5 | 9.65 | 0.98 | 40.5 | 23% | 19% |
| D2_proxy | CV, OLS projection | mtl | 28.5 | 5.82 | 0.99 | 0.0 |  | 19% |
| D2_proxy | CV, OLS projection | enet | 91.2 | 10.02 | 0.87 | 0.0 |  | 11% |
| D2_proxy | best grid cell | growl | 47.9 | 9.93 | 0.99 | 35.5 | 28% | 21% |
| D2_proxy | best grid cell | mtl | 28.5 | 5.82 | 0.99 | 0.0 |  | 19% |
| D2_proxy | best grid cell | enet | 59.4 | 9.63 | 0.97 | 0.0 |  | 18% |
| D2_proxy | true target, CV | growl | 145.9 | 13.97 | 0.90 | 31.1 | 19% | 9% |
| D2_proxy | true target, CV | mtl | 140.4 | 13.90 | 0.88 | 0.0 |  | 9% |
| D2_proxy | true target, CV | enet | 146.7 | 13.98 | 0.87 | 0.0 |  | 9% |
| D2_proxy_strong | CV, OLS projection | growl | 135.0 | 13.41 | 0.94 | 112.9 | 21% | 9% |
| D2_proxy_strong | CV, OLS projection | mtl | 50.8 | 6.70 | 0.96 | 0.0 |  | 14% |
| D2_proxy_strong | CV, OLS projection | enet | 118.2 | 12.52 | 0.87 | 0.0 |  | 9% |
| D2_proxy_strong | best grid cell | growl | 120.5 | 12.40 | 0.96 | 102.2 | 21% | 11% |
| D2_proxy_strong | best grid cell | mtl | 37.6 | 5.45 | 0.99 | 0.0 |  | 15% |
| D2_proxy_strong | best grid cell | enet | 107.7 | 11.88 | 0.89 | 0.0 |  | 10% |
| D2_proxy_strong | true target, CV | growl | 138.2 | 13.79 | 0.92 | 36.7 | 17% | 9% |
| D2_proxy_strong | true target, CV | mtl | 121.9 | 13.53 | 0.93 | 0.0 |  | 10% |
| D2_proxy_strong | true target, CV | enet | 145.8 | 13.95 | 0.87 | 0.0 |  | 9% |
| D2_proxy_weak | CV, OLS projection | growl | 33.4 | 8.06 | 0.99 | 26.0 | 31% | 27% |
| D2_proxy_weak | CV, OLS projection | mtl | 16.3 | 3.89 | 0.99 | 0.0 |  | 23% |
| D2_proxy_weak | CV, OLS projection | enet | 78.7 | 7.82 | 0.70 | 0.0 |  | 10% |
| D2_proxy_weak | best grid cell | growl | 45.3 | 8.61 | 0.99 | 40.0 | 18% | 24% |
| D2_proxy_weak | best grid cell | mtl | 13.5 | 3.37 | 0.99 | 0.0 |  | 20% |
| D2_proxy_weak | best grid cell | enet | 35.8 | 7.51 | 0.98 | 0.0 |  | 28% |
| D2_proxy_weak | true target, CV | growl | 128.8 | 13.76 | 0.92 | 2.1 | 20% | 9% |
| D2_proxy_weak | true target, CV | mtl | 128.5 | 13.76 | 0.92 | 0.0 |  | 10% |
| D2_proxy_weak | true target, CV | enet | 135.8 | 13.87 | 0.91 | 0.0 |  | 9% |
| D2_proxy_balanced | CV, OLS projection | growl | 144.6 | 13.59 | 0.87 | 62.0 | 16% | 9% |
| D2_proxy_balanced | CV, OLS projection | mtl | 110.2 | 11.84 | 0.90 | 0.0 |  | 10% |
| D2_proxy_balanced | CV, OLS projection | enet | 145.1 | 13.47 | 0.78 | 0.0 |  | 9% |
| D2_proxy_balanced | best grid cell | growl | 142.1 | 13.50 | 0.88 | 61.0 | 15% | 9% |
| D2_proxy_balanced | best grid cell | mtl | 103.4 | 11.54 | 0.92 | 0.0 |  | 10% |
| D2_proxy_balanced | best grid cell | enet | 140.1 | 13.26 | 0.81 | 0.0 |  | 9% |
| D2_proxy_balanced | true target, CV | growl | 146.1 | 13.97 | 0.90 | 33.8 | 19% | 9% |
| D2_proxy_balanced | true target, CV | mtl | 140.9 | 13.90 | 0.88 | 0.0 |  | 9% |
| D2_proxy_balanced | true target, CV | enet | 146.9 | 13.98 | 0.87 | 0.0 |  | 9% |
| D2_break | CV, OLS projection | growl | 51.0 | 9.93 | 0.95 | 40.1 | 28% | 23% |
| D2_break | CV, OLS projection | mtl | 31.4 | 5.45 | 0.85 | 0.0 |  | 18% |
| D2_break | CV, OLS projection | enet | 75.2 | 9.00 | 0.70 | 0.0 |  | 15% |
| D2_break | best grid cell | growl | 9.0 | 2.82 | 1.00 | 2.2 | 30% | 23% |
| D2_break | best grid cell | mtl | 6.3 | 2.33 | 1.00 | 0.0 |  | 23% |
| D2_break | best grid cell | enet | 56.9 | 8.51 | 0.84 | 0.0 |  | 19% |
| D2_break | true target, CV | growl | 86.6 | 10.78 | 0.80 | 35.5 | 24% | 13% |
| D2_break | true target, CV | mtl | 69.6 | 8.66 | 0.78 | 0.0 |  | 13% |
| D2_break | true target, CV | enet | 134.0 | 12.93 | 0.57 | 0.0 |  | 9% |
| D2_break_themes | CV, OLS projection | growl | 50.6 | 8.12 | 0.91 | 38.1 | 24% | 20% |
| D2_break_themes | CV, OLS projection | mtl | 24.6 | 4.04 | 0.82 | 0.0 |  | 20% |
| D2_break_themes | CV, OLS projection | enet | 67.1 | 7.41 | 0.76 | 0.0 |  | 14% |
| D2_break_themes | best grid cell | growl | 8.9 | 2.55 | 0.99 | 1.0 | 32% | 29% |
| D2_break_themes | best grid cell | mtl | 7.9 | 2.41 | 0.99 | 0.0 |  | 29% |
| D2_break_themes | best grid cell | enet | 53.2 | 7.61 | 0.85 | 0.0 |  | 22% |
| D2_break_themes | true target, CV | growl | 98.8 | 8.48 | 0.55 | 26.1 | 17% | 10% |
| D2_break_themes | true target, CV | mtl | 78.1 | 6.90 | 0.57 | 0.0 |  | 10% |
| D2_break_themes | true target, CV | enet | 137.2 | 10.85 | 0.37 | 0.0 |  | 8% |

### Accuracy of the tol = 1e-4 solutions at the selected cells

Each selected GrOWL cell is solved again with tol = 1e-9.

| Scenario | CV variant | rule | Jaccard(support, tight support) | share of cells that hit max_iter in a fold |
|---|---|---|---|---|
| D1_sparse | pipeline | cv_min | 1.000 | 0% |
| D1_sparse | pipeline | cv_min_clean | 1.000 | 0% |
| D1_sparse | pipeline | oracle | 0.999 | 0% |
| D1_sparse | pipeline_olsproj | cv_min | 1.000 | 0% |
| D1_sparse | pipeline_olsproj | cv_min_clean | 1.000 | 0% |
| D1_sparse | pipeline_olsproj | oracle | 0.999 | 0% |
| D1_sparse | oracle | cv_min | 0.944 | 42% |
| D1_sparse | oracle | cv_min_clean | 0.962 | 0% |
| D1_sparse | oracle | oracle | 0.944 | 42% |
| D2_proxy | pipeline | cv_min | 1.000 | 0% |
| D2_proxy | pipeline | cv_min_clean | 1.000 | 0% |
| D2_proxy | pipeline | oracle | 1.000 | 0% |
| D2_proxy | pipeline_olsproj | cv_min | 1.000 | 0% |
| D2_proxy | pipeline_olsproj | cv_min_clean | 1.000 | 0% |
| D2_proxy | pipeline_olsproj | oracle | 1.000 | 0% |
| D2_proxy | oracle | cv_min | 1.000 | 0% |
| D2_proxy | oracle | cv_min_clean | 1.000 | 0% |
| D2_proxy | oracle | oracle | 1.000 | 0% |
| D2_proxy_strong | pipeline | cv_min | 1.000 | 0% |
| D2_proxy_strong | pipeline | cv_min_clean | 1.000 | 0% |
| D2_proxy_strong | pipeline | oracle | 1.000 | 0% |
| D2_proxy_strong | pipeline_olsproj | cv_min | 1.000 | 0% |
| D2_proxy_strong | pipeline_olsproj | cv_min_clean | 1.000 | 0% |
| D2_proxy_strong | pipeline_olsproj | oracle | 1.000 | 0% |
| D2_proxy_strong | oracle | cv_min | 1.000 | 0% |
| D2_proxy_strong | oracle | cv_min_clean | 1.000 | 0% |
| D2_proxy_strong | oracle | oracle | 1.000 | 0% |
| D2_proxy_weak | pipeline | cv_min | 1.000 | 0% |
| D2_proxy_weak | pipeline | cv_min_clean | 1.000 | 0% |
| D2_proxy_weak | pipeline | oracle | 1.000 | 0% |
| D2_proxy_weak | pipeline_olsproj | cv_min | 1.000 | 0% |
| D2_proxy_weak | pipeline_olsproj | cv_min_clean | 1.000 | 0% |
| D2_proxy_weak | pipeline_olsproj | oracle | 1.000 | 0% |
| D2_proxy_weak | oracle | cv_min | 1.000 | 0% |
| D2_proxy_weak | oracle | cv_min_clean | 1.000 | 0% |
| D2_proxy_weak | oracle | oracle | 1.000 | 0% |
| D2_proxy_balanced | pipeline | cv_min | 1.000 | 0% |
| D2_proxy_balanced | pipeline | cv_min_clean | 1.000 | 0% |
| D2_proxy_balanced | pipeline | oracle | 1.000 | 0% |
| D2_proxy_balanced | pipeline_olsproj | cv_min | 1.000 | 0% |
| D2_proxy_balanced | pipeline_olsproj | cv_min_clean | 1.000 | 0% |
| D2_proxy_balanced | pipeline_olsproj | oracle | 1.000 | 0% |
| D2_proxy_balanced | oracle | cv_min | 1.000 | 0% |
| D2_proxy_balanced | oracle | cv_min_clean | 1.000 | 0% |
| D2_proxy_balanced | oracle | oracle | 1.000 | 0% |
| D0_market_only | pipeline | cv_min | 1.000 | 0% |
| D0_market_only | pipeline | cv_min_clean | 1.000 | 0% |
| D0_market_only | pipeline | oracle | 1.000 | 0% |
| D0_market_only | pipeline_olsproj | cv_min | 1.000 | 0% |
| D0_market_only | pipeline_olsproj | cv_min_clean | 1.000 | 0% |
| D0_market_only | pipeline_olsproj | oracle | 1.000 | 0% |
| D0_market_only | oracle | cv_min | 0.845 | 0% |
| D0_market_only | oracle | cv_min_clean | 0.845 | 0% |
| D0_market_only | oracle | oracle | 0.872 | 0% |
| D2_break | pipeline | cv_min | 1.000 | 0% |
| D2_break | pipeline | cv_min_clean | 1.000 | 0% |
| D2_break | pipeline | oracle | 1.000 | 0% |
| D2_break | pipeline_olsproj | cv_min | 1.000 | 0% |
| D2_break | pipeline_olsproj | cv_min_clean | 1.000 | 0% |
| D2_break | pipeline_olsproj | oracle | 1.000 | 0% |
| D2_break | oracle | cv_min | 1.000 | 0% |
| D2_break | oracle | cv_min_clean | 1.000 | 0% |
| D2_break | oracle | oracle | 1.000 | 0% |
| D2_break_themes | pipeline | cv_min | 1.000 | 0% |
| D2_break_themes | pipeline | cv_min_clean | 1.000 | 0% |
| D2_break_themes | pipeline | oracle | 1.000 | 0% |
| D2_break_themes | pipeline_olsproj | cv_min | 1.000 | 0% |
| D2_break_themes | pipeline_olsproj | cv_min_clean | 1.000 | 0% |
| D2_break_themes | pipeline_olsproj | oracle | 1.000 | 0% |
| D2_break_themes | oracle | cv_min | 1.000 | 0% |
| D2_break_themes | oracle | cv_min_clean | 1.000 | 0% |
| D2_break_themes | oracle | oracle | 1.000 | 0% |

### Package-solver spot checks

1000 cells re-solved with `growl.fista_solver.growl_fista`: max relative difference 4.5e-04; same iteration count in 99.1% of cells; same support in 100.0% of cells.
