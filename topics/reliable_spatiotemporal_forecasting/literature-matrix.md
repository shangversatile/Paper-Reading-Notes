# Literature Matrix

Do not populate claims without verified primary sources.

| Paper | Problem | Method | Assumptions | Guarantee Type | Evidence | Limitation | Project Connection | Research Context | Verified On |
| ----- | ------- | ------ | ----------- | -------------- | -------- | ---------- | ------------------ | ---------------- | ----------- |
| P-GRAPH-001 | Generalize CNN convolution to irregular graph domains | Spectral graph filtering with Chebyshev polynomial approximation | Data must be meaningfully represented by a sparse graph; locality and stationarity should hold on the graph | K-hop locality and computational efficiency, not statistical reliability guarantee | MNIST and 20NEWS show efficient graph CNN performance and graph-quality sensitivity | Classification experiments only; no forecasting, UQ, calibration, or shift analysis | Foundation for STGCN graph convolution and graph-quality sensitivity | Graph signal processing foundation for spatiotemporal modeling | 2026-06 |

## Planned Extraction Matrix

The entries below are title-level planning records until primary-source
metadata and technical claims are verified. They define what to extract,
not what has already been reproduced.

| Priority | Paper / Component | Verification Status | Project Role | Planned Extraction |
| -------- | ----------------- | ------------------- | ------------ | ------------------ |
| Priority 1 | Quantifying Uncertainty in Deep Spatiotemporal Forecasting | metadata-to-verify | Core UQ and spatiotemporal forecasting bridge | Minimal UQ comparison across MC Dropout, Deep Ensemble, Quantile, and Conformal-style calibration where feasible |
| Priority 1 | DCRNN or STGCN baseline | Existing metadata available for DCRNN / STGCN entries | Forecasting baseline | MAE / RMSE table on traffic or PM2.5-style data |
| Priority 1 | Copula Conformal Prediction for Multi-Step Time-Series Forecasting | Existing metadata available | Multi-step conformal calibration | Coverage, interval width, and multi-step calibration evaluation |
| Priority 1 | Conformal Prediction for Time Series with Change Points / CPTC | Existing metadata available | Dynamic shift and change-point-aware calibration | Synthetic change-point experiment and shift coverage degradation figure |
| Priority 2 | Provably Robust Conformal Prediction with Improved Efficiency | metadata-to-verify | Robust conformal reliability under perturbation | How coverage guarantees behave under perturbation |
| Priority 2 | Evaluating Neuron Explanations: A Unified Framework with Sanity Checks | metadata-to-verify | Trustworthy evaluation and meta-evaluation | Sanity-check logic for reliability metrics themselves |
| Priority 2 | Prediction without Preclusion: Recourse Verification with Reachable Sets | metadata-to-verify | Risk-aware decision-making and actionability | Connection from uncertainty outputs to feasible actions and downstream consequences |
| Priority 2 | U-Cast: A Simple Approach to Calibrated Weather Forecasting | metadata-to-verify | Probabilistic environmental forecasting | CRPS, MC Dropout, and probabilistic forecasting evaluation design |
