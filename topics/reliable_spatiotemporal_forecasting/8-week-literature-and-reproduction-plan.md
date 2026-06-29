# Eight-Week Literature and Reproduction Plan

This public roadmap records objective literature priorities, reproduction
targets, experiment modules, and technical deliverables for reliable
spatiotemporal forecasting under dynamic distribution shift.

## Core Literature Priority

### Priority 1 - Read and Reproduce

| Order | Paper / Component | Role in Project | Planned Output |
| ----- | ----------------- | --------------- | -------------- |
| 1 | Quantifying Uncertainty in Deep Spatiotemporal Forecasting | Core UQ paper connecting deep spatiotemporal forecasting and uncertainty estimation | Minimal reproduction; compare MC Dropout, Deep Ensemble, Quantile, and Conformal-style calibration where feasible |
| 2 | DCRNN: Diffusion Convolutional Recurrent Neural Network | Spatiotemporal forecasting baseline | Establish DCRNN or STGCN baseline on traffic / PM2.5-style data |
| 3 | Copula Conformal Prediction for Multi-Step Time-Series Forecasting | Multi-step conformal calibration | Reproduce or adapt coverage, interval width, and multi-step calibration evaluation |
| 4 | Conformal Prediction for Time Series with Change Points / CPTC | Dynamic distribution shift and change-point-aware conformal calibration | Build synthetic change-point experiment and migrate to traffic / PM2.5 shift setting |

### Priority 2 - Read Deeply, Reproduce Selectively

| Order | Paper / Component | Role in Project | Extraction Target |
| ----- | ----------------- | --------------- | ----------------- |
| 5 | Provably Robust Conformal Prediction with Improved Efficiency | Robust conformal reliability under perturbation | Understand how coverage guarantees behave under perturbation |
| 6 | Evaluating Neuron Explanations: A Unified Framework with Sanity Checks | Trustworthy evaluation and meta-evaluation | Extract sanity-check logic for evaluating reliability metrics themselves |
| 7 | Prediction without Preclusion: Recourse Verification with Reachable Sets | Risk-aware decision-making and actionability | Connect uncertainty outputs to feasible actions and downstream consequences |
| 8 | U-Cast: A Simple Approach to Calibrated Weather Forecasting | Probabilistic environmental forecasting | Extract CRPS, MC Dropout, and probabilistic forecasting evaluation design |

### Priority 3 - Later Extensions

| Direction | Role |
| --------- | ---- |
| SPACY / latent structural causal models from spatiotemporal data | Later extension from distribution shift to latent causal mechanism shift |
| Koopman Neural Forecaster | Later temporal-dynamics shift baseline |
| Spherical DYffusion | Later climate generative forecasting extension |
| Zephyrus | Later scientific-agent / weather reasoning extension |
| Control and decision-focused learning papers | Later forecast-to-control extension |
| Causal data audit papers | Later data quality and causal audit extension |
| LLM / RAG / agent reliability papers | Kept as a separate Litflow-related line, not part of the current spatiotemporal forecasting core |
| AI systems / inference systems papers | Kept as a systems side line, not part of the current spatiotemporal forecasting core |

## Eight-Week Public Reading and Reproduction Plan

### Week 1-2

| Task | Public Deliverable |
| ---- | ------------------ |
| Read Quantifying UQ | 1-page paper note |
| Read CopulaCPTS | 1-page paper note |
| Establish project skeleton | README + environment plan + dataset loader plan |
| Run STGCN/DCRNN baseline | MAE / RMSE table |

### Week 3-4

| Task | Public Deliverable |
| ---- | ------------------ |
| Add MC Dropout | coverage / interval width evaluation |
| Add Deep Ensemble or Quantile Regression | UQ comparison table |
| Implement calibration metrics | ECE / reliability diagram / CRPS |
| Read Robust Conformal Prediction | robust conformal note |

### Week 5-6

| Task | Public Deliverable |
| ---- | ------------------ |
| Implement conformal wrapper | empirical coverage |
| Add temporal shift / missingness / noise | shift stress test |
| Read CPTC | change-point experiment design |
| Run first dynamic-shift experiment | coverage degradation figure |

### Week 7-8

| Task | Public Deliverable |
| ---- | ------------------ |
| Add risk-aware decision metric | Top-K allocation regret / selective risk |
| Read NeuronEval and Recourse Verification | meta-evaluation and decision-risk notes |
| Write technical report | 6-8 page technical report |
| Prepare public project summary | 1-page objective project summary |

## Public Wording Rules

Use objective terms:

* Core literature
* Reproduction priority
* Evaluation module
* Dynamic shift benchmark
* Uncertainty quantification
* Calibration
* Risk-aware decision metric
* Technical report
* Public project summary

Keep non-technical planning details outside public documentation.
