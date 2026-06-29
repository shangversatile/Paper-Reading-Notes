# Reading Map

## Reading Stages

1. Spatiotemporal forecasting baselines
2. Uncertainty quantification
3. Conformal prediction
4. Reliability under distribution shift
5. Decision-aware forecasting
6. Probabilistic environmental forecasting extension
7. Representation, causal, and systems extensions only after the core path is stable

## Active Public Roadmap

The eight-week public literature and reproduction plan is maintained in
`topics/reliable_spatiotemporal_forecasting/8-week-literature-and-reproduction-plan.md`.

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

## Paper Queue

Do not add paper claims until the primary source has been checked.

| Priority | Paper | Primary Source Verified | Why Read | Connected RQ | Reading Status | Notes Path |
| -------- | ----- | ----------------------- | -------- | ------------ | -------------- | ---------- |
| Priority 1 | Quantifying Uncertainty in Deep Spatiotemporal Forecasting | No - metadata-to-verify | UQ comparison for spatiotemporal forecasting | UQ reliability under shift | Planned | TBD |
| Priority 1 | DCRNN: Diffusion Convolutional Recurrent Neural Network | Yes | Forecasting baseline | Baseline error and multi-step forecasting | Queued | TBD |
| Priority 1 | Copula Conformal Prediction for Multi-Step Time-Series Forecasting | Yes | Multi-step conformal calibration | Multi-step coverage and interval width | Queued | TBD |
| Priority 1 | Conformal Prediction for Time Series with Change Points / CPTC | Yes | Change-point-aware conformal calibration | Dynamic shift coverage degradation | Queued | TBD |
| Priority 2 | Provably Robust Conformal Prediction with Improved Efficiency | No - metadata-to-verify | Robust conformal guarantee behavior | Coverage under perturbation | Planned | TBD |
| Priority 2 | Evaluating Neuron Explanations: A Unified Framework with Sanity Checks | No - metadata-to-verify | Meta-evaluation pattern for reliability metrics | Metric sanity checks | Planned | TBD |
| Priority 2 | Prediction without Preclusion: Recourse Verification with Reachable Sets | No - metadata-to-verify | Link predictions to feasible downstream actions | Decision-risk evaluation | Planned | TBD |
| Priority 2 | U-Cast: A Simple Approach to Calibrated Weather Forecasting | No - metadata-to-verify | Probabilistic environmental forecasting design | CRPS and calibrated forecasting evaluation | Planned | TBD |

## Verified Reading Sequence

### Phase A: Before Any Implementation

1. P-GRAPH-001
2. P-GRAPH-002
3. P-ST-001
4. P-ST-002
5. P-ST-003
6. P-UQ-001
7. P-UQ-002
8. P-EVAL-001
9. P-CP-001

### Phase B: Before Shift and Decision Experiments

10. P-SHIFT-001
11. P-CP-002
12. P-CP-003
13. P-SEL-001

### Phase C: Current Priority and Gated Extensions

14. P-STUQ-001
15. P-CP-004
16. P-CP-005
17. P-ROB-001
18. P-EVAL-002
19. P-DEC-001
20. P-ENV-001
21. P-REP-001
22. P-AQ-001

Phase C papers must not expand MVP scope prematurely.

P-AQ-001 is a frontier watchlist preprint, not a mandatory MVP benchmark.

Contextual relevance does not override foundational learning priority.
