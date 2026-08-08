# Reading Map

## Reading Stages

1. Spatiotemporal forecasting baselines
2. Uncertainty quantification
3. Conformal prediction
4. Reliability under distribution shift
5. Decision-aware forecasting
6. Probabilistic environmental forecasting extension
7. Representation, causal, learning-dynamics, and systems extensions only after the core path is stable

## Active Public Roadmap

The eight-week public literature and reproduction plan is maintained in
`topics/reliable_spatiotemporal_forecasting/8-week-literature-and-reproduction-plan.md`.

A dedicated spatiotemporal graph forecasting queue is maintained in
`topics/reliable_spatiotemporal_forecasting/spatiotemporal_graph_forecasting_reading_queue.md`.

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
| Learning dynamics and optimization papers | Gated theory bridge for explaining representation/adaptation behavior through training dynamics, not part of the current forecasting MVP |

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

### Phase D: Gated Learning-Dynamics Bridge

This phase supports the Sadhika Malladi advisor-preparation route only after the core forecasting and reliability path exposes a concrete representation-stability or adaptation question. It is a learning-dynamics bridge, not evidence that Sadhika works on spatiotemporal forecasting.

| Stage | Paper IDs | Research Question |
| ----- | --------- | ----------------- |
| 1 - Optimization Dynamics | P-LD-001 -> P-LD-002 | Can optimization trajectories be modeled as analyzable stochastic dynamics? |
| 2 - Fine-Tuning Dynamics | P-LD-003 -> P-LD-004 | What do pretrained representations preserve, and what does fine-tuning actually change? |
| 3 - Data to Learning Dynamics | P-LD-005 | Which observations drive parameter updates, representation changes, and downstream behavior? |
| 4 - Objective to Unexpected Behavior | P-LD-006 -> P-LD-008 | Does the objective control the behavior we think it controls? |
| 5 - Training Trajectory / Representation Plasticity | P-LD-007 -> P-LD-009 | When do training procedures make representations stable, adaptable, or brittle? |
| 6 - Pretraining to Post-training Theory | P-LD-010 | How does pretraining coverage enable later adaptation and generalization? |

Potential bridge / research hypothesis:

Environment dynamics
-> observed data distribution
-> learning / adaptation dynamics
-> parameter trajectory
-> representation trajectory
-> prediction / decision behavior

This is an analytical bridge from Sadhika Malladi's learning-dynamics methodology to the user's representation/mechanism direction. It does not mean she currently studies spatiotemporal representation drift.
