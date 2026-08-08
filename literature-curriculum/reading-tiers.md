# Reading Tiers

## Tier 0: Foundational

Build concepts, mathematical prerequisites, and terminology.

## Tier 1: Canonical or Landmark

Define influential models, methods, baselines, or evaluation paradigms.

## Tier 2: Frontier

Expose current open problems, limitations, and emerging methods.

## Tier 3: Contextual or Research-Ecosystem Relevance

Support the understanding of how a paper connects to an active
research ecosystem, application context, or collaboration direction.

## Current Reliable Forecasting Priority

This priority schedule is project-facing and does not replace the
foundational tier system above.

### Priority 1 - Read and Reproduce

| Order | Paper / Component | Public Role |
| ----- | ----------------- | ----------- |
| 1 | Quantifying Uncertainty in Deep Spatiotemporal Forecasting | Core UQ and spatiotemporal forecasting bridge |
| 2 | DCRNN or STGCN baseline | Forecasting baseline for traffic / PM2.5-style data |
| 3 | Copula Conformal Prediction for Multi-Step Time-Series Forecasting | Multi-step conformal calibration |
| 4 | Conformal Prediction for Time Series with Change Points / CPTC | Change-point-aware dynamic shift calibration |

### Priority 2 - Read Deeply, Reproduce Selectively

| Order | Paper / Component | Public Role |
| ----- | ----------------- | ----------- |
| 5 | Provably Robust Conformal Prediction with Improved Efficiency | Robust conformal reliability under perturbation |
| 6 | Evaluating Neuron Explanations: A Unified Framework with Sanity Checks | Trustworthy evaluation and meta-evaluation |
| 7 | Prediction without Preclusion: Recourse Verification with Reachable Sets | Risk-aware decision-making and actionability |
| 8 | U-Cast: A Simple Approach to Calibrated Weather Forecasting | Probabilistic environmental forecasting metrics |

### Priority 3 - Later Extensions

| Direction | Public Role |
| --------- | ----------- |
| SPACY / latent structural causal models | Extend shift analysis toward latent causal mechanism shift |
| Koopman Neural Forecaster | Later temporal-dynamics shift baseline |
| Spherical DYffusion | Later climate generative forecasting extension |
| Zephyrus | Later scientific-agent / weather reasoning extension |
| Control and decision-focused learning | Later forecast-to-control extension |
| Causal data audit | Later data quality and causal audit extension |
| LLM / RAG / agent reliability | Separate Litflow-related line |
| AI systems / inference systems | Separate systems side line |

### Learning-Dynamics Bridge Queue - Evidence-Gated

This queue supports the Sadhika Malladi advisor-preparation route. It is a theory bridge for understanding how optimization dynamics produce model states, representations, and behavior. It does not replace the current reliable spatiotemporal forecasting priority.

| Stage | Paper IDs | Why Read |
| ----- | --------- | -------- |
| 1 - Optimization Dynamics | P-LD-001 -> P-LD-002 | Learn how SGD and adaptive optimizers become analyzable stochastic dynamics and how scaling rules emerge from theory. |
| 2 - Fine-Tuning Dynamics | P-LD-003 -> P-LD-004 | Understand what pretrained representations preserve, what fine-tuning changes, and why pretrained models alter the optimization landscape. |
| 3 - Data to Learning Dynamics | P-LD-005 | Ask which observations drive parameter updates, representation changes, and downstream behavior. |
| 4 - Objective to Unexpected Behavior | P-LD-006 -> P-LD-008 | Study how preference objectives can fail to control the behavior they appear to target. |
| 5 - Training Trajectory / Representation Plasticity | P-LD-007 -> P-LD-009 | Track how training procedures and pretraining duration affect learning order and adaptability. |
| 6 - Pretraining to Post-training Theory | P-LD-010 | Connect coverage, representation support, adaptation, and generalization in the latest post-training theory queue. |

## Notes

One paper may have multiple tiers.

Tier 3 is not automatically more important than Tier 0 or Tier 1.

High-learning-value papers must be read even when they have no immediate project context.

Contextual papers may be skimmed rather than deeply reproduced
when their learning value is limited for the current research direction.

Project priority does not imply that reproduction has been completed.
Use planned, queued, or to-study status until evidence exists.
