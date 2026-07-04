# Mechanism-Guided Trustworthy AI Systems

## Core Direction

Mechanism-guided trustworthy AI systems aim to make AI models understandable, calibratable, intervenable, and controllable in complex dynamic environments.

The core idea is to move beyond black-box prediction by combining:

* mechanism interpretability;
* concept representation;
* causal discovery;
* spatiotemporal graph learning;
* uncertainty calibration;
* distribution-shift evaluation;
* system-level monitoring;
* risk-aware decision-making.

## Research Thesis

The central research thesis is:

Reliable AI in complex dynamic environments requires more than high predictive accuracy. A model should expose or recover meaningful mechanisms, represent human- or system-level concepts, quantify uncertainty, remain calibrated under distribution shift, and support monitoring, intervention, and control.

## Current Flagship Project

Reliable Spatiotemporal Forecasting under Dynamic Distribution Shift:
Calibration, Uncertainty Quantification, and Risk-Aware Decision-Making.

This project uses traffic and environmental forecasting as concrete testbeds for studying:

* graph construction validity;
* dynamic propagation;
* uncertainty estimation;
* conformal calibration;
* shift-aware evaluation;
* high-risk decision reliability;
* model monitoring under noisy and incomplete observations.

## Method Stack

| Layer                         | Methods                                                                | Purpose                                                                 |
| ----------------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Mechanism discovery           | SINDy, NRI, causal time-series discovery                               | Recover or approximate underlying dynamic mechanisms                    |
| Spatiotemporal graph learning | ChebNet, STGCN, DCRNN, Graph WaveNet, AGCRN, MTGNN                     | Model structured dependencies over space and time                       |
| Causal reliability            | causal discovery, invariant prediction, shift analysis                 | Separate stable mechanisms from spurious correlations                   |
| Interpretable representation  | concept representation, CKA, probing, representation stability         | Understand what models encode and whether representations remain stable |
| Uncertainty and calibration   | MC Dropout, deep ensembles, calibration, conformal prediction          | Turn predictions into reliable uncertainty-aware forecasts              |
| Monitoring and evaluation     | drift tracking, stress tests, calibration monitoring, failure taxonomy | Detect when deployed AI systems become unreliable                       |
| Decision reliability          | selective prediction, Top-K risk ranking, recourse/actionability       | Connect prediction quality to downstream decisions                      |

## Application Domains

The same research logic can transfer across:

* traffic forecasting;
* PM2.5 and air-quality forecasting;
* energy demand and smart-grid optimization;
* climate and weather-related forecasting;
* healthcare time-series monitoring;
* industrial process monitoring;
* LLM / RAG evaluation and monitoring when system reliability is the central question.

## Research Style

This repository should prioritize:

* mathematical understanding before implementation;
* assumptions before claims;
* reliability evaluation beyond clean accuracy;
* graph and mechanism validity checks;
* reproducible notes and implementation specifications;
* research questions that can become experiments.

## Current Reading Path

Immediate:

1. P-ST-002 / DCRNN
2. P-ST-001 / STGCN
3. P-STUQ-001 / Quantifying Uncertainty in Deep Spatiotemporal Forecasting
4. P-CP-005 / CPTC

Next:

1. P-ST-004 / Graph WaveNet
2. P-ST-005 / AGCRN
3. P-ST-006 / MTGNN
4. P-MECH-001 / SINDy
5. P-MECH-002 / Neural Relational Inference
6. P-CAUSALST-002 / nonlinear time-series causal discovery

## What This Direction Is Not

This direction is not merely:

* chasing benchmark accuracy;
* collecting many forecasting models;
* using explainability as post-hoc decoration;
* treating uncertainty as a secondary metric;
* treating monitoring as an engineering afterthought.

It is about designing AI systems whose behavior can be understood, checked, calibrated, monitored, and eventually intervened upon.