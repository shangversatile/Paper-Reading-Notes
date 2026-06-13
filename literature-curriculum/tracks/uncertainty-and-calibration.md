# Uncertainty and Calibration

## Scope

Study aleatoric and epistemic uncertainty, Bayesian approximation, ensembles, interval quality, probabilistic metrics, and calibration.

## Core Concepts

* aleatoric and epistemic uncertainty
* Bayesian approximation
* ensembles
* interval quality
* probabilistic metrics
* calibration

## Questions to Track

* What uncertainty is being estimated, and by what assumption?
* How should interval quality be evaluated beyond raw coverage?
* When do probabilistic metrics disagree with downstream reliability?
* How does calibration degrade under shift?

## Connection to Active Project

This track supports MC Dropout, Deep Ensemble, probabilistic metrics, and calibration evaluation in the active forecasting project.

## Long-Term Relevance

Uncertainty and calibration are central to reliable AI systems, selective prediction, monitoring, and risk-aware decisions.

## Reading Queue

| Priority | Paper | Reading Tier | Why Read | Reproduction Value | Active Project Connection | Research Context | Primary Source Verified | Status |
| -------- | ----- | ------------ | -------- | ------------------ | ------------------------- | ------------------ | ----------------------- | ------ |
| 1 | P-UQ-001 | Tier 0, Tier 1 | Approximate Bayesian interpretation of dropout uncertainty | TBD after reading | MC Dropout relearning and inference-time sampling protocol | General methodological foundation | Yes | Queued |
| 2 | P-UQ-002 | Tier 1 | Strong and scalable uncertainty baseline | TBD after reading | Deep Ensemble implementation specification and runtime comparison | General methodological foundation | Yes | Queued |
| 3 | P-EVAL-001 | Tier 0 | Foundation for evaluating probabilistic forecasts | TBD after reading | CRPS, sharpness, and probabilistic evaluation protocol | General methodological foundation | Yes | Queued |
