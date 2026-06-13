# RQ002: UQ Ranking under Shift

## Question ID

RQ002

## Research Question

Do MC Dropout, Deep Ensembles, and Split Conformal Prediction retain their reliability ranking under distribution shift?

## Why It Matters

Uncertainty methods are often compared under limited conditions, but operational reliability depends on whether rankings survive realistic shifts.

## Hypothesis

UQ methods that are reliable under near-IID conditions may degrade unevenly under shift.

## Minimal Experiment

Evaluate MC Dropout, Deep Ensembles, and Split Conformal Prediction on the same forecasting backbone and stress-test protocol.

## Required Assets

Shared data splits, uncertainty outputs, calibration protocol, conformal calibration split, and reproducible metric scripts.

## Baselines

Point forecast without UQ, MC Dropout, Deep Ensemble, and Split Conformal Prediction.

## Metrics

Coverage, coverage gap, interval width, sharpness, CRPS, risk–coverage curve, AURC, and tail-risk metrics.

## Risks

Unfair compute budgets, invalid conformal split assumptions, and unverified calibration procedures could bias method rankings.

## Related Papers

To be populated only after primary sources are verified.

## Advisor Connections

Candidate connections: conformal coverage, Bayesian UQ, statistical guarantees, and reliability under shift.

## MVP or Extension

MVP.

## Status

Candidate. No results claimed.
