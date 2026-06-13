# RQ001: Failure under Shift

## Question ID

RQ001

## Research Question

How do spatiotemporal forecasting models fail under noise, spike anomalies, missingness, and temporal distribution shift?

## Why It Matters

Forecasting models can look acceptable under near-IID evaluation while failing under operational corruptions or temporal regime changes.

## Hypothesis

Different shift types may induce distinct error, calibration, and decision-risk failure modes.

## Minimal Experiment

Compare baseline forecasting models on KnowAir PM2.5 under clean evaluation and controlled shift stress tests.

## Required Assets

KnowAir PM2.5 data, documented split logic, corruption protocol, fixed seeds, baseline model outputs, and metric reports.

## Baselines

Persistence, Historical Average, Vanilla STGCN, DCRNN, and Robust-STGCN only after the baseline pipeline is reliable.

## Metrics

MAE, RMSE, MAPE, coverage, coverage gap, interval width, sharpness, CRPS, and failure-case counts.

## Risks

Shift definitions may be too artificial, split leakage may distort conclusions, and old thesis scripts may not be reproducible without audit.

## Related Papers

To be populated only after primary sources are verified.

## Advisor Connections

Candidate connections: robustness, environmental forecasting, dynamic shift, failure analysis.

## MVP or Extension

MVP.

## Status

Candidate. No results claimed.
