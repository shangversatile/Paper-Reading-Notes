# RQ004: Selective Prediction and Review

## Question ID

RQ004

## Research Question

Can uncertainty estimates support selective prediction, conservative allocation, human review, and alert escalation?

## Why It Matters

Reliable systems may need to defer, escalate, or act conservatively when uncertainty is high.

## Hypothesis

Uncertainty-aware selective prediction may reduce severe failures while increasing review rate or average resource cost.

## Minimal Experiment

Define uncertainty thresholds and compare normal prediction, conservative allocation, review, and alert policies under shift.

## Required Assets

Uncertainty scores, calibrated intervals, review-threshold rules, allocation-cost model, and failure labels.

## Baselines

Always predict, Mean Policy, Risk-Averse Policy, Upper-Quantile Policy, and threshold-based review.

## Metrics

Risk–coverage curve, AURC, review rate, average decision cost, tail risk, CVaR, and constraint violation rate.

## Risks

Thresholds may overfit validation data, and human-review simulations may not reflect real deployment workflows.

## Related Papers

To be populated only after primary sources are verified.

## Research Context

Relevant contexts: abstention, failure analysis, decision reliability, and alert escalation.

## MVP or Extension

MVP.

## Status

Candidate. No results claimed.
