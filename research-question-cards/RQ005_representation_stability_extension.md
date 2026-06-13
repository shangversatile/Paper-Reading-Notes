# RQ005: Representation Stability Extension

## Question ID

RQ005

## Research Question

Does prediction-reconstruction multi-task learning improve latent representation stability under shift, and is representation stability associated with calibration or decision stability?

## Why It Matters

Representation-level diagnostics may explain why forecast reliability fails, but they should grow from concrete MVP failures rather than become a separate project too early.

## Hypothesis

Prediction-reconstruction learning may improve representation stability without necessarily improving calibration.

## Minimal Experiment

Compare prediction-only and prediction-reconstruction models after MVP metrics identify a concrete representation-level diagnostic question.

## Required Assets

Saved model outputs, latent activations, shift labels, calibration metrics, decision metrics, and integration target in the main project.

## Baselines

Prediction-only model, prediction-reconstruction multi-task model, and representation diagnostics marked as Candidate.

## Metrics

Layer-wise activation statistics, cosine similarity, PCA or SVD summaries, CKA or comparable similarity, calibration metrics, and decision-stability metrics.

## Risks

Representation changes may be difficult to interpret, and apparent stability may not imply better calibration or better decisions.

## Related Papers

To be populated only after primary sources are verified.

## Research Context

Relevant contexts: latent state, invariance, intervention, and representation stability.

## MVP or Extension

Week 7 extension.

## Status

Candidate. No results claimed.
