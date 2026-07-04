# Paper-Reading-Notes

## Purpose

This repository is a structured research reading and synthesis workspace.

It supports the development of durable technical understanding,
critical reading habits, reproducible research questions,
and executable experimental ideas across reliable machine learning
and related areas.

## Research Workflow

Paper
-> Concepts and Assumptions
-> Critical Analysis
-> Open Question
-> Testable Hypothesis
-> Reproduction or Minimal Experiment
-> Research Output

## Research Direction

This repository supports my long-term research direction:

Mechanism-guided Trustworthy AI Systems.

The goal is to study how AI models can be made understandable,
calibratable, intervenable, and controllable in complex dynamic
environments through mechanism interpretability, concept representations,
causal discovery, spatiotemporal graph learning, uncertainty calibration,
and system-level monitoring.

Current flagship line:

Reliable Spatiotemporal Forecasting under Dynamic Distribution Shift:
Calibration, Uncertainty Quantification, and Risk-Aware Decision-Making.

## Current Core Threads

* Spatiotemporal graph forecasting: ChebNet, STGCN, DCRNN, Graph WaveNet, AGCRN, MTGNN.
* Mechanism discovery: SINDy, Neural Relational Inference, causal time-series discovery.
* Uncertainty and calibration: MC Dropout, Deep Ensembles, calibration, conformal prediction.
* Distribution-shift evaluation: robustness, stress testing, monitoring, failure analysis.
* Interpretable representation: concept representation, CKA, probing, representation stability.
* Decision reliability: selective prediction, risk-aware allocation, actionability.

## Current Public Roadmap

The active public roadmap is tracked in
`topics/reliable_spatiotemporal_forecasting/8-week-literature-and-reproduction-plan.md`.

It records:

* core literature priority
* reproduction priority
* evaluation modules
* dynamic shift benchmark design
* uncertainty and calibration metrics
* risk-aware decision metrics
* 8-week technical deliverables

## Literature Curriculum

The reading curriculum includes foundational, canonical, frontier,
and application-relevant work.

Papers are selected based on conceptual importance,
methodological influence, evaluation value, reproducibility value,
and relevance to active or likely future research questions.

## Repository Structure

* `literature-curriculum/`: long-term reading tracks and verified paper index
* `topics/`: project-specific reading maps and terminology
* `papers/`: verified paper notes and critical reading records
* `research-question-cards/`: candidate questions, hypotheses, and minimal experiments
* `backlog/`: deferred research directions and activation gates
* `templates/`: reusable reading and research-question templates

## Verification Rule

Paper metadata, source links, and technical claims are added only
after primary-source verification.

When metadata is not verified locally, title-level entries are marked
`metadata-to-verify` and are not given invented years, venues, links,
author lists, or paper IDs.
