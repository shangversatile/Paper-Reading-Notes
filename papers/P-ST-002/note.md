# P-ST-002: DCRNN - Diffusion Convolutional Recurrent Neural Network

## Reading Status

Status: Ready for first-pass research reading
Priority: Very high
Role: Directed diffusion, recurrent temporal modeling, and multi-step forecasting baseline
Planned output: DCRNN relearning and independent implementation specification
Advisor relevance: Rose Yu related
Project relevance: directed propagation and PM2.5 dynamic graph transfer
Relation to P-GRAPH-001: shifts from undirected spectral-local filtering to directed diffusion forecasting
Relation to P-ST-001: recurrent directed-diffusion counterpart to STGCN's convolutional temporal modeling

## Paper Metadata

* ID: P-ST-002
* Title: Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting
* Authors: Yaguang Li, Rose Yu, Cyrus Shahabi, Yan Liu
* Venue / year: ICLR 2018
* Topic: spatiotemporal-modeling
* Tier: Tier 1, Tier 3
* Task: traffic forecasting on road networks
* Core idea: model traffic flow as a diffusion process on a directed graph and combine diffusion convolution with recurrent sequence modeling
* Link: https://openreview.net/forum?id=SJiHXGWAZ
* PDF downloaded: Yes

## Why This Paper Comes After STGCN and P-GRAPH-001

P-GRAPH-001 provides the graph convolution foundation.

P-ST-001 / STGCN shows how graph convolution can be combined with temporal convolution.

DCRNN introduces a different spatiotemporal modeling path:

```text
STGCN:
    graph convolution
    temporal convolution
    fully convolutional sequence modeling

DCRNN:
    directed diffusion convolution
    recurrent temporal modeling
    encoder-decoder multi-step forecasting
    scheduled sampling
```

For PM2.5, DCRNN is important because pollution transport may be directional, dynamic, and affected by wind-driven diffusion-like propagation.

## Core Reading Questions

1. How does diffusion convolution differ from Chebyshev graph convolution?
2. Why does DCRNN use random walk transition matrices instead of only symmetric graph Laplacians?
3. What does bidirectional diffusion mean?
4. How does DCRNN combine graph diffusion with recurrent units?
5. Why does multi-step forecasting require encoder-decoder modeling?
6. What problem does scheduled sampling solve?
7. What graph construction assumptions remain in DCRNN?
8. How can wind-driven PM2.5 transport be represented as directed diffusion?
9. What uncertainty, calibration, and distribution-shift problems are not solved by DCRNN?
10. How should DCRNN be used as a reliable PM2.5 forecasting baseline?

## Mathematical Objects to Extract During Reading

Do not fill these until the paper is read carefully.

### Directed graph and random walk matrix

To be filled after reading.

### Diffusion convolution

To be filled after reading.

### Recurrent update

To be filled after reading.

### Encoder-decoder forecasting

To be filled after reading.

### Scheduled sampling

To be filled after reading.

## Reliability Critique Template

During reading, evaluate:

* graph construction validity;
* directed propagation assumption;
* temporal stationarity assumption;
* long-horizon error accumulation;
* robustness under missing sensors;
* robustness under graph perturbation;
* calibration and uncertainty gap;
* PM2.5 transfer limitations;
* comparison against STGCN.

## PM2.5 Transfer Focus

* wind-informed directed graph;
* static vs dynamic diffusion;
* long-horizon error accumulation;
* deterministic backbone before UQ and conformal calibration.
## PM2.5 Transfer Hypotheses

1. A directed wind-informed graph may be more physically meaningful than a symmetric distance graph.
2. Bidirectional diffusion may capture both upstream and downstream relations, but physical interpretation must be checked.
3. Static road-network diffusion may not directly transfer to dynamic air pollution transport.
4. DCRNN can serve as a strong deterministic backbone, but requires UQ or conformal calibration for reliability.
5. PM2.5 graph construction should compare distance, correlation, wind-informed, hybrid, and learned graphs.
6. Long-horizon PM2.5 forecasting requires evaluating error accumulation and calibration degradation.

## Independent Implementation Specification Placeholder

The planned output is a DCRNN relearning and independent implementation specification.

To be filled after reading:

* required inputs;
* directed graph construction;
* random walk matrices;
* diffusion convolution;
* recurrent cell;
* encoder-decoder setting;
* scheduled sampling setting;
* training objective;
* evaluation metrics;
* reliability stress tests;
* PM2.5 adaptation plan.

## Connection to Future Papers

After DCRNN, compare with:

* P-ST-001 / STGCN: convolutional temporal modeling instead of recurrent modeling;
* Graph WaveNet: adaptive adjacency and dilated temporal convolution;
* AGCRN: node-adaptive parameters and data-adaptive graph generation;
* Quantifying Uncertainty in Deep Spatiotemporal Forecasting: UQ wrapper and evaluation;
* CPTC: conformal prediction under time-series change points.

## First-Pass Reading Checklist

* [ ] Read abstract and introduction.
* [ ] Extract problem setting.
* [ ] Derive diffusion convolution.
* [ ] Compare diffusion convolution with ChebNet.
* [ ] Compare DCRNN with STGCN.
* [ ] Understand recurrent unit design.
* [ ] Understand encoder-decoder structure.
* [ ] Understand scheduled sampling.
* [ ] Summarize experiments and baselines.
* [ ] Identify assumptions.
* [ ] Write PM2.5 reliability transfer.
* [ ] Draft DCRNN implementation specification.
