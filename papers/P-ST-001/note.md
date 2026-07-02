# P-ST-001: STGCN - Spatio-Temporal Graph Convolutional Networks

## Reading Status

Status: Ready for first-pass research reading
Priority: Very high
Role: Core convolutional spatiotemporal forecasting baseline
Planned output: Vanilla STGCN relearning and independent implementation specification
Project relevance: deterministic baseline for reliable PM2.5 forecasting
Relation to P-GRAPH-001: direct extension from Chebyshev graph convolution to spatiotemporal forecasting

## Paper Metadata

* ID: P-ST-001
* Title: Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting
* Authors: Bing Yu, Haoteng Yin, Zhanxing Zhu
* Venue / year: IJCAI 2018
* Topic: spatiotemporal-modeling
* Tier: Tier 1
* Task: traffic forecasting
* Core idea: combine graph convolution with gated temporal convolution in a fully convolutional spatiotemporal architecture
* Link: https://www.ijcai.org/proceedings/2018/505
* PDF downloaded: Yes

## Why This Paper Comes After P-GRAPH-001

P-GRAPH-001 explains how Chebyshev polynomial filters make graph convolution localized, parameter-efficient, and scalable.

STGCN is a natural next step because it uses graph convolution inside a temporal forecasting architecture.

The transition is:

```text
ChebNet:
    graph convolution on static graph signals
    Chebyshev polynomial filtering
    no temporal forecasting module

STGCN:
    graph convolution for spatial dependency
    temporal convolution for sequence dependency
    fully convolutional spatiotemporal blocks
    traffic forecasting baseline
```

For PM2.5, STGCN is useful as a deterministic baseline before adding uncertainty quantification, conformal calibration, graph perturbation tests, and decision-aware evaluation.

## Core Reading Questions

1. What graph convolution does STGCN use?
2. How does STGCN connect Chebyshev graph convolution with temporal forecasting?
3. What is the ST-Conv block?
4. How does gated temporal convolution work?
5. Why does STGCN avoid recurrent units?
6. How does STGCN handle multi-step prediction?
7. What graph construction assumptions does STGCN inherit?
8. Does STGCN support directed or dynamic graphs?
9. How should STGCN be reproduced as a vanilla baseline?
10. What reliability gaps remain for PM2.5 forecasting?

## Mathematical Objects to Extract During Reading

Do not fill these until the paper is read carefully.

### Graph convolution module

To be filled after reading.

### Temporal gated convolution

To be filled after reading.

### ST-Conv block

To be filled after reading.

### Forecasting objective

To be filled after reading.

## Reliability Critique Template

During reading, evaluate:

* static graph assumption;
* whether the graph encodes valid locality;
* temporal convolution receptive field;
* robustness under graph mismatch;
* robustness under missing sensors;
* robustness under noise and spike anomalies;
* calibration and uncertainty gap;
* station-level prediction suitability;
* comparison against DCRNN;
* PM2.5 transfer limitations.

## PM2.5 Transfer Focus

* static graph validity;
* station-level forecasting;
* robustness under missingness and graph perturbation;
* baseline before UQ and conformal calibration.
## PM2.5 Transfer Hypotheses

1. STGCN may be easier to train and reproduce than recurrent architectures.
2. Temporal convolution may be more parallelizable and stable than recurrent forecasting.
3. Static graph convolution may be insufficient for wind-driven dynamic PM2.5 transport.
4. STGCN is a good deterministic baseline before adding UQ and conformal calibration.
5. Reliability evaluation should compare STGCN with DCRNN under noise, missingness, temporal shift, and graph perturbation.
6. Vanilla STGCN should be implemented before adding adaptive graphs or uncertainty modules.

## Independent Implementation Specification Placeholder

The planned output is a vanilla STGCN relearning and independent implementation specification.

To be filled after reading:

* required inputs;
* graph construction;
* temporal window;
* model blocks;
* training objective;
* evaluation metrics;
* reliability stress tests;
* PM2.5 adaptation plan.

## Connection to Future Papers

After STGCN, compare with:

* P-ST-002 / DCRNN: directed diffusion and recurrent multi-step prediction;
* Graph WaveNet: adaptive adjacency and dilated temporal convolution;
* AGCRN: node-adaptive parameters and data-adaptive graph generation;
* MTGNN: learned directed graph for multivariate time series;
* Quantifying Uncertainty in Deep Spatiotemporal Forecasting: UQ methods;
* CPTC: conformal prediction under change points.

## First-Pass Reading Checklist

* [ ] Read abstract and introduction.
* [ ] Extract model architecture.
* [ ] Understand graph convolution module.
* [ ] Understand temporal gated convolution.
* [ ] Understand ST-Conv block.
* [ ] Compare with P-GRAPH-001 / ChebNet.
* [ ] Compare with DCRNN after reading P-ST-002.
* [ ] Summarize experiments and baselines.
* [ ] Identify graph assumptions.
* [ ] Write PM2.5 reliability transfer.
* [ ] Draft vanilla STGCN implementation specification.
