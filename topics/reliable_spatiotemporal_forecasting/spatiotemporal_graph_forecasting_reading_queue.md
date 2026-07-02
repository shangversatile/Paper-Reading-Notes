# Spatiotemporal Graph Forecasting Reading Queue

## Completed

1. P-GRAPH-001 / ChebNet
   Role: graph convolution foundation; Chebyshev polynomial filtering; graph quality critique.

## Immediate Reading

1. P-ST-001 / STGCN
   Role: core convolutional spatiotemporal forecasting baseline; vanilla STGCN relearning and independent implementation specification.

2. P-ST-002 / DCRNN
   Role: directed diffusion, recurrent temporal modeling, and multi-step forecasting baseline; Rose Yu related.

## Core Next Models

1. P-ST-003 / PM2.5-GNN
   Role: domain-aware graph construction and environmental forecasting context.

2. P-ST-004 / Graph WaveNet
   Role: adaptive adjacency and hidden spatial dependency learning.

3. P-ST-005 / AGCRN
   Role: node-adaptive parameters and data-adaptive graph generation.

4. P-ST-006 / MTGNN
   Role: learned directed dependency for multivariate time series.

## Advisor-Relevant Reliability Papers

1. P-STUQ-001 / Quantifying Uncertainty in Deep Spatiotemporal Forecasting
   Role: Rose Yu related UQ bridge for spatiotemporal forecasting.

2. P-CP-005 / CPTC
   Role: conformal prediction under time-series change points.

3. P-ROB-001 / Provably Robust Conformal Prediction with Improved Efficiency
   Role: Lily Weng related robust conformal prediction.

4. P-EVAL-002 / Evaluating Neuron Explanations
   Role: Lily Weng related trustworthy evaluation; later representation-analysis stage.

5. P-DEC-001 / Prediction without Preclusion
   Role: Lily Weng / UCSD related actionability and decision-risk extension.

## Mechanism Discovery and Reliable Dynamic Systems

The immediate reading remains P-ST-001 / STGCN and P-ST-002 / DCRNN. These new papers extend the longer-term direction toward mechanism discovery, causal structure learning, reliable uncertainty, and applied forecasting. They should not interrupt the current STGCN-DCRNN-STUQ-CPTC path.

### First mechanism-discovery additions

1. P-MECH-001 / SINDy
   Role: governing-equation discovery for nonlinear dynamical systems.

2. P-MECH-002 / Neural Relational Inference
   Role: latent interaction graph discovery and dynamics learning.

3. P-CAUSALST-002 / Detecting and Quantifying Causal Associations in Large Nonlinear Time Series Datasets
   Role: causal discovery for high-dimensional nonlinear time series.

4. P-CAUSAL-001 / NOTEARS
   Role: continuous optimization foundation for DAG structure learning.

5. P-CAUSAL-002 / Invariant Causal Prediction
   Role: causal invariance and robust prediction across environments.

### Reliability and applied forecasting additions

1. P-CAL-001 / On Calibration of Modern Neural Networks
   Role: calibration and model monitoring foundation.

2. P-TS-001 / Temporal Fusion Transformer
   Role: interpretable multi-horizon forecasting baseline.

3. P-DYN-002 / Neural Controlled Differential Equations
   Role: irregular sampling and missingness-aware continuous-time modeling.

## Deferred Watchlist

Do not add these to master index yet:

* ASTGCN
* GMAN
* PDFormer
* STAEformer
* STG-NCDE

These may be considered after STGCN, DCRNN, Graph WaveNet, AGCRN, MTGNN, STUQ, and CPTC are understood.
