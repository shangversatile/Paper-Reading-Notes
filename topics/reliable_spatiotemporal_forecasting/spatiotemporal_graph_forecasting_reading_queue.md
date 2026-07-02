# Spatiotemporal Graph Forecasting Reading Queue

## Completed

1. P-GRAPH-001 / ChebNet
   Role: graph convolution foundation; Chebyshev polynomial filtering; graph quality critique.

## Immediate Reading

1. P-ST-001 / STGCN
   Role: core convolutional spatiotemporal forecasting baseline; vanilla STGCN relearning and independent implementation specification.

2. P-ST-002 / DCRNN
   Role: directed diffusion, recurrent temporal modeling, and multi-step forecasting baseline; DCRNN relearning and independent implementation specification.

## Next Core Models

1. Graph WaveNet
   Role: adaptive adjacency, dilated temporal convolution, hidden dependency learning.

2. AGCRN
   Role: node-adaptive parameter learning and data-adaptive graph generation.

3. MTGNN
   Role: learned uni-directed graph for multivariate time series forecasting.

## Attention and Transformer Extensions

1. ASTGCN
   Role: spatial-temporal attention; recent, daily, and weekly temporal components.

2. GMAN
   Role: graph multi-attention for long-term traffic prediction.

3. PDFormer
   Role: propagation delay-aware dynamic long-range Transformer.

4. STAEformer
   Role: simple strong Transformer baseline with spatiotemporal adaptive embedding.

## Continuous-Time and Missingness-Aware Extensions

1. STG-NCDE
   Role: graph neural controlled differential equations; irregular and missing time series.

## Reliability and Calibration

1. Quantifying Uncertainty in Deep Spatiotemporal Forecasting
   Role: UQ method taxonomy and evaluation protocol.

2. CPTC
   Role: conformal prediction for time series under change points.

3. Provably Robust Conformal Prediction with Improved Efficiency
   Role: robust conformal prediction under perturbation.

## Project Principle

DCRNN and STGCN are foundational, but spatiotemporal forecasting should not be reduced to only these two models.

A reliable PM2.5 forecasting project must also understand:

* adaptive graph construction;
* dynamic propagation;
* node heterogeneity;
* attention or Transformer-based temporal modeling;
* uncertainty quantification;
* conformal calibration;
* graph validity under distribution shift.