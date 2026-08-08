# Learning Dynamics and Optimization

## Scope

Study how data, objectives, optimizers, hyperparameters, and training procedures produce model states, representations, and downstream behavior.

This track is not a general LLM track. It is the learning-dynamics theory route most relevant to the Sadhika Malladi advisor-preparation path.

## Core Concepts

* SGD and SDE approximation
* adaptive optimizer scaling rules
* fine-tuning dynamics
* kernel-regime approximations for pretrained models
* zeroth-order optimization
* data influence for instruction tuning
* preference-learning objective mismatch
* training trajectory and representation plasticity
* pretraining coverage for post-training

## Questions to Track

* What information about training dynamics is preserved or lost when SGD is approximated by an SDE?
* How do optimizer choices and scaling rules alter the parameter trajectory?
* What does fine-tuning preserve from pretrained representations, and what does it actually change?
* Which samples drive parameter updates and downstream behavior changes?
* When does an objective fail to control the property it appears to optimize?
* Can representation drift under adaptation be studied through the learning dynamics that produce it?

## Connection to Active Project

This is a gated theory bridge for the long-term direction:

Real-world Representation and Mechanism Learning in Dynamic Systems.

The immediate active project remains reliable spatiotemporal forecasting under dynamic distribution shift. P-LD papers become active only after there is a concrete representation-stability or adaptation question that needs a learning-dynamics explanation.

## Boundary

Do not describe Sadhika Malladi as a spatiotemporal forecasting, causal discovery, uncertainty quantification, conformal prediction, or mechanistic interpretability advisor based on this track.

The current bridge is analytical:

data / objective / optimization
-> training dynamics
-> parameter trajectory
-> representation trajectory
-> downstream behavior
-> generalization / failure

## Reading Queue

| Priority | Paper | Reading Tier | Why Read | Reproduction Value | Active Project Connection | Research Context | Primary Source Verified | Status |
| -------- | ----- | ------------ | -------- | ------------------ | ------------------------- | ---------------- | ----------------------- | ------ |
| 1 | P-LD-003 | Tier 1, Tier 3 | Required Sadhika paper for serious conversation about fine-tuning and pretrained representations | Conceptual note first; reproduction TBD | Bridge from representation formation to downstream behavior | Sadhika Malladi advisor-preparation route | Yes | Queued |
| 2 | P-LD-001 | Tier 1, Tier 3 | Foundation for treating SGD as analyzable stochastic dynamics | Math-focused reading note | Supports dynamic-systems intuition for training trajectories | Optimization-dynamics foundation | Yes | Queued |
| 3 | P-LD-002 | Tier 1, Tier 3 | Shows theory -> prediction -> empirical validation for adaptive optimizers and scaling rules | Scaling-rule extraction note | Connects optimizer hyperparameters to dynamics | Optimization-dynamics foundation | Yes | Queued |
| 4 | P-LD-004 | Tier 1, Tier 3 | Explains why zeroth-order fine-tuning can work in pretrained language models | Algorithm-mechanism note | Helps interpret pretrained representation and optimization landscape | Fine-tuning dynamics | Yes | Queued |
| 5 | P-LD-005 | Tier 1, Tier 3 | Links data selection to parameter updates and behavior change | Influence-signal extraction note | Potential route for analyzing adaptation or representation drift | Data-to-learning-dynamics bridge | Yes | Queued |
| 6 | P-LD-008 | Tier 1, Tier 3 | Strong reliability example where post-training objective changes behavior in unintended ways | Failure-mechanism note | Connects objective control to trustworthy model behavior | Objective-behavior mismatch | Yes | Queued |
| 7 | P-LD-006 | Tier 2, Tier 3 | Companion paper on preference objective mismatch | Comparison note with P-LD-008 | Later reliability/alignment extension | Objective-behavior mismatch | Yes | Queued |
| 8 | P-LD-007 | Tier 2, Tier 3 | Studies implicit curriculum induced by progressive distillation | Training-trajectory note | Later representation-plasticity extension | Training trajectory | Yes | Queued |
| 9 | P-LD-009 | Tier 2, Tier 3 | Studies why overtraining may reduce downstream adaptability | Adaptability note | Later stable-vs-adaptable representation question | Representation plasticity | Yes | Queued |
| 10 | P-LD-010 | Tier 2, Tier 3 | Latest frontier route on how pretraining enables post-training | Frontier theory note | Later coverage/support/adaptation question | Pretraining to post-training theory | Yes | Queued |
