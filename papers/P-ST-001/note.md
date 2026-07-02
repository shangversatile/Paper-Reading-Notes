# P-ST-001: STGCN - Spatio-Temporal Graph Convolutional Networks

## 1. Citation

**Paper:** Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting
**Authors:** Bing Yu, Haoteng Yin, Zhanxing Zhu
**Year:** 2018
**Venue:** IJCAI
**Primary Source:** https://www.ijcai.org/proceedings/2018/505
**Verified On:** TBD
**Reading Status:** Ready for first-pass research reading

## 2. Reading Tier and Track

**Reading Tier:** Tier 1
**Track:** spatiotemporal-modeling
**Related Project:** Reliable Spatiotemporal Forecasting under Dynamic Distribution Shift
**Optional Research Context:** General methodological foundation for deterministic spatiotemporal forecasting baseline

### Why This Paper Is in the Curriculum

STGCN is included because it is a canonical convolutional spatiotemporal graph forecasting baseline. It connects graph convolution to temporal convolution and provides a vanilla deterministic baseline before uncertainty quantification, conformal calibration, and graph reliability stress tests.

## 3. Core Problem

This paper addresses traffic forecasting on graph-structured sensor networks, where the model must capture both spatial dependency across sensors and temporal dependency over historical observations.

Current reading goal: identify how STGCN combines graph convolution and temporal convolution, and what assumptions this introduces for PM2.5 forecasting.

## 4. Intuition Before the Math

P-GRAPH-001 explains how to filter a signal over a graph. STGCN asks how to use such graph filtering repeatedly over time. The simplest mental model is:

```text
historical sensor sequence
→ temporal feature extraction
→ graph-based spatial mixing
→ temporal feature extraction
→ future prediction
```

## 5. Mathematical or Algorithmic Setup

| Object | Formal Role | Intuition |
| --- | --- | --- |
| Graph `G` | Sensor network structure | Defines which sensors can exchange spatial information |
| Adjacency `W` | Edge-weight matrix | Encodes spatial dependency assumption |
| Laplacian or graph operator | Graph convolution operator | Defines how graph convolution propagates information |
| Historical window | Input time context | Controls what past observations are visible |
| ST-Conv block | Main model block | Combines temporal and spatial processing |
| Forecast horizon | Prediction target range | Defines single-step or multi-step forecasting demand |

## 6. Method: Step-by-Step Logic

Initial scaffold to be refined after full reading:

1. Represent traffic sensors as graph nodes.
2. Use historical sensor observations as graph signals over time.
3. Apply temporal convolution to capture short-term dynamics.
4. Apply graph convolution to capture spatial dependency.
5. Stack spatiotemporal blocks.
6. Produce future traffic predictions.
7. Evaluate against deterministic forecasting baselines.

## 7. Key Equations and Derivations

Do not fabricate equations before reading.

| Equation | Meaning | Why It Matters | Implementation or Research Implication |
| --- | --- | --- | --- |
| ST-Conv block equation | TBD after reading | Defines core architecture | Needed for vanilla implementation |
| Graph convolution equation | TBD after reading | Defines spatial dependency modeling | Needed to compare with ChebNet and DCRNN |
| Temporal gated convolution equation | TBD after reading | Defines temporal feature extraction | Needed to reproduce baseline |

## 8. Assumptions

### Data Assumptions

* Sensor graph is meaningful.
* Historical traffic patterns contain predictive temporal structure.

### Model Assumptions

* Spatial dependency can be represented by a fixed graph.
* Temporal convolution is sufficient for the forecast horizon.
* Local graph filtering captures useful spatial interactions.

### Optimization or Computation Assumptions

* Fully convolutional training is more parallelizable than recurrent training.
* Model can be trained as deterministic point forecaster.

### Evaluation Assumptions

* Standard forecasting metrics are sufficient for baseline comparison.
* Reliability, calibration, and distribution shift are not primary evaluation targets in the original deterministic baseline.

For PM2.5, these assumptions may fail under wind-driven dynamic transport, missing stations, seasonal shift, sensor noise, and extreme pollution events.

## 9. Experimental Evidence

| Experiment | Dataset / Setting | What It Tests | Main Evidence | Limitation |
| --- | --- | --- | --- | --- |
| Traffic forecasting benchmark | TBD after reading | Deterministic forecasting accuracy | TBD | Need reliability and shift evaluation |
| Baseline comparison | TBD after reading | STGCN vs prior models | TBD | May not test graph perturbation or calibration |

## 10. Limitations

Current known issues to verify:

* likely fixed graph assumption;
* deterministic point prediction;
* limited uncertainty analysis;
* limited distribution-shift evaluation;
* unclear suitability for directed dynamic environmental transport;
* PM2.5 adaptation requires graph construction validation.

## 11. Research-Level Critique

Questions to answer during reading:

* Is STGCN strong because of graph convolution, temporal convolution, or both?
* How sensitive is it to graph construction?
* Does temporal convolution handle long-horizon error accumulation?
* Would dynamic graph construction improve PM2.5 forecasting?
* How should STGCN be stress-tested under missingness and graph perturbation?

## 12. Connection to My Active Project

STGCN connects to the active project as:

* deterministic forecasting backbone;
* vanilla baseline before UQ;
* graph construction stress test;
* station-level PM2.5 prediction;
* distribution shift evaluation.

## 13. Transferable Intuitions

1. Spatiotemporal forecasting requires separating spatial and temporal inductive biases.
2. A deterministic backbone must be reliable before adding UQ.
3. Fixed graph quality is a hidden validity condition.

## 14. Implementation Implications

| Component | Implication | Required Check |
| --- | --- | --- |
| Data loader | Must produce sensor-by-time windows | Verify shape and horizon |
| Graph builder | Must define fixed adjacency | Compare distance/correlation/domain graph |
| Model | Implement vanilla STGCN first | No UQ before baseline is stable |
| Metrics | Start with MAE/RMSE, then add reliability metrics | Separate point accuracy from calibration |
| Stress test | Add missingness/noise/graph perturbation | Evaluate robustness before claims |
| Experiment logging | Log graph type and forecast horizon | Reproducibility check |

## 15. Possible Research Questions

| Question | Why It Matters | Minimal Test or Evidence Needed | Related Project Component |
| --- | --- | --- | --- |
| How sensitive is STGCN to graph construction in PM2.5 forecasting? | Graph validity is central to reliability | Compare distance, correlation, PM2.5 domain graph, shuffled graph | Graph construction |
| Does STGCN degrade gracefully under missing stations? | Deployment data are incomplete | Sensor masking stress test | Missingness robustness |
| Does STGCN preserve high-risk station ranking? | Decision reliability matters | Top-K risk ranking evaluation | Risk-aware decision |

## 16. What I Should Be Able to Explain After Reading

* What problem STGCN solves.
* How STGCN differs from ChebNet.
* What an ST-Conv block does.
* Why temporal convolution is used instead of recurrence.
* What assumptions STGCN makes about graph structure.
* How STGCN should be adapted to PM2.5.
* What reliability tests are missing in the original paper.

## 17. Follow-Up Actions

| Action | Target File or Project Component | Status |
| --- | --- | --- |
| Complete first-pass reading | papers/P-ST-001/note.md | Planned |
| Extract core equations | Section 7 | Planned |
| Draft vanilla implementation specification | Reliable-AI-Research-Lab future task only, not this commit | Planned |
| Compare with DCRNN | papers/P-ST-002/note.md | Planned |

## 18. Completion Criteria

Keep the paper as Ready for first-pass research reading until all core sections are filled. Mark Completed only after assumptions, equations, experiments, limitations, project connection, research questions, and implementation implications are filled.
