# Paper ID Taxonomy

Paper IDs use the pattern:

`P-<TRACK>-<NUMBER>`

* `P` means paper.
* `<TRACK>` is a short research-track prefix.
* `<NUMBER>` is a zero-padded sequence number within that track.
* The prefix is for curriculum organization, not a claim that the paper belongs only to one field.
* If a paper spans multiple areas, it is assigned to the track most relevant to the current project.

| Prefix | Meaning | Typical Role in This Repository |
| ------ | ------- | -------------------------------- |
| P-GRAPH | Graph neural network foundations | Graph convolution, graph filtering, graph representation basics |
| P-ST | Spatiotemporal forecasting models | STGCN, DCRNN, PM2.5-GNN, Graph WaveNet, AGCRN, MTGNN |
| P-STUQ | Spatiotemporal uncertainty quantification | UQ methods designed for or evaluated on spatiotemporal forecasting |
| P-UQ | General uncertainty quantification | MC Dropout, Deep Ensembles, probabilistic uncertainty baselines |
| P-EVAL | Evaluation methodology | Proper scoring rules, neuron explanation evaluation, reliability metrics |
| P-CP | Conformal prediction | Split CP, adaptive CP, time-series CP, multi-step CP |
| P-SHIFT | Distribution shift and robustness | Uncertainty under shift, stress tests, OOD reliability |
| P-SEL | Selective prediction and reject option | Risk-coverage, abstention, human review policy |
| P-REP | Representation learning and analysis | CKA, representation stability, hidden-state diagnostics |
| P-AQ | Air quality and environmental forecasting | Air quality benchmarks and PM2.5 forecasting context |
| P-ROB | Robustness and robust reliability | Robust conformal prediction, perturbation-aware guarantees |
| P-DEC | Decision reliability and actionability | Recourse, downstream decision risk, actionability |
| P-ENV | Environmental probabilistic forecasting | Weather, calibrated environmental forecasting, CRPS-oriented evaluation |
| P-CAUSALST | Causal spatiotemporal modeling | Causal discovery and latent mechanisms in time-dependent systems |
| P-CAUSAL | General causal discovery and invariance | DAG learning, invariant prediction, causal structure learning |
| P-MECH | Mechanism discovery | Governing equations, latent interactions, interpretable dynamics |
| P-DYN | Temporal dynamics modeling | Koopman methods, Neural CDE, continuous-time dynamics |
| P-CLIMATE | Climate forecasting | Climate-specific forecasting and generative modeling |
| P-SCIAGENT | Scientific agent reasoning | Scientific agents, weather reasoning, tool-augmented scientific AI |
| P-TS | General time-series forecasting | Temporal Fusion Transformer and applied multi-horizon forecasting |
| P-CAL | Calibration | Neural network calibration, temperature scaling, monitoring-oriented calibration |

## Design Principles

* IDs are curriculum organization tools, not strict disciplinary labels.
* A paper can influence multiple tracks but receives one primary ID.
* Track prefix should reflect the paper's main role in the user's current research plan.
* Do not create new prefixes unless an existing prefix would be misleading.
* Do not rename existing IDs without a migration reason.
* Do not create near-duplicate spatiotemporal graph prefixes when `P-ST` already exists.

## Current Priority Interpretation

* `Tier 0`: foundational mathematical or conceptual prerequisite.
* `Tier 1`: core project paper or canonical model.
* `Tier 2`: important extension, advisor relevance, or second-stage method.
* `Tier 3`: later-stage, frontier, or optional extension.
