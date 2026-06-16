# Terminology

| Term | Working Placeholder | Status |
| ---- | ------------------- | ------ |
| distribution shift | Change between training, validation, test, or deployment distributions. | Candidate |
| temporal shift | Time-dependent distribution change across seasons, periods, or regimes. | Candidate |
| station holdout | Evaluation where some monitoring stations are withheld from training. | Candidate |
| calibration | Agreement between predicted uncertainty and observed outcomes. | Candidate |
| coverage | Fraction of outcomes falling inside predictive intervals or sets. | Candidate |
| coverage gap | Difference between nominal and empirical coverage. | Candidate |
| interval width | Size of the predictive interval. | Candidate |
| sharpness | Concentration of predictive uncertainty, conditional on calibration checks. | Candidate |
| CRPS | Continuous Ranked Probability Score for probabilistic forecasts. | Candidate |
| selective prediction | System predicts only when confidence or risk criteria are met. | Candidate |
| abstention | Decision to withhold a prediction or defer action. | Candidate |
| risk–coverage curve | Relationship between retained coverage and incurred risk. | Candidate |
| AURC | Area under the risk–coverage curve. | Candidate |
| CVaR | Conditional Value at Risk for tail-risk evaluation. | Candidate |
| representation stability | Stability of internal model representations under shift. | Candidate |
| latent shift | Shift observed in learned latent states or embeddings. | Candidate |
| graph Laplacian | In this project, the graph Laplacian defines how station adjacency shapes spatial smoothing or propagation. The chosen form must be documented because it changes the behavior of STGCN-style graph convolution. | Candidate |
| normalized graph Laplacian | The normalized graph Laplacian rescales adjacency by node degree, reducing the influence of highly connected stations. It matters for fair spatial aggregation when station connectivity is uneven. | Candidate |
| graph Fourier transform | The graph Fourier transform represents station signals in the Laplacian eigenbasis. It is mainly a conceptual tool here for understanding spectral graph convolution, not an operation that should be computed in the scalable baseline. | Candidate |
| spectral filtering | Spectral filtering applies a learnable function of the graph Laplacian to station-level signals. In forecasting, it encodes the assumption that useful spatial patterns can be filtered over the station graph. | Candidate |
| non-parametric spectral filter | A non-parametric spectral filter learns one parameter per graph frequency. It is a poor fit for the planned baseline because it is not localized and depends on the graph Fourier basis. | Candidate |
| polynomial graph filter | A polynomial graph filter expresses graph convolution as powers of the Laplacian. This makes the receptive field explicit and ties the model to a chosen hop range on the station graph. | Candidate |
| K-hop localization | K-hop localization means a graph filter only aggregates information within `K` graph hops. In PM2.5 forecasting, `K` is a modeling assumption about spatial dependency range. | Candidate |
| Chebyshev polynomial approximation | Chebyshev polynomial approximation computes polynomial graph filters by recurrence. It supports efficient sparse graph convolution for fixed station graphs without explicit eigendecomposition. | Candidate |
| scaled Laplacian | The scaled Laplacian maps Laplacian eigenvalues into the range needed for stable Chebyshev recurrence. The scaling choice, including maximum eigenvalue handling, should be reproducible in experiments. | Candidate |
| graph coarsening | Graph coarsening builds lower-resolution graphs by clustering nodes. It is less central for the first PM2.5 baseline but matters when considering hierarchical spatial pooling. | Candidate |
| graph pooling | Graph pooling aggregates signals over coarsened or grouped nodes. For a fixed station forecasting baseline, it should not be assumed necessary unless it improves reliability or interpretability. | Candidate |
| graph quality | Graph quality describes whether edges reflect real spatial, meteorological, or statistical dependencies. Poor graph quality can damage error, calibration, and decision metrics even when model code is correct. | Candidate |
| local stationarity on graphs | Local stationarity on graphs assumes that similar filter behavior is useful across graph neighborhoods. This assumption must be stress-tested because PM2.5 dependencies can vary by geography, season, and weather regime. | Candidate |
