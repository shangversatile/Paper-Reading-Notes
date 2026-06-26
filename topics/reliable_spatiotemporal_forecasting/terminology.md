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
| graph smoothness | Graph smoothness measures how similar a graph signal is across connected nodes. It is not ordinary variance; it is local disagreement under the chosen graph structure, often measured by $\mathbf{x}^T L \mathbf{x}$. | Candidate |
| graph frequency | Graph frequency is defined by the eigenvalues of the graph Laplacian. Small eigenvalues correspond to smooth graph modes, while large eigenvalues correspond to rapidly varying modes across graph edges. | Candidate |
| graph Fourier modes | Graph Fourier modes are the eigenvectors of the graph Laplacian. They form an orthonormal basis for graph signals and represent independent patterns of variation induced by the graph structure. | Candidate |
| graph Fourier transform | The graph Fourier transform represents a node-domain signal in the Laplacian eigenbasis. For a graph signal $\mathbf{x}$ and eigenvector matrix $U$, the transform is $\hat{\mathbf{x}}=U^T\mathbf{x}$. | Candidate |
| orthogonal graph Fourier basis | An orthogonal graph Fourier basis is formed by the orthonormal eigenvectors of a symmetric graph Laplacian. In this case, the eigenvector matrix $U$ satisfies $U^{-1}=U^T$, so $U^T\mathbf{x}$ is the graph Fourier transform and $U\hat{\mathbf{x}}$ is the inverse transform. | Candidate |
| spectral filtering | Spectral filtering modifies graph signals by changing their graph-frequency components. In this paper, the basic form is $\mathbf{y}=Ug_\theta(\Lambda)U^T\mathbf{x}$ before later replacing direct spectral filtering with localized polynomial filters. | Candidate |
| filter response | A filter response specifies how strongly a model preserves, suppresses, or amplifies each graph frequency component. In spectral graph filtering, this is represented by $g_\theta(\Lambda)$. | Candidate |
| spectral coefficient | A spectral coefficient measures how much a graph signal aligns with a graph Fourier mode. For a graph signal $\mathbf{x}$, the coefficient vector is $\hat{\mathbf{x}}=U^T\mathbf{x}$. | Candidate |
| frequency usefulness | Frequency usefulness refers to whether a graph frequency component helps the task objective. It is different from frequency magnitude in the input signal. A signal may contain strong high-frequency components, but the model may still learn to suppress them if they behave like noise. | Candidate |
| non-parametric spectral filter | A non-parametric spectral filter assigns an independent parameter to each graph frequency, often written as $g_\theta(\Lambda)=\mathrm{diag}(\theta)$. It is expressive but generally not localized in the node domain and has learning complexity $O(n)$. | Candidate |
| node-space locality | Node-space locality means that the output at a node depends only on nodes within a limited graph-hop neighborhood. For a $K$-localized filter, nodes farther than $K$ hops should not directly affect the output. | Candidate |
| dense node-space operator | A dense node-space operator is a graph filtering matrix whose entries are generally nonzero for many node pairs. In spectral filtering, $U\mathrm{diag}(\theta)U^T$ is usually dense, which means distant nodes may influence one another. | Candidate |
| polynomial graph filter | A polynomial graph filter constrains the spectral filter to be a polynomial of the graph Laplacian, such as $g_\theta(L)=\sum_{k=0}^{K-1}\theta_kL^k$. This reduces parameter complexity and provides graph-hop localization in the node domain. | Candidate |
| graph-hop locality | Graph-hop locality means that a filtering operation only directly uses information from nodes within a bounded number of graph hops. Polynomial filters obtain this property because powers of the Laplacian only propagate information across a bounded graph-hop range. | Candidate |
| K-hop localization | K-hop localization means a graph filter only aggregates information within $K$ graph hops. In PM2.5 forecasting, $K$ is a modeling assumption about spatial dependency range. | Candidate |
| zero-hop self-information | Zero-hop self-information refers to the identity term $L^0\mathbf{x}=I_n\mathbf{x}=\mathbf{x}$ in a polynomial graph filter. It preserves each node's own feature before graph-neighborhood mixing. | Candidate |
| Chebyshev polynomial approximation | Chebyshev polynomial approximation computes polynomial graph filters by recurrence. It supports efficient sparse graph convolution for fixed station graphs without explicit eigendecomposition. | Candidate |
| Chebyshev recurrence | Chebyshev recurrence computes $T_k(\tilde L)\mathbf{x}$ recursively from previous graph-filtered vectors, avoiding explicit construction of high-order matrix powers or the graph Fourier basis. It provides an efficient way to implement localized polynomial graph filters. | Candidate |
| Chebyshev basis | The Chebyshev basis is a stable polynomial basis defined on $[-1,1]$. In graph filtering, it replaces the ordinary monomial basis $I_n,L,L^2,\ldots$ with $T_0(\tilde L),T_1(\tilde L),T_2(\tilde L),\ldots$. | Candidate |
| scaled Laplacian | The scaled Laplacian is $\tilde L=2L/\lambda_{\max}-I_n$. It maps Laplacian eigenvalues into $[-1,1]$ so that Chebyshev polynomials can be applied stably. The scaling choice, including maximum eigenvalue handling, should be reproducible in experiments. | Candidate |
| recursive graph-local feature | A recursive graph-local feature is a vector such as $\bar{\mathbf{x}}_k=T_k(\tilde L)\mathbf{x}$, computed from earlier vectors using sparse Laplacian-vector multiplications. It contains graph information up to $k$ hops. | Candidate |
| graph coarsening | Graph coarsening builds lower-resolution graphs by clustering nodes. It is less central for the first PM2.5 baseline but matters when considering hierarchical spatial pooling. | Candidate |
| graph pooling | Graph pooling aggregates signals over coarsened or grouped nodes. For a fixed station forecasting baseline, it should not be assumed necessary unless it improves reliability or interpretability. | Candidate |
| graph quality | Graph quality describes whether edges reflect real spatial, meteorological, or statistical dependencies. Poor graph quality can damage error, calibration, and decision metrics even when model code is correct. | Candidate |
| locality on graphs | In graph-based forecasting, locality means that a node's prediction can be meaningfully informed by nearby nodes under the chosen graph. Unlike image grids, this locality is not given by the data format itself. It must be justified by graph construction, such as geographical distance, correlation, meteorological similarity, or physical transport. | Candidate |
| stationarity on graphs | In graph convolution, stationarity means that the same local filtering rule is shared across different graph neighborhoods. For PM2.5 forecasting, this is a strong assumption because local pollution dynamics may differ across regions, seasons, terrain, and pollution regimes. | Candidate |
| graph convolution as an inductive bias | A graph convolution layer assumes that the selected graph makes local aggregation meaningful. It is not merely an architecture choice. If the graph does not match the true dependency structure, the model may propagate misleading information and become unreliable under shift. | Candidate |
| graph mismatch | Graph mismatch occurs when the constructed graph does not represent the task-relevant dependency structure. In reliable spatiotemporal forecasting, graph mismatch can affect not only predictive error but also uncertainty calibration, coverage, and decision quality. | Candidate |
