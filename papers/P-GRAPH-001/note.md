# P-GRAPH-001: Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering

## Citation

Michaël Defferrard, Xavier Bresson, and Pierre Vandergheynst.
Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering.
NeurIPS 2016.

## Reading Tier

Tier 0, Tier 1

## Track

Spatiotemporal modeling

## Why This Paper Matters

This paper is a foundation for understanding graph convolution before treating STGCN as a black-box forecasting baseline. It explains how convolution can be defined on an irregular graph by moving from the vertex domain to the spectral domain of the graph Laplacian, then makes that construction practical through localized polynomial filters.

The central contribution is not merely applying neural networks to graphs. The paper shows how to design graph filters that are:

* localized to a bounded graph neighborhood,
* efficient on sparse graphs,
* computable without explicit graph Fourier basis multiplication at every layer,
* parameterized by a small number of learnable coefficients.

For reliable PM2.5 forecasting, this matters because graph construction, Laplacian normalization, filter support size, and graph quality become modeling assumptions. If those assumptions are wrong, reliability failures may come from the graph rather than from the temporal forecasting architecture alone.

## Core Problem

Classical CNNs rely on regular grids. Pixels have a consistent local neighborhood structure, filters can be shared across locations, and translation-like operations are well defined. Graphs do not provide these conveniences. A node's neighbors are unordered, degrees vary across nodes, and there is no canonical spatial translation operator.

The paper addresses this by defining convolution through spectral graph theory. Instead of sliding a kernel over ordered neighborhoods, it filters graph signals through functions of the graph Laplacian. The main technical challenge is then to make this spectral definition localized and computationally scalable.

## Intuition Before the Math

On images, a convolutional filter can be reused because each local patch has the same grid geometry. A 3 by 3 image filter has a clear meaning everywhere: top-left, center, bottom-right, and so on. On a station graph, a node may have three neighbors, another may have twelve, and there is no natural order saying which neighbor is "left" or "right."

This is why a direct spatial generalization of CNN convolution is difficult. The paper avoids the missing neighbor ordering problem by moving to graph frequencies. The graph Laplacian plays the role of a geometry-aware operator: its eigenvectors define smooth-to-oscillatory modes over the graph. Filtering then means changing how much of each graph-frequency component is retained.

This move solves the conceptual definition problem, but not the engineering problem. Direct spectral filtering is expensive and not localized. The paper's key step is to restrict filters to polynomials of the Laplacian and approximate them with Chebyshev polynomials, which brings the operation back to sparse local message propagation.

## Mathematical Setup

Let the graph be:

$$
G=(V,E,W).
$$

Here, $V$ is the node set, $E$ is the edge set, and $W$ is the weighted adjacency matrix. This is the paper's starting point for replacing regular-grid convolution with graph-domain filtering. In a PM2.5 station graph, nodes are monitoring stations and $W_{ij}$ records the chosen connection strength between stations $i$ and $j$.

A graph signal is:

$$
\mathbf{x}\in\mathbb{R}^n.
$$

Each entry $x_i$ is a value attached to one graph node. This lets the paper treat graph data as a signal over vertices. In PM2.5 forecasting, $\mathbf{x}$ could represent pollution measurements across all monitoring stations at a fixed time.

The unnormalized graph Laplacian is:

$$
L=D-W.
$$

Here, $D$ is the degree matrix and $W$ is the weighted adjacency matrix. This formula gives the paper the graph operator that measures local variation. In a PM2.5 station graph, $L$ encodes which station differences are treated as local disagreements under the selected graph.

The normalized graph Laplacian is:

$$
L=I_n-D^{-1/2}WD^{-1/2}.
$$

Here, $I_n$ is the $n$ by $n$ identity matrix, and the degree normalization rescales adjacency by node connectivity. The paper uses the Laplacian because it encodes graph geometry. It penalizes differences between connected nodes, so its eigenvectors describe patterns ranging from smooth over the graph to rapidly varying across edges. For PM2.5, the choice between unnormalized and normalized $L$ changes how highly connected stations influence spatial filtering.

Because the Laplacian is symmetric for an undirected weighted graph, it can be decomposed as:

$$
L=U\Lambda U^T.
$$

Here, $U$ contains the graph Fourier modes and $\Lambda$ contains the corresponding graph frequencies. This eigendecomposition is the bridge from graph geometry to spectral filtering in Section 2.1. For PM2.5, changing the station graph changes $L$, which changes $U$, $\Lambda$, and therefore the meaning of graph frequency.

The graph Fourier transform is:

$$
\hat{\mathbf{x}}=U^T\mathbf{x}.
$$

Here, $\hat{\mathbf{x}}$ contains the coefficients of $\mathbf{x}$ in the Laplacian eigenbasis. The transform lets the paper describe filtering by graph frequency instead of by ordered neighborhoods. For station data, $\hat{x}_k$ says how much the PM2.5 field aligns with graph mode $\mathbf{u}_k$.

The inverse transform is:

$$
\mathbf{x}=U\hat{\mathbf{x}}.
$$

This reconstructs the node-domain signal from graph Fourier coefficients. The role of this transform is analogous to classical Fourier analysis, but the basis is determined by the graph rather than by a regular grid. For PM2.5 forecasting, this means the graph construction determines the frequency geometry used by the spatial layer.

## Spectral Filtering

A graph signal can be filtered by:

$$
\mathbf{y}=g_\theta(L)\mathbf{x}=Ug_\theta(\Lambda)U^T\mathbf{x}.
$$

This equation defines convolution-like filtering on a graph by using the Laplacian spectrum. It transforms the node signal $\mathbf{x}$ into graph-frequency space with $U^T$, modifies frequency components using $g_\theta(\Lambda)$, and reconstructs the filtered signal $\mathbf{y}$ with $U$. The filter is tied to the chosen graph because $L$, $U$, and $\Lambda$ all depend on $W$.

For forecasting, this matters because the graph defines which spatial patterns are considered smooth, local, or high-frequency. A distance graph, a correlation graph, and a meteorological graph can induce different notions of graph frequency. However, P-GRAPH-001 itself solves the graph-domain convolution definition and efficient localized filtering problem; suitability for spatiotemporal forecasting comes from later models that combine spatial graph convolution with temporal modeling.

## Why Naive Spectral Filters Are Not Enough

A direct non-parametric spectral filter can be written as:

$$
g_\theta(\Lambda)=\mathrm{diag}(\theta).
$$

Here, $\theta$ contains one learnable parameter per graph frequency. This formula solves the definitional problem of spectral filtering, but it is not a good scalable graph CNN layer. For PM2.5 station graphs, this would make the filter tightly dependent on the eigensystem of one constructed graph.

The problems are:

* It is not localized in the vertex domain, so a node's output may depend globally on many other nodes.
* It requires $O(n)$ learnable parameters for a graph with $n$ nodes.
* It requires expensive multiplication by the graph Fourier basis $U$.
* It depends tightly on a specific graph eigensystem, which makes the filter less operationally interpretable as local aggregation.

For PM2.5 forecasting, this is undesirable because local support should be an explicit assumption: the modeler should know whether a station aggregates from one-hop, two-hop, or broader graph neighborhoods.

## Polynomial Filters and K-Hop Locality

The paper replaces arbitrary spectral filters with polynomial filters of the Laplacian:

$$
g_\theta(L)=\sum_{k=0}^{K-1}\theta_k L^k.
$$

Here, $\theta_k$ is the learnable coefficient for the $k$-th power of $L$, and $K$ controls the polynomial order. This is the paper's key move from a non-local spectral definition to localized graph filtering. The key fact is that powers of the Laplacian only mix information along graph paths up to a bounded length. A polynomial of order $K$ produces a filter localized within a $K$-hop neighborhood. If two nodes are farther apart than the filter support, the filter does not directly couple them.

This gives $K$ a concrete modeling meaning: $K$ controls graph-neighborhood support. In station graphs, $K$ should not be treated as a generic hyperparameter only. It is a locality assumption about how far spatial information should propagate through the graph. A larger $K$ may capture broader regional structure, but it may also propagate noise or misleading dependencies if the graph edges are poorly specified.

## Chebyshev Approximation

Polynomial filters still need to be computed efficiently. The paper uses Chebyshev polynomial approximation to avoid explicit eigendecomposition and Fourier-basis multiplication.

First, the Laplacian is rescaled:

$$
\tilde{L}=\frac{2}{\lambda_{\max}}L-I_n.
$$

Here, $\lambda_{\max}$ is the largest eigenvalue of $L$, and $\tilde{L}$ is the rescaled Laplacian. This scaling puts the spectrum into a range suitable for Chebyshev recurrence. For PM2.5 station graphs, the scaling choice should be reproducible because it affects the numerical behavior of the graph convolution layer.

The filter is then written as:

$$
g_\theta(L)\mathbf{x}=\sum_{k=0}^{K-1}\theta_k T_k(\tilde{L})\mathbf{x}.
$$

Here, $T_k$ is the Chebyshev polynomial of order $k$. This equation keeps the spectral-filtering interpretation while making the computation possible without explicitly multiplying by $U$. In PM2.5 forecasting, this is what makes sparse station-graph filtering practical.

The scalar recurrence is:

$$
T_0(z)=1,\quad T_1(z)=z,\quad T_k(z)=2zT_{k-1}(z)-T_{k-2}(z).
$$

Here, $z$ is a scalar input to the Chebyshev polynomial. The recurrence lets the paper compute higher-order filter terms from two previous terms. This is the algebraic reason the graph filter can be implemented efficiently.

For graph signals, this becomes:

$$
\bar{\mathbf{x}}_0=\mathbf{x},\quad
\bar{\mathbf{x}}_1=\tilde{L}\mathbf{x},\quad
\bar{\mathbf{x}}_k=2\tilde{L}\bar{\mathbf{x}}_{k-1}-\bar{\mathbf{x}}_{k-2}.
$$

Here, $\bar{\mathbf{x}}_k$ is the $k$-th recursively computed graph-filtered signal. The important implementation consequence is that each step only needs sparse matrix-vector multiplication by the scaled Laplacian. No explicit graph Fourier transform is needed during filtering. For station graphs, this ties runtime to sparse edges rather than dense spectral basis operations.

The resulting complexity is:

$$
O(K|E|).
$$

Here, $K$ is the filter support size and $|E|$ is the number of graph edges. This is efficient when the graph is sparse. If the station graph is dense or poorly sparsified, this computational advantage weakens and the locality assumption becomes harder to interpret.

## Multi-Feature Graph Convolutional Layer

### 1. From One Graph Signal to Multiple Feature Maps

The previous filtering discussion considered a single graph signal:

$$
\mathbf{y}=g_\theta(L)\mathbf{x},
$$

where:

$$
\mathbf{x}\in\mathbb{R}^n
$$

and:

$$
\mathbf{y}\in\mathbb{R}^n.
$$

This means one input feature value per node is transformed into one output feature value per node. The node set is unchanged; only the value attached to each node is filtered.

A neural network layer usually has multiple input feature maps and multiple output feature maps. For sample $s$, let:

$$
\mathbf{x}_{s,i}\in\mathbb{R}^n
$$

be the $i$-th input feature map, where:

$$
i=1,\ldots,F_{\mathrm{in}}.
$$

The input feature matrix for sample $s$ can therefore be written as:

$$
\mathbf{X}_s=
\left[
\mathbf{x}_{s,1},
\mathbf{x}_{s,2},
\ldots,
\mathbf{x}_{s,F_{\mathrm{in}}}
\right]
\in\mathbb{R}^{n\times F_{\mathrm{in}}}.
$$

Formula (5) in the paper defines the $j$-th output feature map as:

$$
\mathbf{y}_{s,j}
=
\sum_{i=1}^{F_{\mathrm{in}}}
g_{\theta_{i,j}}(L)\mathbf{x}_{s,i},
$$

where:

$$
j=1,\ldots,F_{\mathrm{out}}.
$$

Thus, each output channel is obtained by filtering all input channels and summing their contributions.

### 2. Chebyshev Expansion of Formula (5)

Using Chebyshev filters, each input-output channel filter is expanded as:

$$
g_{\theta_{i,j}}(L)\mathbf{x}_{s,i}
=
\sum_{k=0}^{K-1}
\theta_{i,j,k}T_k(\tilde L)\mathbf{x}_{s,i}.
$$

Substituting this expansion into formula (5) gives:

$$
\mathbf{y}_{s,j}
=
\sum_{i=1}^{F_{\mathrm{in}}}
\sum_{k=0}^{K-1}
\theta_{i,j,k}T_k(\tilde L)\mathbf{x}_{s,i}.
$$

The indices have the following meanings:

* $i$: input feature channel index;
* $j$: output feature channel index;
* $k$: Chebyshev basis or polynomial order index.

The learnable parameter $\theta_{i,j,k}$ specifies how much the $k$-th Chebyshev-filtered version of input channel $i$ contributes to output channel $j$.

For one fixed pair $(i,j)$, the filter has $K$ Chebyshev coefficients:

$$
\theta_{i,j,0},\theta_{i,j,1},\ldots,\theta_{i,j,K-1}.
$$

There are $F_{\mathrm{in}}F_{\mathrm{out}}$ input-output channel pairs. Therefore, the number of learnable parameters in this graph convolutional layer is:

$$
K F_{\mathrm{in}}F_{\mathrm{out}}.
$$

### 3. Why the Layer Sums Over Input Channels

The summation over input channels is analogous to standard multi-channel CNN convolution. In a CNN, one output channel is produced by applying kernels to all input channels and summing the results. Similarly, in graph convolution, one output graph feature map is produced by graph-filtering all input graph feature maps and summing their contributions.

The summation is a linear algebra operation, not a probabilistic assumption. It does not assume that input channels are statistically independent. The input channels may be correlated, redundant, causally related, or jointly shifted. Formula (5) only says that the layer forms a learned linear combination of filtered input features.

The component-level expression makes this clearer. For node $v$:

$$
y_{s,j}(v)
=
\sum_{i=1}^{F_{\mathrm{in}}}
\sum_{k=0}^{K-1}
\theta_{i,j,k}
\left[T_k(\tilde L)\mathbf{x}_{s,i}\right]_v.
$$

This is a sum of filtered feature values at the same output node coordinate $v$. The layer learns how much each input channel and each Chebyshev order should contribute through the parameters $\theta_{i,j,k}$.

However, within one linear graph convolutional layer, cross-channel interaction is linear before any nonlinear activation. More complex interactions among variables require nonlinear activations, deeper layers, temporal modules, attention, gating, or explicitly designed interaction features.

### 4. Output Channels Are Not Independent Random Variables

The output channels:

$$
\mathbf{y}_{s,1},\mathbf{y}_{s,2},\ldots,\mathbf{y}_{s,F_{\mathrm{out}}}
$$

are different learned feature maps. They are not assumed to be statistically independent random variables.

Each output channel has its own set of parameters:

$$
\theta_{i,j,k}.
$$

Therefore, different output channels can learn different graph-structured feature detectors. For example, one output channel may emphasize smooth low-frequency station patterns, while another may emphasize sharper local contrasts, depending on the learned Chebyshev coefficients.

However, the channels are jointly trained under the same loss and can be mixed again by later layers. A better interpretation is that output channels are learned coordinates of a hidden node representation, not independent factors.

### 5. Why Each Output Channel Is Still in $\mathbb{R}^n$

For a fixed sample $s$ and output channel $j$:

$$
\mathbf{y}_{s,j}\in\mathbb{R}^n.
$$

This is because graph convolution changes node features, not the node set. For each input channel:

$$
\mathbf{x}_{s,i}\in\mathbb{R}^n.
$$

Since $T_k(\tilde L)$ is an $n$ by $n$ graph operator:

$$
T_k(\tilde L)\mathbf{x}_{s,i}\in\mathbb{R}^n.
$$

Multiplying by $\theta_{i,j,k}$ keeps the result in $\mathbb{R}^n$, and summing vectors over $i$ and $k$ still gives a vector in $\mathbb{R}^n$. Therefore:

$$
\mathbf{y}_{s,j}
=
\sum_{i=1}^{F_{\mathrm{in}}}
\sum_{k=0}^{K-1}
\theta_{i,j,k}T_k(\tilde L)\mathbf{x}_{s,i}
\in\mathbb{R}^n.
$$

The full layer maps:

$$
\mathbf{X}_s\in\mathbb{R}^{n\times F_{\mathrm{in}}}
$$

to:

$$
\mathbf{Y}_s\in\mathbb{R}^{n\times F_{\mathrm{out}}}.
$$

The number of nodes $n$ remains the same. What changes is the feature dimension attached to each node.

### 6. PM2.5 Forecasting Interpretation

For PM2.5 forecasting, input channels may include:

* PM2.5 concentration;
* temperature;
* humidity;
* wind speed;
* wind direction encoding;
* pressure;
* other pollutant variables;
* historical lag features.

Each input channel is a graph signal over monitoring stations. The graph convolutional layer applies localized graph filters to each variable and combines them into hidden node features.

For example, one output channel may combine:

* local PM2.5 spatial patterns;
* wind-related spatial transport patterns;
* humidity-related modulation patterns;
* station-level anomaly patterns.

These meanings are not guaranteed by the architecture, but they provide possible interpretations to test through experiments.

For STGCN-style models, this is the spatial part of the architecture: graph convolution mixes information across stations, while temporal convolution handles time dynamics. Understanding this separation is important because a forecasting failure may come from temporal modeling, graph construction, or their interaction.

### 7. Reliability Implications

Formula (5) reveals several reliability concerns.

First, all input channels are filtered through the same graph Laplacian. If the graph construction is wrong, then every variable is propagated through a potentially wrong neighborhood structure.

Second, different variables may have different spatial mechanisms. A graph that is suitable for PM2.5 concentration may not be equally suitable for wind, humidity, or temperature.

Third, if some input channels are unstable under distribution shift, missingness, or sensor noise, their graph-filtered representations may contaminate the output channels.

Therefore, reliable spatiotemporal forecasting should not only evaluate final prediction error. It should also examine:

* sensitivity to graph construction;
* sensitivity to the support size $K$;
* channel contribution under shift;
* calibration and coverage degradation;
* downstream decision cost;
* whether graph-filtered variables remain meaningful under missingness and noise.

The key research implication is:

A graph convolutional layer does not merely propagate PM2.5. It propagates and mixes all input variables under the same graph-defined geometry. The reliability of this operation depends on whether that graph geometry is valid for the variables and conditions being modeled.

## Graph Coarsening and Pooling

The paper also proposes graph coarsening and pooling through multilevel clustering. It uses Graclus coarsening and then rearranges nodes into a balanced binary-tree-like ordering so that pooling can be implemented efficiently.

For my current PM2.5 forecasting baseline, this is secondary. A fixed station graph usually has a stable set of monitoring locations, and the first STGCN-style implementation should focus on graph convolution plus temporal modeling rather than hierarchical pooling. Still, this section is useful because it shows that graph CNN design depends not only on filters but also on how graph resolution is defined.

## Experimental Evidence

The paper evaluates the method on MNIST and 20NEWS.

The main lessons for this project are:

* Chebyshev filters make spectral graph convolution computationally practical on sparse graphs.
* On MNIST, graph CNN performance can approach classical CNN performance when the graph reflects the grid structure well.
* On text classification, graph construction choices affect performance.
* Graph quality is not a minor preprocessing issue; it is a condition for the model's usefulness.

The experiments support the method as a graph convolution foundation, but they do not establish reliability for environmental forecasting. They do not test temporal shift, calibration, uncertainty intervals, station holdout, missingness, or downstream decision cost.

## Key Assumptions

The method depends on several assumptions that must be made explicit before using STGCN for PM2.5 forecasting:

* The data domain can be meaningfully represented as a graph.
* The graph is sparse enough for Chebyshev filtering to be efficient.
* Nearby nodes on the graph tend to have statistically meaningful relationships.
* The same type of local filter is useful across different graph neighborhoods.
* The chosen filter support $K$ matches the relevant spatial dependency range.
* The graph is fixed or stable enough for the learned filters to remain meaningful.

These assumptions are not guaranteed for air-quality stations. Edges based on distance, historical correlation, meteorology, wind direction, or domain knowledge can encode different causal and statistical stories.

## Limitations

This paper does not directly solve the reliable forecasting problem.

Important limitations:

* The experiments are classification tasks, not forecasting tasks.
* The graph formulation is mainly undirected, while pollutant transport can be directional.
* Spectral filters are isotropic: they do not naturally distinguish direction-specific influence.
* Graph quality is crucial, but the paper does not provide a rule for constructing PM2.5 station graphs.
* The paper does not address uncertainty quantification.
* The paper does not address calibration.
* The paper does not analyze distribution shift.
* The paper does not evaluate downstream decision reliability.
* The paper does not test station holdout, temporal regime shifts, extreme events, or missing sensor patterns.

These are not flaws in the paper's objective. They define the boundary between graph convolution as a modeling primitive and reliable spatiotemporal forecasting as a research problem.

## Connection to Reliable Spatiotemporal Forecasting

This paper supports the graph convolution foundation of the flagship project.

Concrete implications:

* STGCN should be understood as building on polynomial graph filtering, not as an opaque baseline.
* The graph builder must be treated as a research decision.
* The Laplacian type and normalization must be documented.
* The Chebyshev order $K$ should be reported as a locality assumption.
* Graph quality should be stress-tested because failures may come from graph mismatch.
* PM2.5 graph choices should be compared using reliability metrics, not only MAE or RMSE.

The paper also clarifies what the project must add: uncertainty estimation, calibration checks, shift-aware evaluation, and decision-oriented reliability metrics.

## Research-Level Critique

The paper solves computational and locality problems for spectral graph convolution. It does not solve reliability. This distinction matters because an efficient localized graph filter can still produce unreliable forecasts if the graph does not represent the relationships that matter for the target task.

The graph-quality result is especially important. It implies that graph construction is part of model validity, not a neutral preprocessing step. In PM2.5 forecasting, an incorrect graph may create failure modes even when the neural architecture is implemented correctly: the model may propagate information between stations that are geographically close but meteorologically disconnected, or fail to propagate information along wind-driven pathways.

Therefore, graph perturbation and graph construction comparison should become later stress-test dimensions. A reliable model should not only perform well on one chosen graph; its error, coverage, and decision cost should be examined under plausible alternative graphs and controlled graph corruption.

## My Current Understanding and Open Confusions

### 1. Reinterpreting CNN Assumptions on Graph-Structured Spatiotemporal Data

A key lesson from this paper is that CNN assumptions do not transfer automatically from images to graph-structured spatiotemporal data.

For images, locality is almost built into the data representation. A pixel grid gives each pixel a stable and meaningful neighborhood. In spatiotemporal graph data, however, locality must be justified by the graph construction. For PM2.5 forecasting, an edge may represent geographical distance, historical correlation, meteorological similarity, wind-driven transport, domain knowledge, or a hybrid relation. Therefore, graph locality is not a free assumption; it is a modeling decision that requires evidence.

For images, stationarity means that similar local visual patterns, such as edges, textures, colors, or strokes, can be recognized across different spatial positions using shared filters. In spatiotemporal forecasting, this sharing assumption is less obvious. The same local pattern may not have the same meaning across different cities, seasons, terrain types, or pollution regimes. A shared graph filter may therefore impose a stronger inductive bias than expected.

For images, compositionality is also relatively natural: edges combine into shapes, shapes combine into object parts, and object parts combine into objects. For spatiotemporal graphs, the relationship between local and global structure is more fragile. Local station interactions may or may not compose into meaningful regional pollution dynamics. The boundary of the learned rule must therefore be checked, especially under distribution shift.

This changes how I should read graph convolution. A graph convolution layer is not merely a neural-network operation. It is a claim that the chosen graph makes locality, parameter sharing, and hierarchical composition meaningful for the target task.

### 2. What Spectral Convolution Is Trying to Solve

My current confusion is mainly about spectral convolution. The key clarification is that spectral convolution is not introduced because the paper directly targets spatiotemporal forecasting. Instead, it is introduced because convolution is hard to define on irregular graph domains.

In images, a convolution kernel can slide over a regular grid. The local neighborhood has a stable shape and ordering. On a graph, neighborhoods have variable sizes and no canonical ordering. There is no unique translation operator in the vertex domain.

The spectral approach avoids defining convolution by spatial translation. It uses the graph Laplacian to define graph Fourier modes. These modes provide a frequency-like coordinate system determined by the graph structure itself. A graph signal can then be filtered by transforming it into graph frequency space, modifying its frequency components, and transforming it back.

The important intuition is that the graph Laplacian defines what smoothness means on a graph, and spectral filtering modifies graph signals according to graph-defined frequencies.

Low-frequency graph signals vary smoothly across connected nodes. High-frequency graph signals vary sharply across connected nodes. For PM2.5 forecasting, this means that the meaning of "smooth" or "rough" depends entirely on how the station graph is constructed.

### 3. Why Polynomial Spectral Filters Matter

A naive spectral filter is not enough because it can be non-local and computationally expensive. The paper's key move is to restrict filters to polynomials of the graph Laplacian.

A polynomial filter such as:

$$
g_\theta(L)=\sum_{k=0}^{K-1}\theta_k L^k
$$

has a clear graph interpretation. Here, $L$ is the graph Laplacian, $\theta_k$ is the coefficient for the $k$-th power of $L$, and $K$ controls the filter order. Since $L$ propagates information across graph edges, powers of $L$ correspond to repeated neighborhood propagation. A $K$-order polynomial filter is therefore localized within a $K$-hop neighborhood.

This makes $K$ more than a numerical hyperparameter. In a station graph, $K$ encodes an assumption about how far spatial information should propagate through the graph. If $K$ is too small, the model may miss long-range pollution transport. If $K$ is too large, the model may mix unrelated stations and amplify noisy or misleading dependencies.

### 4. Why Chebyshev Approximation Matters

The Chebyshev approximation makes the spectral idea computationally usable. Direct spectral filtering requires the graph Fourier basis, which depends on eigendecomposition of the graph Laplacian. This is expensive and does not scale well.

Chebyshev recurrence allows the filter to be computed through repeated sparse matrix-vector multiplications. This avoids explicitly computing the Fourier basis and gives complexity approximately proportional to:

$$
O(K|E|)
$$

where $K$ is the filter support size and $|E|$ is the number of graph edges.

This is why the paper is important for later STGCN-style models: it turns spectral graph convolution from an elegant definition into a practical layer.

### 5. How This Should Influence My PM2.5 Project

This paper does not prove that spectral graph convolution is automatically suitable for PM2.5 forecasting. Instead, it tells me what must be justified before using graph convolution responsibly.

For my project, the key questions become:

1. What does an edge in the PM2.5 station graph mean?
2. Does the graph make PM2.5 signals locally smooth in a meaningful way?
3. Are local pollution patterns shared across different regions and regimes?
4. Does the chosen graph remain valid under temporal shift, missingness, spikes, or extreme events?
5. Does graph construction affect not only MAE/RMSE, but also calibration, coverage, and decision cost?
6. Can graph mismatch become a source of reliability failure?

The most important research implication is that graph construction should be treated as part of reliability evaluation. If the graph is wrong, the graph convolution layer may propagate misleading information even when the architecture is implemented correctly.

A later stress test should compare different graph constructions, such as distance-based graphs, correlation-based graphs, meteorology-informed graphs, and perturbed or randomized graphs. The comparison should not only report prediction error, but also uncertainty calibration, coverage degradation, and downstream decision cost.

## Mathematical Deepening: Laplacian, Smoothness, and Graph Fourier Space

### 1. What the Paper Is Doing in Section 2.1

Section 2.1 of the paper moves from the difficulty of defining convolution in the vertex domain to a spectral graph formulation. The key chain is:

1. Define a graph signal on an undirected weighted graph.
2. Define the graph Laplacian.
3. Use the eigendecomposition of the Laplacian to define graph Fourier modes.
4. Define spectral filtering in the graph Fourier domain.
5. Replace naive spectral filters with localized polynomial filters and Chebyshev recurrence.

This means that the Laplacian is not introduced as a decorative matrix. It is the mathematical object that defines what variation, smoothness, and frequency mean on the graph. P-GRAPH-001 solves the graph-domain convolution definition and efficient localized filtering problem. Its relevance to spatiotemporal forecasting comes later, when spatial graph convolution is combined with temporal modeling.

### 2. Graph Weights, Units, and the Meaning of $W$

The paper defines a weighted graph $G=(V,E,W)$, where $W$ is the weighted adjacency matrix. For my project, it is important not to confuse raw physical distance with edge weight.

A raw distance $d_{ij}$ is usually not used directly as $W_{ij}$. Instead, distance or other relational evidence is transformed into a similarity or influence weight, for example:

$$
W_{ij}=\exp\left(-\frac{d_{ij}^2}{\sigma^2}\right).
$$

This formula turns the physical distance $d_{ij}$ into a connection strength $W_{ij}$, with $\sigma$ controlling the distance scale. In the paper's logic, $W$ defines the graph on which the Laplacian is built. For a PM2.5 station graph, this means $W_{ij}$ should be interpreted as a normalized or dimensionless connection strength, not necessarily as a physical distance.

This matters because the graph Laplacian mixes edge weights with node signals:

$$
(L\mathbf{x})_i=\sum_j W_{ij}(x_i-x_j).
$$

Here, $\mathbf{x}$ is the graph signal, $x_i-x_j$ is the signal difference between stations $i$ and $j$, and $W_{ij}$ controls how much that difference matters under the chosen graph structure. In PM2.5 forecasting, graph construction is already a modeling assumption. A distance-based graph, a correlation-based graph, a meteorology-informed graph, and a wind-informed graph define different meanings of local inconsistency.

### 3. Why $L\mathbf{x}$ Measures Local Inconsistency

For the unnormalized graph Laplacian,

$$
L=D-W,
$$

where

$$
D_{ii}=\sum_j W_{ij}.
$$

The first formula defines the Laplacian as degree minus adjacency, and the second formula defines the weighted degree of node $i$. In Section 2.1, these objects define the graph difference operator used to build spectral graph convolution. For PM2.5 station graphs, $D_{ii}$ measures the total graph connection strength around station $i$.

For a graph signal $\mathbf{x}\in\mathbb{R}^n$, the $i$-th component of $L\mathbf{x}$ is:

$$
(L\mathbf{x})_i=(D\mathbf{x})_i-(W\mathbf{x})_i.
$$

This equation separates the self-weighted degree term from the neighbor aggregation term. In the paper's logic, it shows why $L$ acts like a graph difference operator. For PM2.5, it compares station $i$ with the weighted station neighborhood defined by $W$.

Since

$$
(D\mathbf{x})_i=D_{ii}x_i=\left(\sum_j W_{ij}\right)x_i
$$

and

$$
(W\mathbf{x})_i=\sum_j W_{ij}x_j,
$$

we obtain:

$$
(L\mathbf{x})_i
=\sum_j W_{ij}x_i-\sum_j W_{ij}x_j
=\sum_j W_{ij}(x_i-x_j).
$$

This derivation shows that $L\mathbf{x}$ measures the weighted disagreement between each node and its neighbors. If node $i$ has a similar value to its strongly connected neighbors, then $(L\mathbf{x})_i$ is small. If node $i$ differs sharply from strongly connected neighbors, then $(L\mathbf{x})_i$ is large. In PM2.5 forecasting, $L\mathbf{x}$ measures how inconsistent a station's pollution value is relative to its graph-defined neighborhood, but this interpretation is valid only if the graph-defined neighborhood is meaningful.

### 4. Why $\mathbf{x}^T L \mathbf{x}$ Measures Graph Smoothness

The quadratic form of the unnormalized Laplacian is:

$$
\mathbf{x}^T L \mathbf{x}
=\frac{1}{2}\sum_{i,j}W_{ij}(x_i-x_j)^2.
$$

This formula turns local edge disagreements into one global graph roughness score. In the paper's logic, it explains why the Laplacian eigenvalues can be interpreted as graph frequencies: high-energy patterns vary strongly across edges. For PM2.5, the quantity is large when strongly connected stations have sharply different pollution values.

This quantity is not ordinary statistical variance. Variance compares values to a global mean. Graph smoothness compares values across connected nodes.

| Concept | What It Compares | Meaning |
| ------- | ---------------- | ------- |
| Variance | $x_i$ against the global mean | Global dispersion |
| Graph smoothness energy | $x_i$ against graph neighbors $x_j$ | Local disagreement under graph structure |

The square has three roles:

1. It prevents positive and negative differences from canceling.
2. It penalizes large local disagreements more strongly.
3. It creates an energy function that measures how rough the signal is on the graph.

Thus, a signal is graph-smooth when strongly connected nodes have similar values. A signal is graph-rough when strongly connected nodes have sharply different values. For PM2.5, smoothness is not simply low variance. A pollution field may vary globally across regions but still be locally smooth if neighboring stations change gradually. Conversely, a local spike at one station may create high graph energy even if the global variance is not large.

### 5. What the Eigenspace of $L$ Means

Because the graph Laplacian is symmetric positive semidefinite, it can be decomposed as:

$$
L=U\Lambda U^T.
$$

Here, $U$ is the eigenvector matrix and $\Lambda$ is the diagonal eigenvalue matrix. This is the spectral foundation used by the paper to define graph Fourier analysis. For PM2.5 station graphs, changing $W$ changes $L$, and therefore changes the eigenspace used by graph filtering.

The eigenvector matrix is:

$$
U=[\mathbf{u}_1,\mathbf{u}_2,\ldots,\mathbf{u}_n],
$$

and the eigenvalue matrix is:

$$
\Lambda=\mathrm{diag}(\lambda_1,\lambda_2,\ldots,\lambda_n).
$$

Here, each $\mathbf{u}_k$ is an orthonormal eigenvector and each $\lambda_k$ is its corresponding eigenvalue. These symbols define the graph Fourier coordinate system in Section 2.1. In PM2.5 forecasting, each $\mathbf{u}_k$ is a station-level variation pattern induced by the chosen graph.

Each eigenvector satisfies:

$$
L\mathbf{u}_k=\lambda_k\mathbf{u}_k.
$$

This means that $\mathbf{u}_k$ is a stable mode of variation under the graph Laplacian: applying the graph difference operator $L$ does not change the direction of the mode; it only scales it by $\lambda_k$. The eigenspace of $L$ can therefore be understood as the space of graph-induced variation patterns. Each eigenvector is an independent pattern of variation over the nodes, and the eigenvalue measures how rough or smooth that pattern is with respect to the graph.

For a unit-norm eigenvector $\mathbf{u}_k$:

$$
\mathbf{u}_k^T L \mathbf{u}_k=\lambda_k.
$$

This formula connects eigenvalues directly to graph smoothness energy. Small $\lambda_k$ means the eigenvector varies smoothly across graph edges, while large $\lambda_k$ means the eigenvector changes sharply across graph edges. This is the key reason why the paper treats Laplacian eigenvectors as graph Fourier modes.

### 6. Why $U$ Defines the Graph Fourier Basis

In classical Fourier analysis, sine and cosine functions form frequency bases because they are eigenfunctions of standard differential operators on regular domains. On graphs, there is no regular coordinate axis and no natural translation operator. The graph Laplacian replaces the differential operator.

Therefore, the eigenvectors of $L$ play the role of Fourier modes:

| Classical Signal | Graph Signal |
| ---------------- | ------------ |
| Sine and cosine basis | Laplacian eigenvectors |
| Frequency | Laplacian eigenvalue |
| Smooth low-frequency waves | Eigenvectors with small eigenvalues |
| Oscillatory high-frequency waves | Eigenvectors with large eigenvalues |

A graph signal $\mathbf{x}$ can be represented in the Laplacian eigenbasis:

$$
\hat{\mathbf{x}}=U^T\mathbf{x}.
$$

Here, $\hat{\mathbf{x}}$ is the graph Fourier transform of $\mathbf{x}$, and $\hat{x}_k$ is the coefficient of $\mathbf{x}$ along graph mode $\mathbf{u}_k$. In the paper's logic, this representation makes spectral filtering possible. For PM2.5, it describes which graph-induced spatial variation modes compose a station-level pollution field.

The inverse transform is:

$$
\mathbf{x}=U\hat{\mathbf{x}}.
$$

This formula reconstructs the node-domain signal from its graph Fourier coefficients. Thus, node space answers "where is the signal value located?", while graph spectral space answers "which graph variation modes compose this signal?"

### 7. Smoothness Decomposition in the Graph Fourier Basis

If

$$
\mathbf{x}=U\hat{\mathbf{x}},
$$

then:

$$
\mathbf{x}^T L \mathbf{x}
=(U\hat{\mathbf{x}})^T L (U\hat{\mathbf{x}}).
$$

These equations substitute the graph Fourier expansion into the Laplacian quadratic form. In the paper's logic, this is the bridge between graph smoothness and graph frequency. For PM2.5, it explains how roughness of a station pollution field can be decomposed by graph-induced spatial modes.

Using

$$
L=U\Lambda U^T
$$

and

$$
U^TU=I_n,
$$

we get:

$$
\mathbf{x}^T L \mathbf{x}
=\hat{\mathbf{x}}^T\Lambda\hat{\mathbf{x}}
=\sum_{k=1}^n \lambda_k \hat{x}_k^2.
$$

This equation is the rigorous bridge between smoothness and frequency. It says that the total graph roughness of $\mathbf{x}$ is the weighted sum of its spectral coefficients, where the weights are Laplacian eigenvalues. Energy placed on large $\lambda_k$ modes contributes more to roughness. Energy placed on small $\lambda_k$ modes contributes less. This is why the eigenvalues can be interpreted as graph frequencies.

### 8. My Current Interpretation

My current understanding is:

The eigenspace of $L$ is a graph-structured constraint space. It partitions possible graph signals according to how violently or smoothly they vary under the graph-defined neighborhood relation. In this sense, frequency on a graph is not an external time or spatial frequency. It is a measure of variation induced by the graph itself.

Equivalently:

$$
L=U\Lambda U^T
$$

means that the graph Laplacian can be diagonalized in a coordinate system whose axes are graph variation modes. In that coordinate system, $L$ does not mix modes; it only scales each mode by its graph frequency $\lambda_k$. For PM2.5, this means graph frequency is not temporal frequency; it is a property of the station graph and the signal variation across that graph.

This is why spectral filtering can be defined as:

$$
\mathbf{y}=Ug_\theta(\Lambda)U^T\mathbf{x}.
$$

The operation first decomposes $\mathbf{x}$ into graph frequency components, modifies each component according to a learnable frequency response $g_\theta$, and then reconstructs the filtered signal in node space. In the paper's logic, this defines graph-domain filtering before the paper later makes it localized and efficient with polynomial and Chebyshev filters.

### 9. Research Implications for Reliable PM2.5 Forecasting

This interpretation changes how I should think about graph convolution in PM2.5 forecasting.

The graph does not merely connect stations. It defines the frequency geometry of the entire spatial graph model. If the graph is distance-based, then smoothness means nearby stations should have similar pollution values. If the graph is correlation-based, smoothness means historically correlated stations should behave similarly. If the graph is wind-informed, smoothness may encode directional transport or meteorological dependency.

Therefore, graph frequency is graph-dependent.

If the graph is poorly constructed, then:

1. $L\mathbf{x}$ measures disagreement against the wrong neighbors.
2. $\mathbf{x}^T L\mathbf{x}$ gives a misleading smoothness score.
3. The eigenvectors $U$ define misleading graph Fourier modes.
4. Spectral filtering may suppress or amplify the wrong components.
5. A graph convolution layer may propagate misleading information.
6. Reliability failures may appear not only in MAE/RMSE, but also in calibration, coverage, and downstream decision cost.

This gives a concrete research direction:

Graph construction should be evaluated as part of reliability analysis. A reliable spatiotemporal forecasting project should compare different graph definitions and test whether graph mismatch changes not only predictive accuracy, but also uncertainty quality and decision reliability.

## Clarifying the Meaning of $g_\theta(\Lambda)$ and Graph Filtering

### 1. Frequency Positions, Spectral Coefficients, and Filter Response

A key clarification is that $\Lambda$, $U^T\mathbf{x}$, and $g_\theta(\Lambda)$ play different roles.

The eigenvalue matrix

$$
\Lambda=\mathrm{diag}(\lambda_1,\lambda_2,\ldots,\lambda_n)
$$

defines the graph frequency positions. These frequencies are determined by the graph Laplacian and therefore by the graph structure. In the logic of Section 2.1, $\Lambda$ comes from the eigendecomposition of $L$ and fixes the graph-frequency coordinate system.

The graph Fourier transform

$$
\hat{\mathbf{x}}=U^T\mathbf{x}
$$

gives the spectral coefficients of the input signal. The coefficient $\hat{x}_k$ tells how much of the input signal lies along the $k$-th graph Fourier mode. For a PM2.5 station graph, this describes how much the current pollution field aligns with each graph-induced spatial variation pattern.

The filter response

$$
g_\theta(\Lambda)
$$

does not define the frequencies themselves. Instead, it defines how strongly the model keeps, suppresses, or amplifies each frequency component. This is the learnable part of the spectral filtering step, not the graph frequency axis itself.

The basic spectral filtering operation is:

$$
\mathbf{y}=Ug_\theta(\Lambda)U^T\mathbf{x}.
$$

Equivalently, in the graph Fourier domain:

$$
\hat{\mathbf{y}}=g_\theta(\Lambda)\hat{\mathbf{x}}.
$$

For each graph frequency, this means:

$$
\hat{y}_k=g_\theta(\lambda_k)\hat{x}_k.
$$

Thus:

* $\lambda_k$ is the position of the $k$-th graph frequency.
* $\hat{x}_k$ is the amount of the input signal at that frequency.
* $g_\theta(\lambda_k)$ is the learned response to that frequency.
* $\hat{y}_k$ is the filtered spectral coefficient.

This distinction prevents an important misunderstanding: a signal having strong high-frequency components does not automatically mean high frequencies are useful for the task.

### 2. What the Filter Learns

The filter learns a task-dependent frequency response. It does not simply detect whether the input has low, middle, or high frequency. Instead, through training, it learns which frequency components help reduce the loss.

If a certain frequency band consistently helps the task, the learned response may preserve or amplify it. If a frequency band behaves like noise, instability, or irrelevant variation, the learned response may suppress it.

In this sense, the filter is analogous to a CNN kernel. A CNN kernel does not merely observe that an image has edges or textures. It learns which local patterns are useful for the task. Similarly, a graph spectral filter learns which graph variation modes are useful for the task.

### 3. High-Frequency Content Is Not the Same as High-Frequency Usefulness

For a graph signal $\mathbf{x}$, large high-frequency coefficients in $\hat{\mathbf{x}}$ mean that the signal varies sharply across graph edges. In PM2.5 forecasting, this may correspond to local pollution spikes, sensor noise, spatial discontinuities, or graph mismatch.

However, whether these high-frequency components are useful depends on the task and the training objective.

If high-frequency components mostly represent noise, the learned filter should suppress them:

$$
g_\theta(\lambda_k)\approx 0
$$

for large $\lambda_k$.

If high-frequency components represent meaningful local anomalies or sharp pollution boundaries, the learned filter may preserve or amplify them:

$$
g_\theta(\lambda_k)>1
$$

for some high-frequency modes.

Therefore, high-frequency content in the input is a property of the signal. High-frequency usefulness is a property learned from data and loss.

### 4. Non-Parametric Filter as a Fully Free Frequency Response

The paper first introduces a non-parametric spectral filter:

$$
g_\theta(\Lambda)=\mathrm{diag}(\theta),
$$

where

$$
\theta\in\mathbb{R}^n.
$$

This means each graph frequency receives its own independent parameter:

$$
\hat{y}_k=\theta_k\hat{x}_k.
$$

This form is highly expressive because every frequency can be controlled separately. However, the paper points out two major problems:

1. It is not localized in the vertex domain.
2. Its learning complexity is $O(n)$.

This is why the paper later introduces polynomial parametrization. The goal is not merely to learn arbitrary frequency weights, but to learn graph filters that are localized, efficient, and CNN-like.

### 5. PM2.5 Research Interpretation

For PM2.5 station graphs, this distinction is important.

The graph Laplacian defines what counts as low-frequency or high-frequency variation. The input signal determines how much pollution variation lies in each graph frequency. The learned filter determines which of these variation modes are useful for forming predictive node features.

Low-frequency modes may correspond to broad regional pollution trends. Middle-frequency modes may correspond to city-cluster-level spatial structure. High-frequency modes may correspond to local spikes, anomalies, sensor noise, or graph mismatch.

However, these interpretations are not guaranteed by the architecture. They depend on whether the graph construction is meaningful. If the graph is wrong, then the frequency basis is wrong, and the learned filter may preserve or suppress misleading graph modes.

For reliable PM2.5 forecasting, this implies that graph construction and learned frequency response should be evaluated together. A model may appear accurate under standard error metrics while relying on unstable or misleading graph-frequency patterns that fail under distribution shift.

## Why Non-Parametric Spectral Filters Are Not Localized

### 1. Frequency-Domain Simplicity Does Not Imply Node-Space Locality

The paper first introduces the non-parametric spectral filter:

$$
g_\theta(\Lambda)=\mathrm{diag}(\theta),
$$

where

$$
\theta\in\mathbb{R}^n.
$$

In the graph Fourier domain, this looks simple:

$$
\hat{y}_k=\theta_k\hat{x}_k.
$$

This means that each graph frequency component receives an independent learned scaling coefficient.

However, locality is not judged in the graph Fourier domain. Locality is judged in the node domain. The node-space linear operator is:

$$
A=g_\theta(L)=U\mathrm{diag}(\theta)U^T.
$$

Although $g_\theta(\Lambda)$ is diagonal in the spectral basis, $A$ is generally dense in the node basis. This is why a frequency-domain diagonal filter does not automatically behave like a local convolution kernel over graph nodes.

### 2. Expanding the Node-Space Operator

The $(i,j)$ entry of $A$ is:

$$
A_{ij}=\sum_{k=1}^{n}\theta_k U_{ik}U_{jk}.
$$

Therefore, the output at node $i$ is:

$$
y_i=\sum_{j=1}^{n}A_{ij}x_j
=\sum_{j=1}^{n}
\left(
\sum_{k=1}^{n}
\theta_k U_{ik}U_{jk}
\right)
x_j.
$$

This shows that node $i$ may depend on many or even all nodes $j$ in the graph.

The reason is that Laplacian eigenvectors are usually global graph modes. They are not generally sparse one-node or one-neighborhood patterns. Therefore, after transforming back from spectral space to node space, the filter may mix information from distant nodes.

The key point is:

Frequency-domain diagonal does not mean node-domain local.

### 3. Locality Requirement for a CNN-Like Graph Filter

A CNN-like graph filter should satisfy a locality condition. If the graph distance between node $i$ and node $j$ is greater than $K$, then node $j$ should not directly influence the output at node $i$:

$$
A_{ij}=0
\quad
\mathrm{when}
\quad
d_G(i,j)>K.
$$

A non-parametric spectral filter does not guarantee this condition.

Even though it is flexible in frequency space, it does not ensure compact support in node space. This is why the paper says non-parametric spectral filters are not localized.

### 4. Delta Signal Interpretation

A useful way to understand localization is to apply the filter to a delta signal.

Let $\boldsymbol{\delta}_i$ be a signal that is $1$ at node $i$ and $0$ elsewhere. Then:

$$
g_\theta(L)\boldsymbol{\delta}_i
$$

is the graph filter kernel centered at node $i$.

If the filter is localized, this output should be nonzero only around node $i$.

But for the non-parametric spectral filter,

$$
g_\theta(L)\boldsymbol{\delta}_i
=U\mathrm{diag}(\theta)U^T\boldsymbol{\delta}_i,
$$

the response may be spread across many nodes. This means that a signal placed at one node can influence distant nodes after filtering.

### 5. Why This Motivates Polynomial Filters

The paper wants graph filters that are closer to CNN kernels:

* local in the node domain;
* parameter-efficient;
* efficient to compute;
* reusable across the graph.

The non-parametric filter is too free. It gives every graph frequency an independent parameter, so its learning complexity is $O(n)$. It also does not guarantee spatial localization.

This motivates the polynomial parametrization:

$$
g_\theta(\Lambda)=\sum_{k=0}^{K-1}\theta_k\Lambda^k.
$$

Using the eigendecomposition of the Laplacian, this corresponds to the node-space operator:

$$
g_\theta(L)=\sum_{k=0}^{K-1}\theta_k L^k.
$$

This is important because powers of $L$ have graph-local support: $L^k$ can only connect nodes within approximately $k$ graph hops. Therefore, polynomial filters constrain the spectral filter to become localized in the node domain.

### 6. PM2.5 Research Interpretation

For PM2.5 station graphs, non-parametric spectral filters are risky because a station's output may directly depend on distant stations in ways that are not physically or meteorologically justified.

This can weaken interpretability and reliability. The model may exploit global spectral mixing that performs well on a standard split but fails under distribution shift, missingness, spikes, or graph mismatch.

A polynomial filter is more research-friendly because it makes the spatial dependency range explicit. The order $K$ becomes a modeling assumption: it represents the assumed graph-hop range of useful spatial dependency.

This means $K$ should not be treated merely as a hyperparameter. It should be analyzed as part of the reliability protocol:

* Does a small $K$ miss long-range pollution transport?
* Does a large $K$ mix unrelated stations?
* Does the best $K$ change under temporal shift?
* Does graph mismatch change calibration, coverage, or decision cost?

In this sense, polynomial filtering connects the mathematical definition of graph convolution to reliability evaluation.

## Why Polynomial Filters Restore Locality

The previous section explains the failure mode of a non-parametric spectral filter: it is diagonal in the graph-frequency basis, but after transforming back to the node basis it usually becomes a dense node-space operator. This is the main reason that spectral filtering alone is not yet CNN-like. It solves the lack of translation on graphs by defining convolution through the Laplacian eigenbasis, but a fully free frequency response still does not guarantee local support.

Polynomial filters are the paper's key restriction on the spectral filter class. This restriction is not a weakness of the method. It is the inductive bias that makes the filter local, parameter-efficient, and computationally closer to a CNN kernel.

### 1. From Non-Parametric Frequency Weights to Structured Frequency Response

The non-parametric spectral filter is:

$$
g_\theta(\Lambda)=\mathrm{diag}(\theta),
$$

where

$$
\theta\in\mathbb{R}^n.
$$

This gives every graph frequency an independent parameter. In the graph Fourier domain, this is maximally flexible:

$$
\hat{y}_k=\theta_k\hat{x}_k.
$$

However, this flexibility has two costs. First, it has no node-space locality guarantee, because $U\mathrm{diag}(\theta)U^T$ is generally dense. Second, its learning complexity grows with the number of nodes, because it requires $O(n)$ learnable parameters.

The polynomial filter instead uses:

$$
g_\theta(\Lambda)=\sum_{k=0}^{K-1}\theta_k\Lambda^k,
$$

where

$$
\theta\in\mathbb{R}^K.
$$

This means the filter no longer learns one free parameter per graph frequency. Instead, it learns a structured frequency response curve:

$$
g_\theta(\lambda)=\sum_{k=0}^{K-1}\theta_k\lambda^k.
$$

For each eigenvalue $\lambda_i$, the response is:

$$
g_\theta(\lambda_i)=\theta_0+\theta_1\lambda_i+\theta_2\lambda_i^2+\cdots+\theta_{K-1}\lambda_i^{K-1}.
$$

The first benefit is parameter efficiency: the number of learnable parameters becomes $O(K)$ instead of $O(n)$. The second benefit is locality, because this polynomial frequency response can be rewritten as a polynomial of the Laplacian in node space.

### 2. Why a Polynomial in $\Lambda$ Becomes a Polynomial in $L$

The key reason polynomial filters restore locality is that the graph Laplacian is diagonalized by the graph Fourier basis:

$$
L=U\Lambda U^T.
$$

Since the eigenvectors are orthonormal,

$$
U^TU=I_n.
$$

For $m=2$:

$$
\begin{aligned}
L^2
&=(U\Lambda U^T)(U\Lambda U^T) \\
&=U\Lambda(U^TU)\Lambda U^T \\
&=U\Lambda I_n\Lambda U^T \\
&=U\Lambda^2U^T.
\end{aligned}
$$

By the same argument, for any positive integer $m$:

$$
L^m=U\Lambda^mU^T.
$$

Therefore:

$$
\begin{aligned}
\sum_{k=0}^{K-1}\theta_kL^k
&=\sum_{k=0}^{K-1}\theta_kU\Lambda^kU^T \\
&=U
\left(
\sum_{k=0}^{K-1}\theta_k\Lambda^k
\right)
U^T.
\end{aligned}
$$

The expression inside the parentheses is exactly $g_\theta(\Lambda)$. Thus:

$$
g_\theta(L)
=Ug_\theta(\Lambda)U^T
=\sum_{k=0}^{K-1}\theta_kL^k.
$$

This is the central mathematical bridge in Section 2.1. A polynomial frequency response can be implemented directly as a polynomial of the graph Laplacian in node space. The model no longer needs to apply a dense arbitrary spectral operator at every layer.

### 3. Why $L^k$ Gives Graph-Hop Locality

For the unnormalized Laplacian,

$$
L=D-W.
$$

For $i\neq j$,

$$
L_{ij}=-W_{ij}.
$$

Thus, if nodes $i$ and $j$ are not connected by an edge, then $W_{ij}=0$ and $L_{ij}=0$. This means that the off-diagonal nonzero entries of $L$ only connect one-hop neighbors. The diagonal entries preserve self-information through the degree term. The normalized Laplacian changes the weights, but it has the same basic off-diagonal sparsity pattern when the graph is undirected and degrees are nonzero.

For $L^2$:

$$
(L^2)_{ij}=\sum_m L_{im}L_{mj}.
$$

This term can be nonzero only if there exists an intermediate node $m$ such that:

$$
L_{im}\neq 0
$$

and

$$
L_{mj}\neq 0.
$$

This corresponds to a path:

$$
i\rightarrow m\rightarrow j.
$$

Therefore, $L^2$ can transmit information across two-hop neighborhoods. More generally, $(L^k)_{ij}$ can be nonzero only if there exists a path from $i$ to $j$ of length at most $k$. Therefore:

$$
d_G(i,j)>k
\quad\Rightarrow\quad
(L^k)_{ij}=0.
$$

This is why a polynomial in $L$ has graph-local support. If the highest power of $L$ is $K-1$, then the filter can only directly combine information within a bounded graph-hop neighborhood. The exact off-by-one convention depends on whether $K$ denotes the number of coefficients, the polynomial order, or the support size. The essential point is that the maximum power of $L$ determines the maximum graph-hop range.

### 4. What $L^0$ Means

By convention:

$$
L^0=I_n.
$$

This is the identity matrix, not a new graph operator. Therefore:

$$
L^0\mathbf{x}=I_n\mathbf{x}=\mathbf{x}.
$$

In the polynomial filter,

$$
g_\theta(L)
=\theta_0I_n+\theta_1L+\theta_2L^2+\cdots+\theta_{K-1}L^{K-1},
$$

the first term $\theta_0I_n$ corresponds to zero-hop self-information. It preserves each node's own signal before mixing information through graph neighborhoods.

| Term | Interpretation |
| ---- | -------------- |
| $\theta_0I_n\mathbf{x}$ | 0-hop self-information |
| $\theta_1L\mathbf{x}$ | 1-hop graph difference / neighbor relation |
| $\theta_2L^2\mathbf{x}$ | Up to 2-hop propagation |
| $\theta_kL^k\mathbf{x}$ | Up to $k$-hop propagation |

This makes the polynomial filter similar in spirit to a CNN kernel: the support size controls how far the filter can look. The difference is that a graph kernel's support is measured by graph hops, not by regular-grid pixel offsets.

### 5. Why $U^T$ Is the Inverse of $U$ in This Paper

In the paper, $U^T$ is the transpose of $U$. It is also the inverse of $U$ because $U$ is an orthogonal matrix.

This is not true for arbitrary matrices. It holds here because the paper considers undirected connected graphs, whose graph Laplacian is real symmetric and positive semidefinite. By the spectral theorem, $L$ has a complete set of orthonormal eigenvectors.

Therefore:

$$
U^TU=I_n
$$

and

$$
UU^T=I_n.
$$

Hence:

$$
U^{-1}=U^T.
$$

This is why the graph Fourier transform is written as:

$$
\hat{\mathbf{x}}=U^T\mathbf{x},
$$

and the inverse graph Fourier transform is:

$$
\mathbf{x}=U\hat{\mathbf{x}}.
$$

If the graph were directed, or if the relevant graph operator were not symmetric, this simple orthogonal Fourier basis would generally not apply.

### 6. Research Interpretation for PM2.5 Forecasting

For PM2.5 station graphs, polynomial filtering changes the meaning of the model.

A non-parametric spectral filter may mix information from distant stations in an uncontrolled way after projecting back to node space. This can create global dependencies that are difficult to justify physically or meteorologically.

A polynomial filter makes the dependency range explicit:

$$
K=\text{the assumed graph-hop range of useful spatial dependency}.
$$

Thus, $K$ is not merely a tuning parameter. It is a modeling assumption about how far useful pollution-related information can propagate on the chosen graph.

This creates concrete research questions:

* Does a small $K$ miss long-range pollution transport?
* Does a large $K$ mix unrelated stations and amplify graph mismatch?
* Does the best $K$ change under temporal distribution shift?
* Does graph construction change which $K$ is reliable?
* Do different $K$ values affect calibration, coverage, or decision cost differently?

The key reliability implication is that graph locality is only meaningful if the graph itself is meaningful. A $K$-localized filter on a wrong graph is still localized, but localized around the wrong neighbors.

## Chebyshev Recurrence: Efficient Computation of Localized Polynomial Filters

### 1. The Main Clarification

Chebyshev recurrence should not be understood as a simple abbreviation of $L^k$.

The correct relationship is:

$$T_k(\tilde L)\neq L^k.$$

Instead:

$$T_k(\tilde L)\in \mathrm{span}\{I_n,L,L^2,\ldots,L^k\}.$$

That is, $T_k(\tilde L)$ is a degree-$k$ polynomial in $L$. It is not the same as the monomial $L^k$, but it does not leave the polynomial-filtering framework. It remains a graph-local polynomial operator.

The recurrence changes the polynomial basis and the computational path. It does not change the basic idea that localized graph filtering is implemented through finite-degree polynomials of the Laplacian.

### 2. From Monomial Basis to Chebyshev Basis

A standard polynomial filter can be written using the monomial basis:

$$g_\theta(L)\mathbf{x}=\sum_{k=0}^{K-1}\theta_kL^k\mathbf{x}.$$

This uses the basis:

$$I_n,L,L^2,L^3,\ldots.$$

The Chebyshev filter instead uses:

$$g_\theta(L)\mathbf{x}=\sum_{k=0}^{K-1}\theta_kT_k(\tilde L)\mathbf{x}.$$

This uses the basis:

$$T_0(\tilde L),T_1(\tilde L),T_2(\tilde L),\ldots.$$

Both are polynomial bases. The difference is that Chebyshev polynomials provide a numerically stable and recursively computable basis for approximating spectral filter responses.

The first few Chebyshev polynomials are:

$$T_0(z)=1.$$

$$T_1(z)=z.$$

$$T_2(z)=2z^2-1.$$

$$T_3(z)=4z^3-3z.$$

After substituting the scaled Laplacian, the constant polynomial becomes the identity matrix:

$$T_0(\tilde L)=I_n.$$

The next terms are:

$$T_1(\tilde L)=\tilde L.$$

$$T_2(\tilde L)=2\tilde L^2-I_n.$$

$$T_3(\tilde L)=4\tilde L^3-3\tilde L.$$

These terms are still polynomials of the graph Laplacian after substituting the scaled Laplacian. The basis changes from monomials to Chebyshev polynomials, but the filter remains a finite polynomial graph filter.

### 3. Why the Laplacian Is Scaled

The scaled Laplacian is:

$$\tilde L=\frac{2L}{\lambda_{\max}}-I_n.$$

The corresponding scaled eigenvalue matrix is:

$$\tilde \Lambda=\frac{2\Lambda}{\lambda_{\max}}-I_n.$$

The purpose is to map the original Laplacian eigenvalues from:

$$[0,\lambda_{\max}]$$

to:

$$[-1,1].$$

This matters because Chebyshev polynomials are stable and orthogonal on the interval $[-1,1]$. Without this scaling, high-order polynomial terms may become numerically unstable when eigenvalues are large.

Thus, scaling is not a change of graph structure. It is a numerical normalization that makes Chebyshev polynomial approximation stable. For PM2.5 station graphs, this means the handling of $\lambda_{\max}$ should be documented because it affects the numerical behavior of the graph convolution layer.

### 4. Expanding the Relationship to $L^k$

Let:

$$a=\frac{2}{\lambda_{\max}}.$$

Then:

$$\tilde L=aL-I_n.$$

For the first order:

$$T_1(\tilde L)=\tilde L=aL-I_n.$$

Therefore:

$$T_1(\tilde L)\mathbf{x}=aL\mathbf{x}-\mathbf{x}.$$

This mixes 0-hop self-information and 1-hop graph-difference information.

For the second order:

$$T_2(\tilde L)=2\tilde L^2-I_n.$$

Since:

$$\tilde L^2=(aL-I_n)^2=a^2L^2-2aL+I_n,$$

we have:

$$T_2(\tilde L)=2a^2L^2-4aL+I_n.$$

Therefore:

$$T_2(\tilde L)\mathbf{x}=2a^2L^2\mathbf{x}-4aL\mathbf{x}+\mathbf{x}.$$

This shows that $T_2(\tilde L)\mathbf{x}$ is not simply $L^2\mathbf{x}$. It is a stable combination of 0-hop, 1-hop, and 2-hop graph information.

More generally, $T_k(\tilde L)\mathbf{x}$ is a combination of:

$$I_n\mathbf{x},L\mathbf{x},L^2\mathbf{x},\ldots,L^k\mathbf{x}.$$

Therefore, it can contain information up to $k$ graph hops, but not beyond $k$ hops.

### 5. What the Recurrence Actually Simplifies

The recurrence relation is:

$$T_k(z)=2zT_{k-1}(z)-T_{k-2}(z),$$

with:

$$T_0(z)=1,\qquad T_1(z)=z.$$

Substituting $z=\tilde L$ gives:

$$T_k(\tilde L)=2\tilde LT_{k-1}(\tilde L)-T_{k-2}(\tilde L).$$

Instead of constructing the matrix $T_k(\tilde L)$, the paper defines:

$$\bar{\mathbf{x}}_k=T_k(\tilde L)\mathbf{x}.$$

Then:

$$\bar{\mathbf{x}}_0=\mathbf{x}.$$

$$\bar{\mathbf{x}}_1=\tilde L\mathbf{x}.$$

For $k\ge 2$:

$$\bar{\mathbf{x}}_k=2\tilde L\bar{\mathbf{x}}_{k-1}-\bar{\mathbf{x}}_{k-2}.$$

This is the key computational simplification.

The model does not need to explicitly construct:

$$L^2,L^3,\ldots,L^k.$$

It also does not need to explicitly construct:

$$T_k(\tilde L).$$

It only needs to recursively compute the vectors:

$$\bar{\mathbf{x}}_0,\bar{\mathbf{x}}_1,\ldots,\bar{\mathbf{x}}_{K-1}.$$

Each recursive step requires one multiplication of the sparse matrix $\tilde L$ with a vector.

### 6. Why This Is Fast

The direct spectral filtering form is:

$$\mathbf{y}=Ug_\theta(\Lambda)U^T\mathbf{x}.$$

This requires multiplication by the dense graph Fourier basis $U$, which costs $O(n^2)$ and requires storing or computing the eigenbasis.

Chebyshev recurrence avoids this during filtering. More precisely, it avoids:

* explicit Fourier basis multiplication by $U$ and $U^T$;
* explicit eigendecomposition during the filtering operation;
* explicit construction of dense matrix powers such as $L^2,L^3,\ldots,L^k$;
* explicit construction of the matrix $T_k(\tilde L)$.

The final filtering operation is:

$$\mathbf{y}=\sum_{k=0}^{K-1}\theta_k\bar{\mathbf{x}}_k.$$

Equivalently:

$$\mathbf{y}=[\bar{\mathbf{x}}_0,\bar{\mathbf{x}}_1,\ldots,\bar{\mathbf{x}}_{K-1}]\theta.$$

Since each $\bar{\mathbf{x}}_k$ is computed by one sparse Laplacian-vector multiplication, each step costs $O(|E|)$ on a sparse graph. Across $K$ terms, the paper's localized filtering cost is:

$$O(K|E|).$$

The final weighted sum over the $K$ vectors costs $O(Kn)$, which is compatible with the same sparse-graph scaling when $|E|$ is proportional to graph size. This is the computational meaning of fast localized spectral filtering.

### 7. Why Locality Is Preserved

Chebyshev recurrence preserves graph-hop locality because $T_k(\tilde L)$ is still a degree-$k$ polynomial in $L$.

Since $L^k$ cannot propagate information beyond $k$ graph hops, any polynomial involving only powers up to $L^k$ also cannot introduce dependencies beyond $k$ hops.

Therefore, $T_k(\tilde L)\mathbf{x}$ may mix 0-hop through $k$-hop information, but it cannot directly depend on nodes farther than $k$ hops.

The recurrence is a fast way to compute these graph-local polynomial features, not a mechanism that introduces global mixing.

### 8. Intuition for PM2.5 Forecasting

For a PM2.5 station graph, let $\mathbf{x}_t$ be the PM2.5 signal at time $t$.

Then:

$$\bar{\mathbf{x}}_0=\mathbf{x}_t$$

is the original station-level signal.

The first recursive term is:

$$\bar{\mathbf{x}}_1=\tilde L\mathbf{x}_t$$

which contains one-hop local contrast or local graph-difference information.

The second recursive term is:

$$\bar{\mathbf{x}}_2=T_2(\tilde L)\mathbf{x}_t$$

which contains a stable combination of 0-hop, 1-hop, and 2-hop graph information.

More generally:

$$\bar{\mathbf{x}}_k=T_k(\tilde L)\mathbf{x}_t$$

contains graph-structured information up to $k$ hops.

The learned coefficients $\theta_k$ determine how these different graph-local patterns are combined into a spatial feature representation:

$$\mathbf{h}_t=\sum_{k=0}^{K-1}\theta_k\bar{\mathbf{x}}_k.$$

This provides an efficient local spatial feature extractor. It is not a reliability guarantee. If the graph construction is wrong, Chebyshev recurrence will still efficiently propagate information over the wrong neighborhoods.

### 9. Final Understanding

The most important summary is:

Chebyshev recurrence simplifies the computational implementation, not the mathematical meaning of graph filtering.

It replaces explicit dense Fourier-basis multiplication and explicit high-order matrix construction with recursive sparse Laplacian-vector multiplications.

It is best understood as:

1. Choose a stable polynomial basis.
2. Scale the Laplacian eigenvalues into $[-1,1]$.
3. Generate graph-local basis features recursively.
4. Learn coefficients to combine these basis features.
5. Preserve $K$-hop locality because the filter remains a finite-degree polynomial in $L$.

## Additional Clarifications on Chebyshev Filtering

The previous sections establish the main logic of Section 2.1: direct spectral filtering defines graph convolution, polynomial filtering restores locality, and Chebyshev recurrence makes localized polynomial filtering efficient and stable. The following clarifications answer four further questions about the choice of $K$, the interpretability tradeoff between bases, the actual computational saving, and the meaning of the recurrence coefficients.

### 1. How Should the Hyperparameter $K$ Be Chosen?

In this paper, $K$ controls the polynomial order and therefore the graph-filter support size. Under the paper's notation, the Chebyshev filter uses the terms $k=0,\ldots,K-1$, so the highest explicit polynomial degree is $K-1$. The essential modeling meaning is that $K$ controls the maximum graph-hop range that the filter can directly use, up to this notation convention.

$K$ is not learned by gradient descent as a model parameter. It is a model-design hyperparameter. The learned quantities are the filter coefficients such as $\theta_k$; $K$ determines how many such basis terms are available and how far the local graph propagation can reach.

Increasing $K$ lets the filter use information from a larger graph-hop neighborhood. It also increases parameter count and computation approximately linearly, because the layer must compute and combine more Chebyshev basis features:

$$
\mathbf{y}=\sum_{k=0}^{K-1}\theta_k\bar{\mathbf{x}}_k.
$$

A small $K$ makes the filter more local and computationally cheaper, but it may miss useful long-range spatial dependency. A large $K$ allows broader spatial mixing, but it may also mix unrelated nodes and amplify graph mismatch.

For PM2.5 forecasting, $K$ should be selected through a combination of:

* validation error;
* calibration and coverage behavior;
* robustness under temporal shift, missingness, and noise;
* downstream decision cost;
* graph construction sensitivity;
* domain knowledge about pollution transport and station connectivity.

Thus, $K$ should not be treated as a purely mechanical tuning parameter. It represents an assumption about the maximum graph-hop range of useful spatial dependency. A useful research protocol is to evaluate several values of $K$ under the same graph, and then repeat the comparison under different graph constructions. If the best $K$ changes dramatically under temporal shift, missingness, noise, or graph perturbation, this may indicate that the learned locality is not robust.

### 2. Interpretability Tradeoff: Monomial Basis vs Chebyshev Basis

A monomial polynomial filter has the form:

$$
g_\theta(L)\mathbf{x}
=
\theta_0I_n\mathbf{x}
+
\theta_1L\mathbf{x}
+
\theta_2L^2\mathbf{x}
+
\cdots.
$$

This basis has a relatively direct node-space interpretation:

* $\theta_0$ weights self-information;
* $\theta_1$ weights information produced by one application of the graph Laplacian;
* $\theta_2$ weights information produced by two applications of the graph Laplacian;
* $\theta_k$ weights information contained in $L^k\mathbf{x}$.

Even in the monomial basis, however, $L^k\mathbf{x}$ should not be interpreted as pure $k$-hop information. Because the Laplacian contains diagonal self terms, powers of $L$ mix self-information and lower-hop information.

For example, if the normalized Laplacian is written as:

$$
L=I_n-S,
$$

where $S=D^{-1/2}WD^{-1/2}$, then:

$$
L^2=(I_n-S)^2=I_n-2S+S^2.
$$

The term $I_n\mathbf{x}$ is 0-hop self-information, $S\mathbf{x}$ is one-hop normalized neighbor mixing, and $S^2\mathbf{x}$ contains information up to two hops. Thus, $L^2\mathbf{x}$ is not pure 2-hop information. The more precise statement is that $L^k\mathbf{x}$ contains information up to $k$ hops.

The Chebyshev filter uses:

$$
g_\theta(L)\mathbf{x}
=
\sum_{k=0}^{K-1}\theta_kT_k(\tilde L)\mathbf{x}.
$$

Here, $\theta_k$ is no longer directly the weight of the $k$-th monomial term $L^k\mathbf{x}$. Instead, it is the coefficient of the $k$-th Chebyshev basis function in the learned spectral filter response.

For example:

$$
T_2(\tilde L)=2\tilde L^2-I_n.
$$

Since $\tilde L=aL-I_n$ with $a=2/\lambda_{\max}$:

$$
\tilde L^2=(aL-I_n)^2=a^2L^2-2aL+I_n.
$$

Therefore:

$$
T_2(\tilde L)=2a^2L^2-4aL+I_n.
$$

So $T_2(\tilde L)\mathbf{x}$ already mixes:

$$
I_n\mathbf{x},\quad L\mathbf{x},\quad L^2\mathbf{x}.
$$

Therefore, Chebyshev coefficients are less directly interpretable as hop-specific weights. The correct interpretation is:

* $\theta_k$ is a coordinate of the spectral filter response in the Chebyshev polynomial basis;
* $\bar{\mathbf{x}}_k=T_k(\tilde L)\mathbf{x}$ is a Chebyshev-filtered graph signal;
* $\bar{\mathbf{x}}_k$ contains information up to $k$ graph hops, but it is not pure $k$-hop information.

Thus, Chebyshev filtering trades some hop-level coefficient interpretability for numerical stability and efficient recurrence.

### 3. What Computation Does Chebyshev Recurrence Actually Simplify?

Chebyshev recurrence simplifies the computational path in two ways.

First, it avoids explicit multiplication by the graph Fourier basis. The direct spectral filtering form is:

$$
\mathbf{y}=Ug_\theta(\Lambda)U^T\mathbf{x}.
$$

This requires the dense eigenvector matrix $U$. Multiplication by $U$ or $U^T$ is generally expensive, and computing or storing the eigendecomposition is also costly.

By using Chebyshev polynomials, the filter can be written directly in the node domain:

$$
\mathbf{y}
=
\sum_{k=0}^{K-1}
\theta_kT_k(\tilde L)\mathbf{x}.
$$

This avoids explicit graph Fourier transform and inverse graph Fourier transform during filtering.

Second, it avoids explicit construction of high-order matrices such as:

$$
L^2,\ L^3,\ldots,L^k
$$

or:

$$
T_k(\tilde L).
$$

Instead, it recursively computes the vectors:

$$
\bar{\mathbf{x}}_k=T_k(\tilde L)\mathbf{x}.
$$

The recurrence is:

$$
\bar{\mathbf{x}}_0=\mathbf{x},
$$

$$
\bar{\mathbf{x}}_1=\tilde L\mathbf{x},
$$

and for $k\ge 2$:

$$
\bar{\mathbf{x}}_k
=
2\tilde L\bar{\mathbf{x}}_{k-1}
-
\bar{\mathbf{x}}_{k-2}.
$$

Each step only requires a sparse matrix-vector multiplication with $\tilde L$.

Therefore, the main simplification is not simply a space-for-speed tradeoff. The recurrence reduces both computational and storage burden by avoiding dense Fourier-basis operations and explicit high-order matrix construction. The total recurrence cost is:

$$
O(K|E|),
$$

because each recurrence step uses the sparse graph structure. The final weighted combination over the $K$ vectors costs $O(Kn)$.

### 4. Why Does the Recurrence Have This Form?

The Chebyshev recurrence is not arbitrary. It comes from the definition of Chebyshev polynomials:

$$
T_k(\cos\alpha)=\cos(k\alpha).
$$

The trigonometric product-to-sum identity gives:

$$
2\cos\alpha\cos((k-1)\alpha)
=
\cos(k\alpha)+\cos((k-2)\alpha).
$$

Rearranging:

$$
\cos(k\alpha)
=
2\cos\alpha\cos((k-1)\alpha)
-
\cos((k-2)\alpha).
$$

Now set:

$$
z=\cos\alpha.
$$

Using the defining relationship again:

$$
T_k(z)=\cos(k\alpha),
$$

$$
T_{k-1}(z)=\cos((k-1)\alpha),
$$

and:

$$
T_{k-2}(z)=\cos((k-2)\alpha).
$$

Substituting these into the trigonometric identity gives:

$$
T_k(z)
=
2zT_{k-1}(z)-T_{k-2}(z).
$$

After replacing $z$ by $\tilde L$, the same recurrence becomes a recurrence over graph-filtered signals:

$$
\bar{\mathbf{x}}_k
=
2\tilde L\bar{\mathbf{x}}_{k-1}
-
\bar{\mathbf{x}}_{k-2}.
$$

The constants $2$ and $-1$ in the recurrence are not learned graph weights and not hop-specific coefficients. They are fixed coefficients inherited from the Chebyshev polynomial basis. The learnable coefficients are $\theta_k$ in the filter expansion.

In the frequency domain, the filter response is:

$$
g_\theta(\tilde \lambda)
=
\sum_{k=0}^{K-1}\theta_kT_k(\tilde \lambda).
$$

Thus, $\theta_k$ measures how much the $k$-th Chebyshev basis function contributes to the learned frequency response. It is a coordinate of the filter response function in the Chebyshev polynomial basis, not a coordinate of the input signal itself and not a pure $k$-hop weight.

In the node domain, the corresponding basis feature is:

$$
\bar{\mathbf{x}}_k=T_k(\tilde L)\mathbf{x}.
$$

The final filtered signal is:

$$
\mathbf{y}
=
\sum_{k=0}^{K-1}
\theta_k\bar{\mathbf{x}}_k.
$$

Therefore, the model first generates Chebyshev-basis graph-local features, then learns how to combine them.

### 5. Final Refined Understanding

Chebyshev filtering should be understood as follows:

1. $K$ controls the maximum graph-hop support, subject to the paper's indexing convention, and should be selected by validation error, calibration and coverage behavior, robustness under shift, decision cost, graph construction sensitivity, and domain knowledge.
2. Moving from monomial basis to Chebyshev basis reduces direct hop-level interpretability of individual coefficients.
3. The benefit is that the Chebyshev basis gives a stable and recursively computable approximation of the spectral filter response.
4. The recurrence avoids explicit multiplication by $U$, explicit eigendecomposition during filtering, explicit construction of $L^k$, and explicit construction of $T_k(\tilde L)$.
5. $\theta_k$ is a Chebyshev-basis coefficient of the filter response, not a pure $k$-hop weight.
6. $\bar{\mathbf{x}}_k$ is a graph-local Chebyshev-filtered feature containing information up to $k$ hops.
7. For PM2.5 forecasting, Chebyshev filtering is an efficient local spatial feature extraction mechanism, not a reliability guarantee.
8. If the station graph is wrong, the recurrence still propagates information efficiently, but it propagates the wrong local information.

## Questions for My Project

1. What should an edge represent in a PM2.5 station graph?
2. Should the graph be based on distance, historical correlation, meteorology, wind direction, domain knowledge, or a hybrid design?
3. How sensitive are STGCN predictions to graph construction?
4. Does graph quality affect uncertainty calibration and empirical coverage under shift?
5. Does increasing Chebyshev order $K$ improve long-horizon forecasts or amplify noisy spatial propagation?
6. Should graph perturbation be included as a post-MVP reliability stress test?
7. Can graph mismatch explain downstream decision instability?
8. Should graph construction be evaluated separately under normal periods, extreme pollution events, and seasonal regimes?
9. How should directed environmental transport be handled if the baseline uses an undirected Laplacian?
10. What evidence would justify one station graph over another?

## Implementation Implications

Before implementing STGCN, I need to specify:

* adjacency matrix construction,
* edge weighting rule,
* sparsification rule,
* Laplacian type,
* Laplacian normalization,
* Chebyshev order $K$,
* maximum eigenvalue handling,
* sparse matrix representation,
* graph construction reproducibility,
* graph-quality sensitivity tests.

A minimal Chebyshev graph convolution interface should make the graph assumptions visible:

* Input: feature matrix, adjacency or Laplacian, and Chebyshev order $K$.
* Output: graph-filtered features.
* Assumptions: undirected graph, fixed graph, and sparse Laplacian.
* Tests: shape preservation, $K$-hop locality, and deterministic output under a fixed graph.

For the reliability project, implementation tests are not enough. The experimental protocol should later compare at least distance-based, correlation-based, and perturbed graph variants under forecasting error, coverage, sharpness, and decision-cost metrics.

## What I Should Be Able to Explain After Reading

1. Why is translation not naturally defined on graphs?
2. What role does the graph Laplacian play?
3. Why do Laplacian eigenvectors form graph Fourier modes?
4. What does spectral filtering mean?
5. Why is a non-parametric spectral filter not localized?
6. Why does a polynomial in $L$ produce $K$-hop locality?
7. Why is Chebyshev recurrence computationally efficient?
8. What does $O(K|E|)$ mean?
9. Why does graph quality matter?
10. How does this paper prepare me to understand STGCN?

## Reading Status

Reading

## Follow-Up Actions

1. Use this note as the foundation for reading P-GRAPH-002 and STGCN.
2. Map the STGCN graph convolution module back to Chebyshev polynomial filtering.
3. Document graph construction choices before implementing the PM2.5 baseline.
4. Add graph construction comparison to the post-MVP reliability stress-test backlog.
5. Track whether graph mismatch affects calibration, coverage, and decision-cost metrics.
