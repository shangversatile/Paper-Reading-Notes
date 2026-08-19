# P-ROB-001: Provably Robust Conformal Prediction with Improved Efficiency

## 1. Citation

**Paper:** Provably Robust Conformal Prediction with Improved Efficiency
**Authors:** Ge Yan, Yaniv Romano, Tsui-Wei Weng
**Year:** 2024
**Venue:** ICLR 2024
**Primary Source:** ICLR 2024 proceedings paper, https://proceedings.iclr.cc/paper_files/paper/2024/file/8759c20675d67ef3b91c4607cecbb27e-Paper-Conference.pdf
**Paper ID:** P-ROB-001
**Reading Status:** Reading

**Current close-reading scope:** whole-paper framework, Introduction-level motivation, prerequisite probability/randomness clarifications, Section 2.1 vanilla split conformal prediction first-pass foundations, Appendix A.2 randomized-score rank-coverage proof skeleton, and formal transition readiness for Section 2.2.

**Explicitly deferred:** Section 2.2 full RSCP derivation, Section 3, Theorem 1, Corollary 2, Empirical Bernstein refinement, PTT, RCT, experiments, final critique, and final research questions.

This note remains a research-reading workspace. Do not add DOI, OpenReview ID, arXiv ID, page numbers, official code URL, experiment results, or ablation claims until they are independently verified from the paper or another primary source.

---

## 2. Reading Tier and Track

**Reading Tier:** Tier 1
**Primary Track:** Robustness and Robust Reliability
**Related Project:** Reliable Spatiotemporal Forecasting under Dynamic Distribution Shift
**Current Reading Scope:** foundational framework and Section 2.1 first-pass foundations substantially complete; Section 2.2 ready / beginning; paper still Reading

**Secondary Topics:**

* Conformal Prediction
* Uncertainty Quantification
* Adversarial Robustness
* Randomized Smoothing
* Calibration
* Statistical Learning
* Certified Robustness
* Model Reliability

### Why This Paper Is in the Curriculum

This paper is included because it sits at the intersection of conformal prediction, adversarial robustness, randomized smoothing, finite-sample calibration, and efficiency of prediction sets. The reading objective is to understand how a formal robustness guarantee is connected to the random object actually implemented, and where efficiency may be lost or recovered.

The paper is especially relevant because it separates two problems that are easy to conflate:

1. **Challenge 1: robustness certification / theory-implementation alignment.** A robustness theorem should certify the score that the computer actually computes, not only an ideal population-smoothed score.
2. **Challenge 2: prediction-set efficiency / conservativeness.** A valid robust set can still become so large that it is uninformative.

This is a foundational curriculum placement and a partial close-reading record, not a completed judgment about the whole paper.

### Epistemic Discipline for This Note

Future entries should label claims when needed:

* **Paper statement:** explicitly stated by the paper.
* **Derived interpretation:** reconstructed from definitions, equations, or proofs.
* **Research interpretation:** my own abstraction or critique.
* **Project connection:** relevance to dynamic-system reliability research.
* **Open question:** not yet resolved by the reading.
* **Not yet verified:** plausible but not checked against the paper.

Current labels used below:

* **Paper statement:** definitions, claims, or proof skeletons attributed to the paper.
* **Derived explanation:** reconstruction from the paper's definitions and conformal prediction logic.
* **Pedagogical toy example:** a simplified example used for intuition, not a paper theorem or experiment.
* **Research interpretation:** my abstraction, critique, or transfer framing.
* **Project connection:** relevance to dynamic-system reliability research.
* **Not yet verified:** deferred material that must not be treated as completed.

---

## 3. Core Problem

**Derived explanation.** The paper starts from ordinary conformal prediction and asks how to preserve a meaningful coverage guarantee when the test input may be adversarially perturbed, while still keeping the prediction set informative.

The whole-paper problem decomposition is:

```text
Vanilla conformal prediction
        v
finite-sample marginal coverage under exchangeability
        v
adversarially perturbed test input
        v
ordinary clean conformal rank argument no longer directly applies
        v
RSCP introduces randomized smoothing
        v
population smoothed score admits perturbation bound
        v
practical randomized smoothing uses finite Monte Carlo estimation
        v
original RSCP does not explicitly control this implementation error
        v
RSCP+
        v
formal certificate for implemented Monte Carlo score
        v
but robust correction is highly conservative
        v
large / trivial prediction sets
        v
PTT and RCT reshape score/model geometry
        v
validity + robustness + informativeness
```

The first conceptual obstacle is that ordinary split conformal prediction obtains coverage from exchangeability of calibration and test **true-label scores**. If an adversary changes only the test input, the clean calibration scores and the attacked test score generally no longer have the same distribution. The ordinary clean-data rank argument therefore cannot be silently reused.

The second obstacle is implementation. Randomized smoothing naturally defines a population quantity:

```math
S_{\mathrm{RS}}(x,y)
=
\mathbb E_Z[S(x+Z,y)].
```

But the actual algorithm cannot evaluate this expectation exactly. It uses a finite Monte Carlo estimator:

```math
\widehat S_{\mathrm{RS},N_{\mathrm{MC}}}(x,y)
=
\frac1{N_{\mathrm{MC}}}
\sum_{j=1}^{N_{\mathrm{MC}}}
S(x+Z_j,y).
```

For finite `N_MC`, these are different mathematical objects. The paper's Section 3 motivation, as currently understood, is that the original RSCP robustness certificate lives at the population-smoothed-score level but the implemented method uses the Monte Carlo score. RSCP+ is introduced to close this theory-implementation gap.

The third obstacle is efficiency. Robust certification can require threshold inflation. Inflated thresholds can make prediction sets large or even trivial. PTT and RCT are later-paper mechanisms intended to improve prediction-set efficiency, but this note has not yet completed those sections.

The Section 2.1 foundation now has a precise causal chain:

```text
user-specified target coverage 1-alpha
        v
required pooled rank k
        v
calibration empirical quantile / threshold tau
        v
candidate-label score comparison
        v
prediction set C(x)
        v
true-label membership event
        v
marginal finite-sample coverage
```

This chain is the clean conformal mechanism that Section 2.2 must modify for adversarially perturbed test inputs.

### Four Evaluation Axes

The paper should be evaluated along four separate axes:

**Validity.** Whether the stated coverage probability is mathematically guaranteed.

**Robustness.** Whether the guarantee transfers under a specified adversarial perturbation set.

**Prediction-set efficiency.** Whether the returned set is small, selective, and informative.

**Computational efficiency.** Monte Carlo cost, training cost, runtime, and implementation overhead.

These are not the same as generic "performance." A method can be valid but conservative, robust but computationally expensive, or efficient in set size but uncertified under attack.

---

## 4. Intuition Before the Math

**Derived explanation.** The clean conformal mechanism is a rank argument. If calibration true-label scores and the test true-label score are exchangeable, the test score does not have a privileged position among the pooled scores. Choosing a high enough calibration order statistic therefore gives finite-sample marginal coverage.

Adversarial perturbation attacks this rank mechanism indirectly. The calibration scores are computed on clean calibration inputs, while the deployed test score may be computed on an attacked input. Even if the attacked test input remains independent of the calibration sample, its score distribution can differ from the clean calibration-score distribution. The relevant structure broken is the identical-score-distribution / exchangeability structure, not necessarily independence.

Randomized smoothing changes the score as a function of input. Instead of evaluating the base score at `x`, it averages the score under Gaussian perturbations around `x`. After an inverse Gaussian CDF transform, the smoothed score has a certified Lipschitz-type relation under `L_2` perturbations. This creates a bridge from clean membership to adversarial membership by inflating the conformal threshold.

The implementation problem is that Gaussian averaging is an expectation, while the computer computes a sample average. LLN says the average converges asymptotically; CLT gives an approximate large-sample error distribution; Hoeffding gives the finite-sample high-probability control needed for a certificate.

At the end of Section 2.1, the key transition is:

```text
clean exchangeable scores
        v
rank-calibrated threshold
        v
clean marginal coverage
        v
adversarial test score is no longer in the same exchangeable score sequence
        v
need a certified clean-to-attacked score transfer
```

RSCP does not make the attacked score magically exchangeable with clean calibration scores. It introduces a different bridge: a robustness relation between a clean robust score and an attacked robust score.

### Whole-Paper Dependency Graph

Research map - later stages are not yet completed.

```text
Split Conformal Prediction
        v
exchangeability / rank / empirical quantile
        v
marginal coverage
        v
Randomized Smoothed CP
        v
Gaussian smoothing perturbation relation
        v
finite Monte Carlo implementation
        v
RSCP implementation-certificate gap
        v
RSCP+
        v
Hoeffding + smoothing relation + union bound
        v
Theorem 1
        v
Corollary 2
        v
robust conformal coverage
        v
conservativeness
        v
PTT / RCT
```

This graph is a dependency map, not a completion record. The only completed close-reading layers here are the framework, prerequisite randomness distinctions, vanilla split conformal foundations, and Appendix A.2 randomized-score coverage proof skeleton.

### Mechanism Chain for Randomized Smoothing

The idealized RSCP intuition is:

```text
Gaussian averaging
        v
smooths the score as a function of input
        v
inverse-Gaussian transform creates a quantity
with a certified Lipschitz-type bound
        v
L2-bounded adversarial perturbation
        v
worst-case score inflation is bounded
        v
inflate the conformal threshold by the same amount
        v
clean membership can be transferred
to adversarial membership
```

This chain is only the population-level logic. The implemented algorithm still needs Monte Carlo estimation and finite-sample concentration control.

### Uncertainty Bookkeeping

The note must keep the following random or uncertain objects separate:

| Object | Source | Role in This Reading |
| --- | --- | --- |
| Data randomness | `(X_i,Y_i)` drawn from the data-generating process | Drives exchangeability and conformal rank coverage |
| Score randomization | APS uses `U\sim\mathrm{Uniform}[0,1]` | Randomized scores can still be conformally valid if generated symmetrically |
| Gaussian smoothing randomness | `Z\sim\mathcal N(0,\sigma^2 I_p)` | Defines the population smoothed score |
| Monte Carlo sampling randomness | finite samples `Z_1,\ldots,Z_{N_{\mathrm{MC}}}` | Makes the implemented smoothed score random around its expectation |
| Adversarial perturbation | bounded vector `\Delta` | Changes the deployed test input and is not necessarily random |

Do not call all of these "noise." They live in different probability or uncertainty layers and support different proof mechanisms.

### Validity / Robustness / Efficiency Separation

These four evaluation axes must not be mixed. This table is the reading checklist for later sections:

| Axis | Meaning | Reading Check |
| --- | --- | --- |
| Validity | Whether the stated coverage guarantee holds | Section 2.1 marginal coverage now established |
| Robustness | Whether the guarantee transfers under the specified perturbation model | Perturbation bookkeeping established; full RSCP/RSCP+ proof deferred |
| Prediction-set efficiency | Whether the set is small and informative enough to use | Conservativeness motivation established; PTT/RCT deferred |
| Computational efficiency | Runtime, smoothing samples, training cost, and implementation overhead | Monte Carlo cost identified; experiments deferred |

Experiments and theory should be read against these axes separately.

---

## 5. Mathematical or Algorithmic Setup

This section contains two layers:

1. prerequisite clarifications needed before reading the formal RSCP/RSCP+ derivations;
2. the paper's Section 2.1 vanilla split conformal prediction foundations.

Later sections must not assume these objects are interchangeable.

### Local Navigation for the Foundation Layer

The intended reading order of this section is:

```text
prerequisite randomness and perturbation distinctions
        v
vanilla split conformal objects
        v
target coverage semantics
        v
rank-to-threshold derivation
        v
coverage event equivalence
        v
marginal probability interpretation
        v
efficiency and model-quality interpretation
        v
adversarial failure of clean rank symmetry
        v
ready state for Section 2.2 RSCP
```

This is not a second template. It is a map for the expanded Section 2.1 proof logic.

### Foundational Clarifications Before the Formal Derivations

#### Clarification 1 - Two Completely Different Perturbations

**Derived explanation.** This note uses different symbols for adversarial perturbation and randomized smoothing noise, even if the paper uses similar notation in nearby formulas. Notation adapted for conceptual clarity.

Adversarial perturbation is denoted by:

```math
\Delta.
```

The attacked input is:

```math
\widetilde X
=
X+\Delta,
```

with threat-model constraint:

```math
\|\Delta\|_2 \leq \epsilon.
```

The adversarial vector `\Delta` is not assumed to be Gaussian, not assumed to have mean zero, and not even necessarily assumed to be random. It can be a deterministic or worst-case vector chosen by an adversary for the input, subject only to the allowed perturbation set. In this paper's robustness geometry, that set is an `L_2` ball.

Gaussian smoothing noise is denoted by:

```math
Z.
```

The smoothing procedure samples:

```math
Z\sim \mathcal N(0,\sigma^2 I_p).
```

Therefore:

```math
\mathbb E[Z]=0,
\qquad
\mathrm{Cov}(Z)=\sigma^2 I_p.
```

This is randomness generated by the randomized smoothing procedure itself.

The key separation is:

```text
adversary:
worst-case \Delta

is not the same object as

randomized smoothing:
Gaussian Z
```

Confusing these objects breaks the probability semantics. The adversary describes a robustness threat model; `Z` describes an averaging distribution used by the algorithm.

#### Clarification 2 - What Exactly Does Adversarial Perturbation Break?

**Derived explanation.** It is too imprecise to write only that "adversarial perturbation breaks i.i.d." The more useful statement is about exchangeability of scores.

In the clean calibration/test setting:

```math
(X_1,Y_1),\ldots,(X_n,Y_n),(X_{n+1},Y_{n+1})
\overset{\mathrm{i.i.d.}}{\sim}
P_{XY}.
```

Thus calibration and test samples come from the same data-generating mechanism. The more fundamental conformal condition is exchangeability, not i.i.d. itself.

Suppose the adversary attacks only the test input:

```math
\widetilde X_{n+1}
=
T(X_{n+1})
=
X_{n+1}+\Delta(X_{n+1}).
```

The calibration samples remain clean, but the deployed model receives:

```math
(\widetilde X_{n+1},Y_{n+1}).
```

In general:

```math
P_{\widetilde X,Y}
\neq
P_{X,Y}.
```

The most direct break is therefore the identical-distribution / exchangeable-score structure required by the ordinary conformal rank argument.

Independence may survive. If `X_{n+1}` is independent of the calibration data and `\widetilde X_{n+1}=T(X_{n+1})` is only a deterministic measurable transformation of the test variable, then `\widetilde X_{n+1}` is still independent of the calibration data. Therefore the correct statement is:

> Test-time adversarial perturbation generally changes the distribution of the test score relative to the clean calibration scores, breaking the exchangeability / identical-score-distribution structure required by the ordinary conformal rank argument.

It should not be recorded as:

> adversarial perturbation necessarily breaks independence.

Mean shift is neither necessary nor sufficient for distribution shift. A simple example with a changed mean is:

```math
X\sim\mathcal N(0,1),
\qquad
\widetilde X=X+0.1
\sim\mathcal N(0.1,1).
```

But equal means do not imply equal distributions. For example:

```math
X\sim\mathcal N(0,1),
\qquad
\widetilde X\sim\mathcal N(0,4),
```

while:

```math
\mathbb E[X]
=
\mathbb E[\widetilde X]
=
0.
```

Thus:

```text
same mean
does not imply
same distribution
```

The conformal issue is not "the mean changed." The issue is that the calibration and test score variables may no longer be exchangeable.

#### Clarification 3 - Randomized Smoothing From a One-Dimensional Toy Example

**Pedagogical toy example.** Fix label `y` and consider the one-dimensional step score:

```math
S(x,y)
=
\begin{cases}
0, & x<0,\\
1, & x\geq 0.
\end{cases}
```

Near `x=0`, a tiny input move:

```math
-0.0001
\quad\text{to}\quad
+0.0001
```

can change the score:

```math
0
\quad\text{to}\quad
1.
```

The raw score is therefore extremely sensitive to small perturbations.

Gaussian averaging defines:

```math
S_{\mathrm{RS}}(x,y)
=
\mathbb E_Z[S(x+Z,y)],
\qquad
Z\sim\mathcal N(0,\sigma^2).
```

For the step score:

```math
S_{\mathrm{RS}}(x,y)
=
\mathbb P(x+Z\geq 0).
```

Then:

```math
S_{\mathrm{RS}}(x,y)
=
\mathbb P(Z\geq -x)
=
\Phi\left(\frac{x}{\sigma}\right).
```

The correct Gaussian CDF argument is `x/\sigma`, not `\sigma x`.

Now define the inverse-Gaussian transformed score:

```math
\widetilde S(x,y)
=
\Phi^{-1}
\left(
S_{\mathrm{RS}}(x,y)
\right).
```

For this toy example:

```math
\widetilde S(x,y)
=
\Phi^{-1}
\left(
\Phi\left(\frac{x}{\sigma}\right)
\right)
=
\frac{x}{\sigma}.
```

For an attacked input:

```math
\widetilde x=x+\Delta,
```

we obtain:

```math
\widetilde S(\widetilde x,y)
-
\widetilde S(x,y)
=
\frac{\Delta}{\sigma}.
```

Thus:

```math
|\Delta|\leq \epsilon
```

implies:

```math
\left|
\widetilde S(\widetilde x,y)
-
\widetilde S(x,y)
\right|
\leq
\frac{\epsilon}{\sigma}.
```

The toy example shows how:

```text
discontinuous step score
        v
Gaussian averaging
        v
inverse Gaussian CDF
        v
linear transformed score
```

directly exposes the source of the Lipschitz certificate.

#### Clarification 4 - General Randomized Smoothing Relation

**Paper statement / derived explanation.** At the paper level, the base score is assumed to be bounded:

```math
S(x,y)\in[0,1].
```

The population smoothed score is:

```math
S_{\mathrm{RS}}(x,y)
=
\mathbb E_{Z\sim\mathcal N(0,\sigma^2 I_p)}
\left[
S(x+Z,y)
\right].
```

The inverse-Gaussian transformed score is:

```math
\widetilde S(x,y)
=
\Phi^{-1}
\left[
S_{\mathrm{RS}}(x,y)
\right].
```

The randomized smoothing property used by the paper is more safely written without a quotient as:

```math
\left|
\widetilde S(\widetilde x,y)
-
\widetilde S(x,y)
\right|
\leq
\frac{
\|\widetilde x-x\|_2
}{\sigma}.
```

Equivalently, for `\widetilde x\neq x`, the quotient form is:

```math
\frac{
\left|
\widetilde S(\widetilde x,y)
-
\widetilde S(x,y)
\right|
}{
\|\widetilde x-x\|_2
}
\leq
\frac1{\sigma}.
```

The inequality form above avoids a division-by-zero edge case.

Therefore, if:

```math
\|\widetilde x-x\|_2\leq\epsilon,
```

then:

```math
\widetilde S(\widetilde x,y)
\leq
\widetilde S(x,y)
+
\frac{\epsilon}{\sigma}.
```

If clean membership requires:

```math
\widetilde S(x,y)\leq\tau,
```

then the attacked input is certified by the inflated threshold:

```math
\widetilde S(\widetilde x,y)
\leq
\tau+\frac{\epsilon}{\sigma}.
```

The ideal robust threshold adjustment is:

```math
\tau_{\mathrm{adj}}
=
\tau+\frac{\epsilon}{\sigma}.
```

This is still population-level RSCP logic. It has not yet solved finite Monte Carlo implementation error.

#### Clarification 5 - Where Monte Carlo Error Comes From

**Derived explanation.** For fixed realized `(x,y)`, define the Monte Carlo random variable:

```math
W
=
S(x+Z,y).
```

The target quantity is:

```math
\mu
=
\mathbb E_Z[W]
=
S_{\mathrm{RS}}(x,y).
```

For fixed `(x,y)`, `mu` is deterministic. If `(X,Y)` is itself random, then `S_{\mathrm{RS}}(X,Y)` is a random variable induced by the data-generating process. These are different probability layers.

This expectation is generally not available in closed form. The implementation samples:

```math
Z_1,\ldots,Z_{N_{\mathrm{MC}}}
\overset{\mathrm{i.i.d.}}{\sim}
\mathcal N(0,\sigma^2 I_p),
```

and computes:

```math
W_i
=
S(x+Z_i,y).
```

The Monte Carlo estimator is:

```math
\widehat S_{\mathrm{RS},N_{\mathrm{MC}}}(x,y)
=
\frac{1}{N_{\mathrm{MC}}}
\sum_{i=1}^{N_{\mathrm{MC}}}
S(x+Z_i,y).
```

The distinction is:

```text
S_{\mathrm{RS}}
=
population expectation

\widehat S_{\mathrm{RS},N_{\mathrm{MC}}}
=
finite-sample random estimator
```

For finite `N_MC`, the estimator is random and generally not equal to the population expectation.

#### What Is a Monte Carlo Estimator?

**Derived explanation.** In general:

```math
\mu
=
\mathbb E_{Z\sim p}[f(Z)]
=
\int f(z)p(z)\,dz.
```

If the integral is high-dimensional or analytically unavailable, sample:

```math
Z_1,\ldots,Z_N
\overset{\mathrm{i.i.d.}}{\sim}
p,
```

and construct:

```math
\widehat\mu_N
=
\frac1N
\sum_{i=1}^{N}
f(Z_i).
```

This sample average is the Monte Carlo estimator.

Important semantics:

* the target `mu` is a deterministic population quantity once the distribution and function are fixed;
* the estimator `hat mu_N` is a random variable;
* each resampling of `Z_1,...,Z_N` can produce a different estimate;
* finite-sample error does not mathematically disappear just because `N` is large;
* a formal certificate must explicitly control this randomness.

#### LLN vs CLT vs Concentration

The Law of Large Numbers answers:

> Does the estimator converge to the expectation?

For randomized smoothing:

```math
\widehat S_{\mathrm{RS},N_{\mathrm{MC}}}(x,y)
\xrightarrow{\mathrm{a.s.}}
S_{\mathrm{RS}}(x,y)
```

as:

```math
N_{\mathrm{MC}}\to\infty.
```

The Central Limit Theorem answers:

> What is the approximate large-sample distribution of the estimation error?

Under appropriate conditions:

```math
\frac{
\sqrt{N_{\mathrm{MC}}}
\left(
\widehat S_{\mathrm{RS},N_{\mathrm{MC}}}(x,y)
-
S_{\mathrm{RS}}(x,y)
\right)
}{
\sqrt{\mathrm{Var}(W)}
}
\xrightarrow{d}
\mathcal N(0,1)
```

when `\mathrm{Var}(W)>0`. This does not say the finite estimator is exactly equal to the expectation.

The paper needs finite-sample, high-probability error control over Monte Carlo sampling randomness. Since:

```math
0\leq S(x,y)\leq 1,
```

each Monte Carlo variable `W_i` is bounded. Hoeffding's inequality gives:

```math
\mathbb P_Z
\left(
\widehat S_{\mathrm{RS},N_{\mathrm{MC}}}(x,y)
-
S_{\mathrm{RS}}(x,y)
\geq t
\right)
\leq
e^{-2N_{\mathrm{MC}}t^2}.
```

The reverse tail is controlled analogously. Setting:

```math
e^{-2N_{\mathrm{MC}}t^2}
=
\beta
```

gives:

```math
t
=
\sqrt{
\frac{-\ln\beta}
{2N_{\mathrm{MC}}}
}.
```

Define:

```math
b_{\mathrm{Hoef}}(\beta)
=
\sqrt{
\frac{-\ln\beta}
{2N_{\mathrm{MC}}}
}.
```

The implementation trade-off is:

```text
N_MC increases
        v
error radius decreases
        v
certificate tightens
        v
computation increases
```

This is only concentration background here. Theorem 1 is not derived in this update.

#### Standard Randomized Smoothing Implementation

The practical pipeline is:

```text
Input x and candidate label y
        v
sample Gaussian noises Z_1,...,Z_N
        v
construct x+Z_i
        v
run predictor / compute score
        v
S(x+Z_i,y)
        v
average
        v
\widehat S_{\mathrm{RS},N_{\mathrm{MC}}}(x,y)
```

There are two separate steps:

```text
Step 1
Monte Carlo approximation

Step 2
finite-sample concentration / confidence control
```

The Section 3 criticism, at the conceptual level, is that original RSCP uses Monte Carlo approximation in practice but does not fully include finite Monte Carlo estimation error in the implementation-level robustness certificate.

#### Original RSCP Gap Without Proving RSCP+

The ideal theorem operates on:

```math
S_{\mathrm{RS}}(x,y)
=
\mathbb E_Z
S(x+Z,y).
```

The actual computer operates on:

```math
\widehat S_{\mathrm{RS},N_{\mathrm{MC}}}(x,y).
```

For finite:

```math
N_{\mathrm{MC}},
```

one generally has:

```math
\widehat S_{\mathrm{RS},N_{\mathrm{MC}}}
\neq
S_{\mathrm{RS}}.
```

Therefore the proof pattern:

```text
prove property of S_{\mathrm{RS}}
        v
silently substitute \widehat S_{\mathrm{RS},N_{\mathrm{MC}}}
        v
claim the same formal certificate
```

is not valid without an explicit estimator-to-population argument.

#### Why Not Simply Add a Per-Sample Hoeffding Bound?

**Derived explanation from Appendix A.1 motivation.** A naive idea is to put a high-confidence Monte Carlo interval around every calibration score and then calibrate. The difficulty is that the conformal threshold depends on the entire calibration score set.

If each calibration estimate is controlled with confidence `0.999`, and the calibration size is `10,000`, then requiring all pointwise events to hold simultaneously gives a joint confidence on the order of:

```math
(0.999)^{10000}
\approx
4.5\times 10^{-5}.
```

This example does not prove that every simultaneous-bound method must fail this way. It shows that naive pointwise Monte Carlo confidence control over the entire calibration set can become catastrophically conservative.

The design motivation is therefore:

```text
do not first try to exactly recover population S_{\mathrm{RS}}
and then run conformal calibration

instead

use the computable randomized estimator \widehat S_{\mathrm{RS},N_{\mathrm{MC}}}
directly as the conformity score
```

This motivates RSCP+, but the full Theorem 1 proof remains deferred.

#### Randomness Source Bookkeeping

The proof objects must be separated as follows.

**Data randomness.**

```math
(X_i,Y_i)
```

comes from the data-generating process and supports the exchangeability/rank argument.

**Score randomization.** APS contains:

```math
u\sim U[0,1].
```

This is randomization inside the conformity score.

**Gaussian smoothing randomness.**

```math
Z\sim\mathcal N(0,\sigma^2 I_p)
```

defines the population smoothed score.

**Monte Carlo randomness.**

```math
Z_1,\ldots,Z_{N_{\mathrm{MC}}}
```

are the finite smoothing samples actually used by the implemented estimator.

**Adversarial perturbation.**

```math
\Delta
```

comes from a bounded threat set and is not necessarily random.

Later RSCP+ arguments must also distinguish population smoothing randomness from finite Monte Carlo sampling randomness.

### Section 2.1 - Vanilla Split Conformal Prediction

#### Central Causal Chain

**Derived explanation.** The basic split conformal pipeline should be read as a sequence of different objects, not as one vague "confidence" number:

```text
user-specified target coverage 1-alpha
        v
required pooled rank k
        v
calibration empirical quantile / threshold tau
        v
candidate-label score comparison
        v
prediction set C(x)
        v
true-label membership event
        v
marginal finite-sample coverage
```

Each arrow has a different mathematical meaning. The user chooses `1-\alpha`. The rank `k` is determined by finite-sample coverage arithmetic. The threshold `tau` is calibrated from held-out true-label scores after the predictive model and score function are fixed. The prediction set is built by comparing each candidate-label score to `tau`. Coverage follows only after translating true-label membership into a true-label score event.

#### Target Coverage Is a Requirement

The quantity:

```math
1-\alpha
```

is a user-specified target marginal coverage level chosen before running the conformal procedure. For example:

```math
1-\alpha=0.8.
```

The intended coverage statement is:

```math
\mathbb P
\left(
Y_{\mathrm{new}}
\in
C(X_{\mathrm{new}})
\right)
\geq
0.8.
```

This is not the classifier accuracy statement:

```math
\mathbb P
\left(
\widehat Y=Y
\right)
\geq
0.8.
```

It is also not a statement that, for a single already-realized test input, the prediction set has "80 percent confidence." Before realization, the theorem controls a population probability over future calibration/test randomness. After realization, either the realized true label is in the realized set or it is not.

#### Target Coverage Versus Calibration Threshold

The target `1-\alpha` is specified by the user. The threshold `\tau` is a data-dependent calibrated quantity determined jointly by:

* the fixed predictive model;
* the chosen score function;
* the held-out calibration sample;
* the chosen target coverage level.

The threshold is not a trained model parameter. More precisely:

> `\tau` is calibrated from held-out calibration scores after the predictive model has been fixed.

For fixed calibration scores:

```math
1-\alpha \uparrow
\quad\Longrightarrow\quad
k \uparrow.
```

Since empirical order statistics are monotone in the requested rank:

```math
k_1\leq k_2
\quad\Longrightarrow\quad
S_{(k_1)}
\leq
S_{(k_2)}.
```

Thus the calibrated threshold is weakly nondecreasing as requested coverage increases.

With the paper's nonconformity convention:

```math
C(x;\tau)
=
\{
y\in[K]:
S(x,y)\leq\tau
\},
```

we have:

```math
\tau_1\leq\tau_2
\quad\Longrightarrow\quad
C(x;\tau_1)
\subseteq
C(x;\tau_2).
```

Therefore:

```text
higher requested coverage
        v
more permissive threshold
        v
weakly larger prediction set
        v
lower informativeness
```

This is the most basic version of the coverage-efficiency tension that later robust methods must confront.

#### Dataset and Split

**Paper statement / derived explanation.** Let the full realized dataset be:

```math
D
=
\{(x_i,y_i)\}_{i=1}^{N_{\mathrm{all}}},
```

where:

```math
x_i\in\mathbb R^p,
\qquad
y_i\in[K].
```

Uppercase `(X_i,Y_i)` denotes random variables. Lowercase `(x_i,y_i)` denotes realized data. This distinction matters because conformal guarantees are probability statements over the random data-generating process, while implementation computes with realized scores.

Notation convention for this note: `N_{\mathrm{all}}` denotes the full realized dataset size, while `n` denotes the calibration-set size in the split conformal proof. This avoids overloading the rank argument's `n+1` pooled calibration-plus-test sample size.

If following the paper's shorter dataset notation, the full dataset may also be written as:

```math
D
=
\{(x_i,y_i)\}_{i=1}^{n}.
```

In the formulas below, however, `n` is reserved for the calibration size.

Split conformal prediction separates:

```text
training set
        v
fit classifier

calibration set
        v
estimate score threshold

test sample
        v
construct set-valued prediction
```

The calibration data are not used to keep training the classifier. They are held out to estimate the empirical score threshold. Reusing calibration data for model fitting would generally entangle the fitted model with the calibration scores and invalidate the simple exchangeability proof unless a different analysis is supplied.

#### Classifier Output

The classifier output is:

```math
\widehat\pi(x):
\mathbb R^p
\to
[0,1]^K.
```

The component:

```math
\widehat\pi_k(x)
```

is the model's predictive score or probability-like output for class `k`.

When treated as a class-probability vector, the components also satisfy:

```math
\sum_{k=1}^{K}
\widehat\pi_k(x)
=
1.
```

Conformal coverage is not the same as classifier probability calibration. The proof does not require:

```math
\widehat\pi_y(x)
=
\mathbb P(Y=y\mid X=x).
```

The model quality affects set size, but the validity proof comes from exchangeability of scores.

#### Non-Conformity Score

A non-conformity score is a scalar statistic:

```math
S(x,y):
\mathbb R^p\times[K]
\to
\mathbb R.
```

The paper uses the convention:

```text
smaller score
=
more conforming
```

Thus prediction sets use:

```math
S(x,k)\leq\tau.
```

The score is not a ground-truth probability. It is a scalar statistic used to establish a calibration/test rank ordering.

#### HPS

The homogeneous prediction score is:

```math
S_{\mathrm{HPS}}(x,y)
=
1-\widehat\pi_y(x).
```

Example. If:

```math
\widehat\pi(x)
=
(0.10,0.70,0.20),
```

then for candidate label `2`:

```math
S_{\mathrm{HPS}}(x,2)
=
1-0.70
=
0.30.
```

For candidate label `1`:

```math
S_{\mathrm{HPS}}(x,1)
=
1-0.10
=
0.90.
```

Higher model probability gives lower nonconformity.

For HPS, membership in the conformal set has a simple equivalent form:

```math
S_{\mathrm{HPS}}(x,y)
\leq
\tau
```

if and only if:

```math
1-\widehat\pi_y(x)
\leq
\tau,
```

which is equivalent to:

```math
\widehat\pi_y(x)
\geq
1-\tau.
```

Thus:

```text
tau larger
        v
required classifier score 1-tau becomes smaller
        v
more labels satisfy membership
        v
prediction set becomes larger
```

This makes the efficiency effect visible in one line of algebra. For a high threshold, even labels with modest classifier scores can enter the prediction set.

#### APS

The adaptive prediction score is a randomized score. Let:

```math
U
\sim
\mathrm{Uniform}[0,1],
```

with `U` sampled independently when the randomized score is evaluated. Then:

```math
S_{\mathrm{APS}}(x,y;U)
=
\sum_{y'\in[K]}
\widehat\pi_{y'}(x)
\mathbf 1
\{
\widehat\pi_{y'}(x)
>
\widehat\pi_y(x)
\}
+
\widehat\pi_y(x)U.
```

For a realized draw `u`, the realized APS score is:

```math
S_{\mathrm{APS}}(x,y;u).
```

APS does not only inspect the candidate label's own probability. It accumulates the probability mass of labels ranked ahead of the candidate and then randomizes within the candidate's own mass.

The APS random variable is a useful precursor for this paper because it shows that randomized conformity scores can still be conformally valid, provided the same randomized scoring procedure is applied symmetrically to calibration and test examples and the required independence/exchangeability conditions hold.

#### Calibration Scores

For a calibration sample:

```math
(x_i,y_i),
```

compute the true-label score:

```math
s_i
=
S(x_i,y_i).
```

The corresponding calibration score random variable is:

```math
S_i
=
S(X_i,Y_i).
```

The calibration distribution uses true-label scores, not all candidate-label scores mixed together. The target coverage event is:

```math
Y_{n+1}\in C(X_{n+1}),
```

which is equivalent to a statement about:

```math
S(X_{n+1},Y_{n+1}).
```

Therefore calibration must estimate the distribution of the same random variable: the score of the true label under a fresh example.

This is why the calibration stage does not pool:

```math
S(X_i,1),\ldots,S(X_i,K)
```

into the same empirical distribution. The theorem is not trying to guarantee that every candidate-label score behaves like a true-label score. It is trying to guarantee that the random variable:

```math
S(X_{\mathrm{new}},Y_{\mathrm{new}})
```

falls below the threshold with high probability. Wrong-label scores matter for efficiency because they determine how many false candidate labels also pass the threshold, but they are not the calibration target for marginal coverage.

#### Prediction Set

Given a threshold `tau`, the paper's prediction set form is:

```math
C(x_{n+1};\tau)
=
\{
k\in[K]
:
S(x_{n+1},k)\leq\tau
\}.
```

The proof bridge is the event equivalence:

```math
Y_{n+1}
\in
C(X_{n+1};\tau)
```

if and only if:

```math
S(X_{n+1},Y_{n+1})
\leq
\tau.
```

Equivalently, as events:

```math
\{
Y_{n+1}
\in
C(X_{n+1};\tau)
\}
=
\{
S(X_{n+1},Y_{n+1})
\leq
\tau
\}.
```

This equivalence is why conformal coverage can be proved by controlling the rank of the test true-label score.

#### Exchangeability

The i.i.d. setup is:

```math
(X_1,Y_1),\ldots,(X_{n+1},Y_{n+1})
\overset{\mathrm{i.i.d.}}{\sim}
P_{XY}.
```

The deeper condition is exchangeability. Random variables:

```math
Z_1,\ldots,Z_{n+1}
```

are exchangeable if, for every permutation `rho`:

```math
(Z_1,\ldots,Z_{n+1})
\overset{d}{=}
(Z_{\rho(1)},\ldots,Z_{\rho(n+1)}).
```

i.i.d. variables imply exchangeability, but exchangeability does not necessarily imply independence. The conformal rank proof uses permutation symmetry.

#### Rank Argument

Define calibration true-label scores:

```math
S_i
=
S(X_i,Y_i).
```

for:

```math
i=1,\ldots,n.
```

Define the future true-label score:

```math
S_{n+1}
=
S(X_{n+1},Y_{n+1}).
```

If:

```math
S_1,\ldots,S_n,S_{n+1}
```

are exchangeable, then the test score `S_{n+1}` has no special index status in the pooled sample.

**Pedagogical no-tie argument.** In a continuous-score or suitably randomized tie-breaking setting, define:

```math
R
=
\mathrm{rank}
\left(
S_{n+1};
S_1,\ldots,S_n,S_{n+1}
\right).
```

Here `R` is the ascending pooled rank of the test true-label score among the `n+1` scores. Rank symmetry gives:

```math
\mathbb P(R=r)
=
\frac{1}{n+1},
\qquad
r=1,\ldots,n+1.
```

Therefore:

```math
\mathbb P
\left(
R\leq k
\right)
=
\frac{k}{n+1}.
```

The target is:

```math
\mathbb P(R\leq k)
\geq
1-\alpha.
```

Thus:

```math
\frac{k}{n+1}
\geq
1-\alpha.
```

Equivalently:

```math
k
\geq
(n+1)(1-\alpha).
```

Since `k` is an integer, the smallest feasible choice is:


```math
k
=
\left\lceil
(n+1)(1-\alpha)
\right\rceil.
```

The `+1` is not an ad-hoc correction. It comes from the pooled rank space:

```text
n calibration observations
+
1 future test observation
```

With ties, the equality `\mathbb P(R\leq k)=k/(n+1)` should not be treated as a universal theorem. The Appendix A.2 indicator proof below gives the conservative finite-sample inequality needed for coverage without relying on this no-tie simplification.

#### Why the k-th Calibration Score Becomes the Threshold

Let the calibration order statistics be:

```math
S_{(1)}
\leq
S_{(2)}
\leq
\cdots
\leq
S_{(n)}.
```

In the usual nontrivial case:

```math
k\leq n,
```

define:

```math
\widehat\tau_\alpha
=
S_{(k)}.
```

If the pooled test rank satisfies:

```math
R\leq k,
```

then the test score cannot be strictly larger than the `k`th calibration order statistic in the no-tie pedagogical setting. Therefore:

```math
S_{n+1}
\leq
\widehat\tau_\alpha.
```

Consequently:

```math
\mathbb P
\left(
S_{n+1}\leq\widehat\tau_\alpha
\right)
\geq
\mathbb P(R\leq k)
\geq
1-\alpha.
```

Finally, using the prediction-set event equivalence:

```math
\mathbb P
\left(
Y_{n+1}\in C(X_{n+1};\widehat\tau_\alpha)
\right)
\geq
1-\alpha.
```

This is the intuitive rank proof. Appendix A.2 uses a more formal indicator argument that also handles ties and the quantile convention carefully.

#### Finite-Sample Quantile Correction

The conformal quantile correction:

```math
(1-\alpha)
\left(
1+\frac1n
\right)
```

is not a heuristic. The calibration sample has only `n` scores, but the rank argument is over the pooled `n+1` calibration-plus-test scores.

Let:

```math
q
=
1-\alpha.
```

The target order statistic index is:

```math
k
=
\left\lceil
(n+1)q
\right\rceil.
```

When expressed as an empirical quantile level over only the `n` calibration scores, this corresponds to:

```math
q
\left(
1+\frac1n
\right).
```

The paper writes this threshold using `Q_{1-\alpha}`, but the empirical quantile level is the finite-sample corrected level:

```math
(1-\alpha)
\left(
1+\frac1n
\right).
```

The identity:

```math
n
(1-\alpha)
\left(
1+\frac1n
\right)
=
(n+1)(1-\alpha)
```

shows that the paper's empirical quantile convention and the pooled-rank index:

```math
\left\lceil
(n+1)(1-\alpha)
\right\rceil
```

are two expressions of the same finite-sample correction.

The split conformal threshold can be written as:

```math
\widehat\tau_\alpha
=
\inf
\left\{
s\in\mathbb R:
\frac1n
\sum_{i=1}^{n}
\mathbf 1
\{S_i\leq s\}
\geq
(1-\alpha)
\left(
1+\frac1n
\right)
\right\}.
```

Equivalently, when the quantile level is at most one, `\widehat\tau_\alpha` is the `k`th order statistic of the calibration scores:

```math
S_{(1)}
\leq
\cdots
\leq
S_{(n)},
\qquad
\widehat\tau_\alpha
=
S_{(k)}.
```

**Derived finite-sample clarification.** If:

```math
\left\lceil
(n+1)(1-\alpha)
\right\rceil
>
n,
```

then the calibration sample has no such order statistic. The usual conservative split-conformal convention is:

```math
\widehat\tau_\alpha
=
+\infty.
```

This returns the full label set and gives the most conservative result. It also exposes a finite-sample resolution limit:

```text
small n_cal
+
1-alpha close to 1
        v
required rank may exceed n
        v
nontrivial finite-sample calibration may be impossible
```

This is not the paper's main contribution. It is a basic finite-sample consequence of distribution-free conformal calibration.

Numerical example:

```text
n_cal = 9
1-alpha = 0.9
```

Then:

```math
k
=
\left\lceil
10\times0.9
\right\rceil
=
9.
```

The threshold is the 9th calibration order statistic. The test score only needs to avoid falling beyond the conformal rank cutoff in the pooled 10-score system to obtain at least 90 percent rank coverage.

#### Marginal vs Conditional Coverage

The conformal guarantee is marginal:

```math
\mathbb P
\left(
Y_{n+1}
\in
C(X_{n+1})
\right)
\geq
1-\alpha.
```

In split conformal prediction, the fitted model is treated as fixed before calibration. The probability statement is over the random calibration/test examples and any auxiliary score randomization used symmetrically. If the training set is also random, the unconditional statement can include training randomness as well, but the rank mechanism itself is driven by exchangeability of the calibration and test true-label scores after the model has been fixed.

It does not automatically imply pointwise conditional coverage:

```math
\mathbb P
\left(
Y_{n+1}
\in
C(X_{n+1})
\mid
X_{n+1}=x
\right)
\geq
1-\alpha
```

for every `x`.

To make the probability semantics precise, define:

```math
I
=
\mathbf 1
\{
Y\in C(X)
\}.
```

Then:

```math
\mathbb E[I]
=
\mathbb P
\left(
Y\in C(X)
\right).
```

Thus marginal coverage can also be written as:

```math
\mathbb E[I]
\geq
1-\alpha.
```

This does not mean the conformal theorem is caused by a sample mean converging to an expectation. The conformal rank theorem lower-bounds the population coverage probability. The law of large numbers only explains why an empirical average over many independent test samples can approximate that probability.

There are three different objects:

**A. Single realized event.** Once:

```math
(x,y,C(x))
```

are all realized:

```math
\mathbf 1
\{
y\in C(x)
\}
\in
\{0,1\}.
```

The outcome is covered or not covered.

**B. Population marginal coverage.** For a future random pair `(X,Y)`:

```math
\mathbb P
\left(
Y\in C(X)
\right)
```

is the probability object controlled by the conformal theorem.

**C. Empirical coverage over many test samples.** For `m` test samples:

```math
\widehat{\mathrm{cov}}_m
=
\frac1m
\sum_{j=1}^{m}
\mathbf 1
\{
Y_j\in C(X_j)
\}.
```

Under appropriate conditions:

```math
\widehat{\mathrm{cov}}_m
\to
\mathbb P
\left(
Y\in C(X)
\right).
```

The responsibilities are different:

```text
conformal rank theorem
        v
lower-bounds the population coverage probability

LLN
        v
explains why large empirical test averages approximate that probability
```

They should not be reversed.

Why is it called marginal? By the law of total probability:

```math
\mathbb P
\left(
Y\in C(X)
\right)
=
\mathbb E_X
\left[
\mathbb P
\left(
Y\in C(X)
\mid X
\right)
\right].
```

The marginal probability averages conditional coverage over the distribution of `X`. Good marginal coverage can therefore hide poor reliability in a low-probability or difficult region.

For example, suppose:

```math
\mathbb P(X\in A)=0.8,
```

and:

```math
\mathbb P
\left(
Y\in C(X)
\mid X\in A
\right)
=
0.99.
```

For a harder region:

```math
\mathbb P(X\in B)=0.2,
```

and:

```math
\mathbb P
\left(
Y\in C(X)
\mid X\in B
\right)
=
0.60.
```

The overall marginal coverage is:

```math
0.8\times0.99
+
0.2\times0.60
=
0.912.
```

The marginal coverage is `91.2` percent even though the hard region has only `60` percent coverage.

Thus:

```text
good marginal coverage
does not imply
good conditional / subgroup coverage everywhere
```

A concrete point can still have a conditional probability before the label is realized. If only `X=x` is fixed and `Y` is still random, then:

```math
\mathbb P
\left(
Y\in C(x)
\mid X=x
\right)
```

is meaningful. After both `x` and `y` are realized, the indicator:

```math
\mathbf 1
\{
y\in C(x)
\}
```

is only `0` or `1`. The distinction is:

```text
probability before realization
is different from
indicator after realization
```

Marginal validity does not imply uniform subgroup or conditional reliability. This distinction is critical for high-stakes and dynamic-system applications, where failure can concentrate in particular regimes.

#### Why Classifier Correct Specification Is Not Required

The coverage proof does not rely on:

```math
\widehat\pi_y(x)
=
\mathbb P(Y=y\mid X=x).
```

It relies on exchangeability of calibration/test true-label scores. Therefore:

```text
model correctness
is not the source of conformal validity
```

But:

```text
score/model quality
strongly affects prediction-set efficiency
```

Thus:

```text
validity source
is different from
efficiency source
```

#### Why Score Choice Matters

Two scores can satisfy the same marginal coverage guarantee while producing very different prediction-set sizes.

Good score geometry tends to separate true labels from wrong labels. Then fewer wrong-label candidate scores fall below the threshold, producing smaller prediction sets.

Poor score geometry creates overlap between true-label and wrong-label scores. Then many candidate labels fall below the threshold, producing larger prediction sets.

#### Bad Model Does Not Break Validity - It Hurts Efficiency

Consider an informative model whose true-label nonconformity scores are often small:

```text
0.10
0.15
0.20
0.25
```

For a given target coverage level, the calibrated threshold `\tau` may be relatively low. Then only a small number of candidate labels satisfy:

```math
S(x,y)\leq\tau.
```

The prediction sets are small and informative.

Now consider a weak model whose true-label scores are often larger:

```text
0.60
0.75
0.85
0.90
```

To achieve the same target:

```math
1-\alpha,
```

the calibration threshold must allow larger true-label scores. This means `\tau` becomes larger, and wrong labels are more likely to satisfy:

```math
S(x,y)\leq\tau.
```

The prediction sets inflate.

The correct interpretation is:

```text
bad base model
does not necessarily destroy conformal marginal validity

instead

bad base model / bad score geometry
        v
larger calibrated threshold or poorer class separation
        v
larger prediction sets
        v
worse efficiency
```

This is not a vicious cycle. It is conformal prediction honestly exposing the informational weakness of the underlying predictor.

#### Validity and Intelligence Are Different

These questions are distinct:

```text
Conformal validity asks:
Does the true label enter sufficiently often?

Classifier quality asks:
Can the model discriminate the correct class from alternatives?

Conformal efficiency asks:
How many alternatives must be retained to achieve validity?
```

In the extreme case:

```math
C(x)
=
[K]
```

gives:

```math
\mathbb P
\left(
Y\in C(X)
\right)
=
1.
```

But its informativeness is essentially zero. This is the foundation-layer predecessor of the paper's Challenge 2: robust validity is not enough if the prediction set becomes uninformative.

This is a conceptual precursor for PTT, but the PTT formulas and theory are not filled in this update.

#### Formal Transition to Section 2.2 - Why Clean Rank Coverage Is Not Enough

In the clean conformal setting:

```math
S_i
=
S(X_i,Y_i),
\qquad
i=1,\ldots,n+1.
```

The sequence:

```math
S_1,\ldots,S_n,S_{n+1}
```

has the exchangeability needed for the rank proof.

After a test-time attack:

```math
\widetilde X_{n+1}
=
X_{n+1}
+
\Delta.
```

The actual deployed test score becomes:

```math
\widetilde S_{n+1}
=
S(\widetilde X_{n+1},Y_{n+1}).
```

The calibration scores remain:

```math
S(X_i,Y_i),
\qquad
i=1,\ldots,n.
```

Therefore the sequence:

```math
S(X_1,Y_1),
\ldots,
S(X_n,Y_n),
S(\widetilde X_{n+1},Y_{n+1})
```

is generally no longer exchangeable. Independence may survive, but the identical-distribution / permutation-symmetry structure generally does not.

**Derived probabilistic interpretation.** If:

```math
\widetilde X
=
T(X),
```

then the attacked test distribution is the push-forward:

```math
P_{\widetilde X,Y}
=
(T,\mathrm{id})_{\#}P_{X,Y}.
```

In general:

```math
P_{\widetilde X,Y}
\neq
P_{X,Y}.
```

Therefore one also generally has:

```math
\mathcal L
\left(
S(\widetilde X,Y)
\right)
\neq
\mathcal L
\left(
S(X,Y)
\right).
```

This is the formal distribution-level reason adversarial perturbation breaks the ordinary true-label score rank structure.

#### HPS Attack Toy Example

**Pedagogical toy example.** Suppose the calibration true-label scores are:

```text
0.10, 0.15, 0.20, 0.25, 0.30
```

and:

```math
\tau=0.30.
```

For a clean test point, suppose:

```math
\widehat\pi_y(x)
=
0.82.
```

Then:

```math
S_{\mathrm{HPS}}(x,y)
=
1-0.82
=
0.18.
```

Since:

```math
0.18
\leq
0.30,
```

the true label is included.

After attack, suppose:

```math
\widehat\pi_y(\widetilde x)
=
0.30.
```

Then:

```math
S_{\mathrm{HPS}}(\widetilde x,y)
=
0.70.
```

Since:

```math
0.70
>
0.30,
```

the true label is excluded.

This toy example does not prove that every attack works this way. It only shows:

```text
clean calibration threshold
does not itself control
adversarially inflated test scores
```

#### Exact Bridge Needed for RSCP

Vanilla conformal gives a clean-score statement:

```math
S_{\mathrm{clean}}
\leq
\tau
```

with the desired marginal probability under exchangeability. For an adversarial test score, the ordinary rank argument no longer directly applies.

RSCP's strategy is not to claim the attacked score remains exchangeable with clean calibration scores. Instead, the needed bridge has the abstract form:

```math
S_{\mathrm{rob}}(\widetilde x,y)
\leq
S_{\mathrm{rob}}(x,y)
+
M_\epsilon.
```

Then:

```math
S_{\mathrm{rob}}(x,y)
\leq
\tau
```

implies:

```math
S_{\mathrm{rob}}(\widetilde x,y)
\leq
\tau+M_\epsilon.
```

Define:

```math
\tau_{\mathrm{adj}}
=
\tau+M_\epsilon.
```

The desired Section 2.2 bridge is:

```text
clean conformal membership
+
certified score-inflation bound
        v
adversarial membership
```

This is where the next formal reading unit should begin. Even though the prerequisite section already previewed randomized smoothing intuition, the formal Eq. (6)-(11) derivation and the exact paper-level expression for `M_\epsilon` remain deferred to Section 2.2.

---

## 6. Method: Step-by-Step Logic

**Current reading roadmap and status.** This is not a completed-paper record.

1. Stage 1 - Vanilla split conformal prediction: first-pass research understanding substantially complete.
2. Stage 2 - Randomized smoothed conformal prediction: ready / beginning; prerequisite intuition only; full Section 2.2 deferred.
3. Stage 3 - RSCP theory-implementation gap: conceptual motivation recorded; formal details deferred.
4. Stage 4 - RSCP+ and Theorem 1: not completed.
5. Stage 5 - Corollary 2: not completed; Appendix A.2 proposition recorded only as a future dependency.
6. Stage 6 - Concentration bounds: Hoeffding background recorded; Empirical Bernstein deferred.
7. Stage 7 - Robust prediction-set inflation: conceptual threshold inflation recorded; full robust coverage transfer deferred.
8. Stage 8 - PTT: rank transformation: not completed.
9. Stage 9 - PTT: sigmoid transformation: not completed.
10. Stage 10 - PTT theory and failure modes: not completed.
11. Stage 11 - Robust conformal training: not completed.
12. Stage 12 - Experiments: not read.
13. Stage 13 - Ablations: not read.
14. Stage 14 - Research-level critique: not final.
15. Stage 15 - Dynamic-system transfer: boundary-level interpretation recorded, not a method transfer proof.

### Current Method Map

The method logic to keep in mind is:

```text
clean split conformal
        v
rank coverage from exchangeable true-label scores
        v
test-time adversarial input changes test-score distribution
        v
randomized smoothing creates a perturbation-controlled population score
        v
finite Monte Carlo estimator is the implemented score
        v
RSCP+ should certify the implemented randomized score
        v
robust threshold inflation can hurt set efficiency
        v
PTT / RCT are later efficiency mechanisms
```

The clean split conformal part is now developed through target coverage semantics, finite-sample rank derivation, empirical quantile construction, marginal coverage meaning, efficiency interpretation, and the adversarial failure transition. Section 2.2 is ready to begin but has not been completed.

---

## 7. Key Equations and Derivations

This section records derivations that are now structurally important. It intentionally does not complete Theorem 1 or the later robust coverage proof.

### Vanilla CP Coverage Proof Chain

The Section 2.1 proof skeleton is:

```text
exchangeability
        v
rank symmetry
        v
finite-sample corrected k
        v
calibration threshold tau
        v
population true-label score event controlled at level 1-alpha
        v
membership-event equivalence
        v
marginal membership coverage at level 1-alpha
```

In formal notation, the central event equivalence is:

```math
\{
Y_{n+1}\in C(X_{n+1};\tau)
\}
=
\{
S(X_{n+1},Y_{n+1})\leq\tau
\}.
```

The rank argument controls the right-hand event. The conformal prediction guarantee is then transferred to the left-hand membership event by definition of the prediction set.

### Why the Intuitive Rank Proof and the Appendix A.2 Proof Both Stay

**Derived explanation.** The intuitive rank proof explains why exchangeability creates coverage: the test score is not special among the pooled calibration-plus-test scores.

The Appendix A.2 indicator proof provides the formal finite-sample argument, including the quantile convention and a randomized Monte Carlo score. These are not duplicate content. A research-level note should preserve both:

```text
intuition
+
formal proof
```

The intuition helps reconstruct the idea months later. The formal proof prevents mistakes about the empirical quantile and randomized-score object.

### Appendix A.2 Proof Skeleton - Randomized Score Coverage

#### Lemma A.1 - Randomized Monte Carlo Scores Are i.i.d.

**Paper statement / derived proof mechanism.** Define the implemented Monte Carlo score:

```math
S_i
=
\widehat S_{\mathrm{RS},N_{\mathrm{MC}}}(X_i,Y_i)
=
\frac1{N_{\mathrm{MC}}}
\sum_{j=1}^{N_{\mathrm{MC}}}
S(X_i+Z_{ij},Y_i).
```

Conditions:

* `(X_i,Y_i)` are i.i.d.;
* all `Z_{ij}` are i.i.d. Gaussian;
* Monte Carlo noises are independent of the data;
* the same randomized scoring procedure is used for calibration and test.

Proof skeleton:

```math
T_i
=
\left(
(X_i,Y_i),
Z_{i1},\ldots,Z_{iN_{\mathrm{MC}}}
\right).
```

Under the stated conditions:

```math
T_1,\ldots,T_{n+1}
```

are i.i.d. The score is the same measurable function of each tuple:

```math
S_i
=
g(T_i).
```

Therefore:

```math
S_1,\ldots,S_{n+1}
```

are i.i.d., hence exchangeable.

The key implementation point is that randomized Monte Carlo scores can be conformal scores if the randomization is part of the score-generating procedure and is applied symmetrically. A weaker exchangeability condition would also be enough for conformal coverage, but the lemma uses i.i.d. conditions to establish it cleanly.

#### Lemma A.2 - Empirical Quantile Gives Coverage

**Paper statement / formal proof skeleton.** Let:

```math
S_1,\ldots,S_n,S_{n+1}
```

be i.i.d. or exchangeable scores, and let `q` be the target coverage level.

Define:

```math
I_i
=
\mathbf 1
\left\{
\left|
\{j:S_i>S_j\}
\right|
\geq
\left\lceil(n+1)q\right\rceil
\right\}.
```

Property 1:

```math
\mathbb E[I_1]
=
\cdots
=
\mathbb E[I_{n+1}].
```

This follows from i.i.d. / exchangeability symmetry.

Property 2:

```math
\sum_{i=1}^{n+1} I_i
\leq
(n+1)-\left\lceil(n+1)q\right\rceil.
```

Reason: the smallest first:

```math
\left\lceil(n+1)q\right\rceil
```

scores cannot each have at least that many scores strictly smaller than themselves.

Combining the two properties:

```math
(n+1)\mathbb E[I_{n+1}]
=
\mathbb E
\left[
\sum_{i=1}^{n+1}I_i
\right]
\leq
(n+1)(1-q).
```

Therefore:

```math
\mathbb E[I_{n+1}]
\leq
1-q.
```

Now define the empirical quantile threshold. When the set below is nonempty:

```math
\tau_{\mathrm{MC}}
=
\min
\left\{
s\in\{S_1,\ldots,S_n\}
:
\frac1n
\sum_{i=1}^{n}
\mathbf 1
\{S_i\leq s\}
\geq
q\left(1+\frac1n\right)
\right\}.
```

Equivalently:

```math
\tau_{\mathrm{MC}}
=
\min
\left\{
s\in\{S_1,\ldots,S_n\}
:
\frac{
\left|
\{i:S_i\leq s\}
\right|
}{n}
\geq
q\left(1+\frac1n\right)
\right\}.
```

If the set is empty, the conservative convention is `\tau_{\mathrm{MC}}=+\infty`, in which case the event `S_{n+1}\leq\tau_{\mathrm{MC}}` holds trivially.

For the nontrivial finite-threshold case, this definition implies:

```math
\left|
\{i:S_i\leq\tau_{\mathrm{MC}}\}
\right|
\geq
\left\lceil(n+1)q\right\rceil.
```

If:

```math
S_{n+1}>\tau_{\mathrm{MC}},
```

then at least:

```math
\left\lceil(n+1)q\right\rceil
```

calibration scores are strictly smaller than `S_{n+1}`, so:

```math
I_{n+1}=1.
```

Thus:

```math
\mathbb P
\left(
S_{n+1}\leq\tau_{\mathrm{MC}}
\right)
\geq
1-
\mathbb P
\left(
I_{n+1}=1
\right)
\geq
q.
```

This is the formal finite-sample rank argument for the randomized score.

#### Proposition A.1 - Future Dependency

**Paper statement.** With:

```math
q
=
1-\alpha+2\beta,
```

the randomized Monte Carlo score satisfies:

```math
\mathbb P
\left[
\widehat S_{\mathrm{RS},N_{\mathrm{MC}}}
(X_{n+1},Y_{n+1})
\leq
\tau_{\mathrm{MC}}
\right]
\geq
1-\alpha+2\beta.
```

The probability is over the data and the randomized score-generation procedure, including finite Monte Carlo smoothing samples under the conditions above.

This proposition is a future dependency for Corollary 2. In this update it is recorded only as a randomized-score conformal coverage fact, not as a completed robust coverage proof.

---

## 8. Assumptions

Each assumption must be tied to a theorem, algorithm step, experiment, or implementation requirement. Current confirmed assumptions are limited to the foundational layers above.

### Vanilla CP Validity Ledger

Vanilla split conformal validity currently relies on:

* calibration/test exchangeability of the true-label scores;
* the same score construction for calibration and future test examples;
* the predictive model being fixed before calibration in split conformal prediction;
* if the score contains auxiliary randomness, the randomization mechanism being applied symmetrically and independently as required.

It does not require:

```math
\widehat\pi_y(x)
=
\mathbb P(Y=y\mid X=x).
```

It does not imply:

* exact conditional coverage;
* small prediction sets;
* model correctness;
* robustness to shifted or adversarially modified test inputs.

### Data Assumptions

* Clean split conformal validity uses exchangeability of calibration and test true-label scores.
* i.i.d. data are sufficient for exchangeability, but exchangeability is the deeper requirement.
* Training, calibration, and test roles must be separated for the simple split conformal proof.
* The classification setup uses realized inputs `x_i\in\mathbb R^p` and realized labels `y_i\in[K]`.

### Probability and Calibration Assumptions

* The empirical quantile correction accounts for a pooled rank space of size `n+1`.
* Calibration scores must be true-label scores, because the coverage event is a true-label score threshold event.
* Marginal coverage is the target in Section 2.1; pointwise conditional coverage is not automatically guaranteed.
* Randomized scores remain valid only when the randomized scoring procedure is applied symmetrically and independently as required.

### Robustness and Geometry Assumptions

* Adversarial perturbation is represented as `\Delta` with an `L_2` constraint.
* `\Delta` is not assumed Gaussian, mean-zero, or random.
* Gaussian smoothing uses `Z\sim\mathcal N(0,\sigma^2 I_p)`.
* The randomized smoothing relation gives an `\epsilon/\sigma` population-level score inflation bound.
* Full robust coverage transfer is deferred to RSCP/RSCP+ sections.

### Randomization and Computation Assumptions

* The base score is bounded in `[0,1]` for Hoeffding-style concentration.
* Monte Carlo smoothing samples are independent of the data and of each other under Lemma A.1.
* `S_{\mathrm{RS}}` is a population expectation; `\widehat S_{\mathrm{RS},N_{\mathrm{MC}}}` is a finite random estimator.
* Theorem 1-level confidence/union-bound bookkeeping remains deferred.

### Model and Training Assumptions

* The classifier output is a vector `\widehat\pi(x)\in[0,1]^K`.
* Conformal validity does not require correct classifier specification.
* Model/score quality affects set efficiency, not the source of marginal validity.
* RCT assumptions are not yet read.

### Evaluation Assumptions

* Separation of validity, robustness, prediction-set efficiency, and computational efficiency.
* Experiments must be checked later against the exact theorem object and implemented score.
* Static classification, norm-bounded perturbations, and Gaussian smoothing geometry are not automatically transferable to dynamic-system forecasting.

---

## 9. Experimental Evidence

Not yet read.

| Evidence Slot | What to Extract Later | Current State |
| --- | --- | --- |
| Datasets and tasks | Which settings are actually tested | TBD |
| Baselines | What comparisons establish improved efficiency | TBD |
| Validity evidence | Whether empirical coverage matches the stated guarantee | TBD |
| Robustness evidence | Whether perturbation robustness is stress-tested | TBD |
| Prediction-set efficiency | Whether sets become smaller or more informative | TBD |
| Computational efficiency | Runtime, smoothing samples, and training overhead | TBD |
| Ablations | Which component explains the reported behavior | TBD |

Do not add CIFAR, ImageNet, numeric results, or ablation conclusions until the experiments are read.

---

## 10. Limitations

This section is not a final critique. It records limitation categories that already follow from the foundational setup, plus categories that must be tested after later sections.

### Currently Supported Boundaries

* Section 2.1 split conformal validity is marginal, not pointwise conditional validity.
* Clean conformal validity relies on exchangeability of true-label scores.
* Test-time adversarial perturbation can break the clean calibration/test score symmetry even if independence from calibration data survives.
* Randomized smoothing population guarantees are not the same as finite Monte Carlo implementation guarantees.
* Naive pointwise Monte Carlo confidence control over every calibration score can become severely conservative.
* Prediction-set validity and prediction-set usefulness are separate.

### Marginal-Not-Conditional Limitation

Global marginal coverage can conceal poor reliability in specific regions. The hard-region example in Section 2.1 shows that a high distribution-weighted average can coexist with weak subgroup coverage.

### Efficiency Limitation

Validity can remain correct while prediction sets become uninformative. The extreme set `C(x)=[K]` has coverage one but almost no decision value. A weak model or poor score geometry tends to expose itself through larger calibrated thresholds and larger sets.

### Exchangeability Limitation

Ordinary split conformal coverage does not automatically survive distribution shift or adversarially modified test inputs. The clean proof requires calibration and test true-label scores to have the needed exchangeable rank structure.

### Limitation Categories for Later Reading

* assumption-derived limitations;
* proof conservativeness;
* prediction-set inflation;
* population-to-implementation mismatch;
* Monte Carlo cost;
* score-geometry sensitivity;
* training surrogate gap;
* empirical coverage versus certified robustness;
* transfer limits outside static classification.

No experiment-derived limitation claim is made yet.

---

## 11. Research-Level Critique

Not yet completed as a final critique.

Future critique must distinguish:

* what the paper proves;
* what the experiments support;
* what remains an implementation choice;
* what remains a modeling assumption;
* what is useful but conservative;
* what is efficient in prediction-set size versus efficient in computation;
* what can transfer to dynamic-system reliability and what cannot.

This section should not be filled with generic criticism. Each critique must point to a verified assumption, proof step, estimator gap, empirical anomaly, or deployment mismatch.

### Current Research-Level Tension

**Research interpretation.** The foundational tension is already visible:

```text
conformal validity
comes from rank/exchangeability

robustness
requires transferring membership under adversarial input changes

implementation validity
requires proving statements about the finite randomized score

prediction-set efficiency
depends on score geometry and threshold inflation
```

The research lesson is not simply "use randomized smoothing." The stronger lesson is that every guarantee must name the random object it certifies.

---

## 12. Connection to My Active Project

Active long-term direction:

```text
Real-world Representation and Mechanism Learning in Dynamic Systems
```

Recent testbed:

```text
Reliable Spatiotemporal Forecasting under Dynamic Distribution Shift
```

Relevant project dimensions:

* reliable forecasting;
* uncertainty quantification;
* conformal prediction;
* temporal and spatial shift;
* missingness;
* sensor noise;
* model monitoring;
* representation drift;
* graph drift;
* reliability trigger.

### Transferable Mathematical Ideas

These are currently supported at the level of mathematical reading discipline:

* finite-sample calibration;
* concentration bounds;
* explicit randomness accounting;
* theory-implementation consistency;
* certificate construction;
* guarantee-efficiency trade-off.

### Transferable to Dynamic-System Reliability

**Project connection.** The following ideas are transferable as research principles:

* explicit randomness bookkeeping;
* finite-sample calibration;
* theory-implementation alignment;
* coverage versus usefulness;
* concentration-based implementation certificates;
* monitoring of set size / interval width.

### Potentially Transferable but Not Yet Established

**Research interpretation - not a result of this paper.**

* efficiency degradation as a reliability signal;
* finite-sample uncertainty accounting in rolling deployment;
* using prediction-set inflation as a model-health indicator.

These require new analysis before being used in time-series or spatiotemporal forecasting.

### Transferable Research Philosophy

Research interpretation for this reading, not a paper quotation:

> A formal guarantee should apply to the random object actually implemented, rather than only to an idealized population quantity.

This principle may be important for reliable forecasting systems because deployed systems use finite samples, finite computation, noisy sensors, and changing environments.

### Dynamic-System Research Order

In temporal systems, even without adversarial perturbation:

```math
(X_t,Y_t)
```

and:

```math
(X_{t+1},Y_{t+1})
```

often exhibit temporal dependence or nonstationarity. Thus ordinary exchangeable split conformal prediction may already be the first point of failure.

The research order for dynamic systems should not be:

```text
vanilla CP works
        v
directly add RSCP
```

The prior question is:

```text
what replaces exchangeability?
```

Possible future literature directions include:

* weighted conformal;
* online conformal;
* adaptive conformal;
* conformal methods for dependent time series.

These are future directions, not results of this paper.

### Non-Transferable Assumptions

The following assumptions cannot be imported directly into the dynamic-system project without new analysis:

* static image classification;
* exchangeable/i.i.d. test setting;
* norm-bounded adversarial perturbation;
* Gaussian smoothing geometry.

For time series, the first problem may arise even before adversarial robustness:

```text
temporal dependence
        v
exchangeability failure
        v
ordinary conformal rank proof may not directly apply
```

Therefore RSCP+ should not be directly transplanted to spatiotemporal forecasting as if it solved temporal dependence, generic covariate shift, concept drift, graph topology shift, missingness, or mechanism shift.

---

## 13. Transferable Intuitions

Current status: foundational intuitions supported; later method-specific intuitions still deferred.

| Candidate Intuition | Why It May Matter | Verification Needed |
| --- | --- | --- |
| Finite-sample calibration should be tied to the actual deployed random object | Prevents theory from proving a guarantee for a quantity different from the implementation | Foundation established; verify full RSCP+ proof next |
| Randomness accounting is part of reliability | Data randomness, APS randomization, Gaussian smoothing, Monte Carlo sampling, and adversarial perturbation play different roles | Foundation established |
| Target coverage is a requirement, not a model belief | Prevents confusing `1-\alpha` with classifier accuracy or per-instance confidence | Foundation established |
| Validity and intelligence are different | A full label set can be valid but useless | Foundation established |
| Certificates can trade informativeness for safety | Robust threshold inflation can enlarge sets | Verify set-size mechanism and experiments |
| Score geometry matters | True-label and wrong-label score separation affects set size under the same coverage target | Foundation established; PTT details deferred |
| Clean-to-attacked transfer is a separate proof obligation | Adversarial test scores generally do not inherit clean exchangeable rank symmetry | Foundation established; Section 2.2 begins next |
| Training surrogates are not automatically final guarantees | Differentiable objectives may optimize a proxy rather than the exact conformal set | Read RCT and final guarantee relationship |

These intuitions should guide later reading without being upgraded into final paper claims prematurely.

---

## 14. Implementation Implications

No experiment code should be derived from this note yet. The implementation implications below are audit requirements for future code or reproduction work.

Current and future implementation checks:

| Component | Implication to Verify Later | Required Check |
| --- | --- | --- |
| Data split | Calibration must be isolated from training and final testing | Audit split and exchangeability assumptions |
| Score computation | Implemented score must match the theorem's random object | Compare paper definition to code design |
| Quantile computation | Indexing and finite-sample correction must use the `n+1` rank logic | Match the empirical quantile convention |
| Monte Carlo smoothing | Number of samples and estimator randomness must be logged | Track random seeds and confidence budget |
| Robust certificate | Perturbation radius and smoothing assumptions must match implementation | Verify certificate-object correspondence |
| Prediction set | Validity and efficiency metrics must be separated | Report coverage and set size separately |
| Training surrogate | RCT objective must not be confused with final validity guarantee | Identify final calibration step |
| Experiment logging | Validity, robustness, prediction-set efficiency, and compute cost must be logged separately | Design metrics after reading |

---

## 15. Possible Research Questions

No final research questions are written at this stage.

A future research question must come from at least one verified source:

* explicit assumption;
* theoretical limitation;
* proof conservativeness;
* implementation mismatch;
* empirical anomaly;
* application mismatch.

Before recording a question as research-worthy, check:

1. Has it already been solved by existing work?
2. Is it falsifiable?
3. What is the mathematical object?
4. What is the experimental object?
5. Would a negative result still teach something?

| Candidate Question | Source Requirement | Status |
| --- | --- | --- |
| TBD | Must pass the quality-control gate above | Not yet formulated |

---

## 16. What I Should Be Able to Explain After Reading

This paper should help build advisor-level reading discipline for trustworthy ML and Tsui-Wei Weng / Lily Weng related research style, but only through technical reconstruction rather than superficial evaluation.

### After This Foundations Update

I should now be able to explain:

1. why adversarial `\Delta` and Gaussian smoothing `Z` are different mathematical objects;
2. what adversarial perturbation primarily breaks in the conformal rank argument;
3. why independence from calibration data may survive a deterministic test-input transformation;
4. why mean shift is not the definition of distribution shift or i.i.d. failure;
5. how Gaussian averaging and inverse Gaussian CDF transformation create a perturbation-controlled score;
6. why the one-dimensional toy example gives `\Phi(x/\sigma)` and then `x/\sigma`;
7. what a Monte Carlo estimator is;
8. why a population expectation and finite estimator are not the same object;
9. what LLN, CLT, and Hoeffding each answer;
10. why original RSCP has an implementation-theory gap;
11. why naive per-calibration-point Monte Carlo confidence control is difficult;
12. what training, calibration, and test splits do in split conformal prediction;
13. why `1-\alpha` is a target marginal coverage requirement, not model accuracy or per-instance confidence;
14. how `1-\alpha` determines `k`, then `\tau`, then `C(x;\tau)`;
15. what a non-conformity score means;
16. how HPS and APS differ;
17. why HPS membership is equivalent to `\widehat\pi_y(x)\geq 1-\tau`;
18. why calibration uses true-label scores;
19. why coverage is equivalent to a true-label score threshold event;
20. how exchangeability creates rank symmetry;
21. why the finite-sample correction involves `n+1`;
22. how the empirical quantile convention matches the corrected rank;
23. what finite-sample edge case forces `\widehat\tau_\alpha=+\infty`;
24. how Appendix A.2 establishes randomized-score coverage through the indicator proof;
25. why conformal validity does not require correct classifier specification;
26. why bad model quality hurts efficiency rather than necessarily breaking marginal validity;
27. why marginal coverage differs from conditional coverage;
28. why marginal coverage differs from empirical test coverage;
29. why LLN and conformal rank coverage have different proof roles;
30. why adversarial attack breaks ordinary score-rank symmetry;
31. why RSCP needs a separate clean-to-attacked score transfer bridge;
32. why validity, robustness, prediction-set efficiency, and computational efficiency must stay separate.

### Still Deferred

I should not yet claim I can fully explain:

* Section 2.2 RSCP as presented by the paper;
* Theorem 1;
* Corollary 2;
* Empirical Bernstein;
* PTT;
* RCT;
* experiments and ablations;
* final paper-level critique.

---

## 17. Follow-Up Actions

| Action | Target File or Project Component | Status |
| --- | --- | --- |
| Completed first-pass research understanding: Part 0 whole-paper map | papers/P-ROB-001/note.md | Done |
| Completed first-pass research understanding: Part 1 vanilla split conformal prediction | papers/P-ROB-001/note.md | Done |
| Completed foundational randomness / Monte Carlo prerequisites | papers/P-ROB-001/note.md | Done |
| Current transition: Section 2.1 to Section 2.2 | papers/P-ROB-001/note.md | Ready |
| Begin next formal reading unit: Section 2.2 Eq. (6)-(11), score inflation to robust set construction to randomized smoothing certificate | papers/P-ROB-001/note.md | Next |
| Reconstruct RSCP population score, perturbation relation, and prediction set only after Section 2.2 is read | Sections 5-8 and 16-18 as needed | Planned |
| Keep Section 3, Theorem 1, Corollary 2, PTT, RCT, experiments, and critique deferred until their own scoped reading updates | This note | Planned |
| Verify primary-source metadata before adding DOI, OpenReview, arXiv, or code links | Section 1 | Planned |
| Commit and push each narrowly scoped future reading update | Git repository | Planned |

Future workflow:

```text
discussion completed
v
receive a narrowly scoped update prompt
v
update only the corresponding note sections
v
commit
v
push
```

Do not independently finish the whole paper.

---

## 18. Completion Criteria

**Current paper status:** Reading

### Current Stage Completion Record

* [x] Whole-paper framework established
* [x] Introduction mapped
* [x] Prerequisite uncertainty/randomness distinctions established
* [x] Vanilla split conformal foundations developed
* [x] Section 2.1 first-pass research understanding substantially complete
* [x] Appendix A.2 randomized-score coverage proof skeleton preserved/organized
* [x] Project-connection boundary clarified
* [x] Section 2.2 ready / beginning state established
* [ ] Section 2.2 Randomized Smoothed Conformal Prediction completed
* [ ] Section 3 / Theorem 1 completed
* [ ] Corollary 2 completed
* [ ] PTT / RCT completed
* [ ] Experiments completed

Mark this paper as `Completed` only after the following criteria are genuinely satisfied.

* [x] Vanilla split conformal coverage mechanism understood
* [x] Vanilla split conformal pipeline understood
* [x] Target coverage semantics understood
* [x] Exchangeability and finite-sample rank argument understood
* [x] Finite-sample corrected rank derived
* [x] Empirical quantile correction derived
* [x] Empirical quantile relation understood
* [x] Coverage event equivalence understood
* [x] HPS / APS role understood
* [x] Marginal vs conditional coverage understood
* [x] Classifier quality vs conformal validity understood
* [x] Score choice / efficiency distinction understood
* [ ] RSCP motivation understood
* [ ] Randomized smoothing construction understood
* [ ] Gaussian smoothing robustness relation understood
* [ ] Population score vs Monte Carlo estimator distinction understood
* [ ] Original RSCP certificate gap understood
* [ ] RSCP+ Theorem 1 independently reconstructed
* [ ] Probability space of Theorem 1 identified
* [ ] Hoeffding bound derived
* [ ] Union-bound bookkeeping understood
* [ ] Corollary 2 robust coverage logic reconstructed
* [ ] Hoeffding vs empirical Bernstein analyzed
* [ ] Robust prediction-set inflation mechanism understood
* [ ] PTT ranking transformation understood
* [ ] Rank transformation exchangeability argument understood
* [ ] Sigmoid transformation geometry understood
* [ ] PTT failure mode analyzed
* [ ] RCT differentiable conformal training understood
* [ ] Soft quantile and soft prediction set gap understood
* [ ] Experimental hypotheses and evidence analyzed
* [ ] Ablations analyzed
* [ ] Assumptions completed
* [ ] Limitations completed
* [ ] Research-level critique completed
* [ ] Dynamic-system transfer boundaries analyzed
* [ ] Research questions passed quality-control filter
* [ ] Markdown/math rendering verified
* [ ] Note remains evidence-based
* [ ] Repository boundary verified
* [ ] Git working tree clean after completion

This note remains an explicitly unfinished research container.

### Next-Stage Reading Roadmap

The next actual reading stage is:

```text
Part 2 - Section 2.2: Randomized Smoothed Conformal Prediction
```

The Part 2 reading order should be:

1. paper's formal RSCP score definition;
2. paper's notation for Gaussian smoothing and any adapted notation needed for clarity;
3. population transformed score;
4. perturbation certificate under `L_2` threat model;
5. RSCP prediction set and threshold inflation;
6. exact object used by the original RSCP theorem;
7. exact object computed by finite Monte Carlo implementation;
8. where Section 3 begins the RSCP+ motivation.

Stop before Theorem 1 until Section 2.2 and the RSCP implementation gap are genuinely understood.
