# Paper Reading Note Template

> This template is for research-level reading notes.
> The goal is not to summarize a paper mechanically, but to extract understanding,
> assumptions, limitations, transferable ideas, and research questions.

## Research Notebook Principle

这个模板不是为了写浅层 paper summary。

一篇 research-level paper note 不应该只是回答：

"What did the authors do?"

它应该回答：

1. 这篇 paper 如何 formulate the problem？
2. 为什么这种 formulation 在数学上或科学上是 natural 的？
3. 哪些 assumptions 使这个 formulation valid？
4. 引入了哪些 mathematical objects？
5. 关键 equations 在 mechanism 上意味着什么？
6. 这些 equations 如何连接到 implementable computation？
7. 这个 model 编码了哪些 inductive biases？
8. 哪些 failure modes 会从这些 assumptions 中产生？
9. 在 noise、missingness、distribution shift、graph shift 或 deployment constraints 下，method 会如何表现？
10. 哪些 ideas 可以 transfer 到我自己的 research projects？
11. 这篇 paper 提出了哪些新的 research questions 或 experiments？

对于每个 central model component 或 equation，在适用时使用以下 reasoning chain：

Problem
→ Motivation
→ Mathematical object
→ Derivation
→ Mechanistic intuition
→ Implementation implication
→ Assumption
→ Failure mode
→ Project transfer
→ Research question

一篇 paper note 只有在能帮助未来的我重新 reconstruct paper 的 core idea、解释它的 mathematical logic、识别 hidden assumptions、critique 它的 limitations，并把它转化为 research action 时才有用。

## Equation Reading Standard

对于每个 structurally important equation，不要只复制 formula。

需要解释：

1. 为什么这个 equation 会出现。
2. 它解决了什么 problem。
3. 每个 symbol 的 formal meaning 是什么。
4. 每个 symbol 的 intuitive meaning 是什么。
5. 这个 equation 如何从前面的 definitions 推导出来。
6. 这个 equation 暗含了什么 computation。
7. 哪个 assumption 会破坏这个 equation 的 usefulness。
8. 这个 equation 如何连接到 active research project。

每个 important equation 都应该按照以下链条阅读：
derivation → intuition → implementation → critique → transfer。

## Model Reading Standard

对于每个 important model component，需要解释：

1. 它解决了之前的哪个 bottleneck。
2. 为什么 simpler baseline 不够。
3. 它引入了什么 inductive bias。
4. 这个 component 计算了什么。
5. 什么 information flows through it。
6. 它对 data、structure、time、causality 或 noise 做了什么 assumptions。
7. 如果这些 assumptions fail，会发生什么。
8. 这个 component 如何被 tested、ablated 或 stress-tested。
9. 它如何在我自己的 research 中被 reused 或 modified。

## Research Critique Standard

critique section 不能只是 generic limitations list。

它应该讨论：

1. paper 中最强的 idea。
2. 使 method work 的 hidden assumption。
3. 最可能出现的 failure mode。
4. predictive performance 与 reliability 之间的 gap。
5. method 提供的是 understanding、control、calibration，还是 only accuracy。
6. 这篇 paper 让什么问题更容易研究。
7. 这篇 paper 留下了什么 unresolved。
8. 如果我要 extend 这篇 paper，下一步应该运行什么 experiment。

## Project Transfer Standard

当把 paper 连接到我的 active project 时，避免 forced connections。

对于 relevant papers，需要明确判断这篇 paper 是否贡献了：

- forecasting backbone；
- graph construction idea；
- uncertainty estimation method；
- calibration method；
- distribution-shift evaluation protocol；
- robustness stress test；
- decision-reliability metric；
- representation diagnostic；
- mechanism-discovery idea；
- implementation pattern；
- 或 failure mode。

对于当前 flagship project：

Reliable Spatiotemporal Forecasting under Dynamic Distribution Shift:
Calibration, Uncertainty Quantification, and Risk-Aware Decision-Making

note 应该说明这篇 paper 是否帮助：

- reliable forecasting backbone；
- graph construction validation；
- uncertainty quantification；
- conformal calibration；
- missingness/noise/shift robustness；
- Top-K high-risk decision evaluation；
- representation stability；
- model monitoring；
- 或 risk-aware decision-making。

## Completion Discipline

不要仅仅因为 note 填满了全部 18 个 sections 就把 paper 标记为 Completed。

只有在满足以下条件时，paper 才能被标记为 Completed：

1. 我可以不看材料解释 core problem。
2. 我可以在 high level 上 reconstruct key mathematical logic。
3. 我可以 mechanistically 解释 main model components。
4. 我理解 assumptions 和 failure modes。
5. 我可以区分 paper 证明了什么、experiments 支持了什么、以及什么仍然 untested。
6. 我可以在不 forced connection 的情况下把 paper 连接到我的 project。
7. 我已经提取了至少一个有用的 research question、experiment 或 implementation implication。
8. note 已经通过 rendering 和 repository checks。

在此之前，将 status 保持为：
Reading
或
Revisit Needed

具体取决于 understanding 的深度。

---

# <Paper ID>: <Paper Title>

## 1. Citation

**Paper:**
**Authors:**
**Year:**
**Venue:**
**Primary Source:**
**Verified On:**
**Reading Status:** Queued / Reading / Completed / Revisit Needed

---

## 2. Reading Tier and Track

**Reading Tier:** Tier 0 / Tier 1 / Tier 2 / Tier 3
**Track:**
**Related Project:**
**Optional Research Context:**

### Why This Paper Is in the Curriculum

Explain why this paper is worth reading.

Do not write generic statements such as “this paper is important.”
Specify whether the paper contributes:

* a foundational concept,
* a canonical model,
* a benchmark or evaluation paradigm,
* a mathematical tool,
* an implementation pattern,
* a failure mode,
* a frontier research direction,
* or a project-specific missing link.

---

## 3. Core Problem

What problem is the paper trying to solve?

A good answer should identify:

* the previous limitation or bottleneck,
* why the problem matters,
* what would remain impossible or inefficient without this paper,
* and how the paper reframes the problem.

Avoid simply copying the abstract.

---

## 4. Intuition Before the Math

Explain the main idea in plain language before writing equations.

Use this section to answer:

* What is the conceptual obstacle?
* What is the key move made by the paper?
* What should I visualize geometrically, statistically, or computationally?
* What is the simplest mental model of the method?

This section should help future me remember the paper without rereading it.

---

## 5. Mathematical or Algorithmic Setup

List the key objects introduced by the paper.

For each object, explain both:

1. its formal definition,
2. its intuitive role.

Example format:

| Object | Formal Role         | Intuition                                              |
| ------ | ------------------- | ------------------------------------------------------ |
| `L`    | Graph Laplacian     | Measures signal variation across graph edges           |
| `K`    | Filter support size | Controls how many graph hops information can propagate |

Use equations when necessary, but every equation must be explained in words.

---

## 6. Method: Step-by-Step Logic

Break the method into a clean logical chain.

Example:

1. Start from the original problem.
2. Identify why the naive solution fails.
3. Introduce the main mathematical object.
4. Show how the method becomes efficient or expressive.
5. Explain what assumptions make the method valid.
6. Explain what the method outputs.

This section should make the method implementable in principle, but it should not contain code.

---

## 7. Key Equations and Derivations

Record only the equations that are structurally important.

For each equation, include:

* equation,
* what each symbol means,
* why the equation appears,
* what would break if this equation were misunderstood.

| Equation | Meaning | Why It Matters | Implementation or Research Implication |
| -------- | ------- | -------------- | -------------------------------------- |
| TBD      | TBD     | TBD            | TBD                                    |

Do not collect equations passively.
Only include equations that support understanding, implementation, or critique.

---

## 8. Assumptions

List the assumptions required by the method.

Separate them into:

### Data Assumptions

* TBD

### Model Assumptions

* TBD

### Optimization or Computation Assumptions

* TBD

### Evaluation Assumptions

* TBD

Then answer:

Which assumptions are realistic for my current project?
Which assumptions may fail under distribution shift, missingness, noise, or deployment constraints?

---

## 9. Experimental Evidence

Summarize what the paper actually demonstrates.

Include:

| Experiment | Dataset / Setting | What It Tests | Main Evidence | Limitation |
| ---------- | ----------------- | ------------- | ------------- | ---------- |
| TBD        | TBD               | TBD           | TBD           | TBD        |

Distinguish carefully between:

* what the paper proves,
* what the experiments support,
* what remains untested,
* and what should not be overclaimed.

---

## 10. Limitations

This section is mandatory.

List what the paper does not solve.

Possible categories:

* limited datasets,
* unrealistic assumptions,
* missing baselines,
* missing uncertainty analysis,
* missing robustness evaluation,
* no distribution-shift analysis,
* no causal interpretation,
* no decision-level evaluation,
* unclear implementation cost,
* weak connection to real deployment.

Be specific.
Do not write “future work is needed” without saying what exactly is missing.

---

## 11. Research-Level Critique

This is the most important section.

Go beyond summary and ask:

* What is the strongest idea in the paper?
* What is the hidden assumption?
* What would make the method fail?
* What does the paper make easier to study?
* What does the paper still leave conceptually unresolved?
* If I were reviewing this paper today, what would I praise?
* What would I question?
* What would I test next?

The goal is to develop research judgment, not just comprehension.

---

## 12. Connection to My Active Project

Explain how the paper connects to the current flagship project:

Reliable Spatiotemporal Forecasting under Dynamic Distribution Shift:
Calibration, Uncertainty Quantification, and Risk-Aware Decision-Making.

Address only relevant connections.

Possible dimensions:

* model architecture,
* graph construction,
* uncertainty estimation,
* calibration,
* conformal prediction,
* distribution shift,
* decision reliability,
* representation stability,
* evaluation protocol,
* implementation design,
* failure analysis.

Avoid forced connections.
If the paper is foundational but not directly project-specific, say so clearly.

---

## 13. Transferable Intuitions

What general ideas can transfer beyond this paper?

Examples:

* locality as an inductive bias,
* calibration as a system property,
* graph quality as a hidden validity condition,
* uncertainty as decision information,
* representation stability as a diagnostic tool,
* evaluation metrics as assumptions about what matters.

List transferable insights:

1. TBD
2. TBD
3. TBD

---

## 14. Implementation Implications

What should change in future code or experiment design because of this paper?

Include:

| Component          | Implication | Required Check |
| ------------------ | ----------- | -------------- |
| Data loader        | TBD         | TBD            |
| Model              | TBD         | TBD            |
| Metrics            | TBD         | TBD            |
| Stress test        | TBD         | TBD            |
| Experiment logging | TBD         | TBD            |

This section should not contain code.
It should define what implementation must respect.

---

## 15. Possible Research Questions

Convert the paper into candidate research questions.

Each question should be testable or at least sharpenable.

| Question | Why It Matters | Minimal Test or Evidence Needed | Related Project Component |
| -------- | -------------- | ------------------------------- | ------------------------- |
| TBD      | TBD            | TBD                             | TBD                       |

Do not write broad questions like “Can this be improved?”
Write questions that can eventually become experiments.

---

## 16. What I Should Be Able to Explain After Reading

List concrete understanding checks.

Example:

1. Why does the paper need this mathematical tool?
2. What problem does the main method solve?
3. What assumption makes the method valid?
4. What is the main limitation?
5. How would I explain the method to someone who knows ML but not this paper?
6. How does this paper affect my project design?

A paper is not completed until I can answer these questions without looking at the note.

---

## 17. Follow-Up Actions

List concrete next steps.

Examples:

* update terminology file,
* update literature matrix,
* update model relearning plan,
* create implementation specification,
* derive one equation by hand,
* compare with another paper,
* add a research-question card,
* design a minimal experiment,
* defer to later stage.

| Action | Target File or Project Component | Status  |
| ------ | -------------------------------- | ------- |
| TBD    | TBD                              | Planned |

---

## 18. Completion Criteria

Mark this paper as `Completed` only when:

* I understand the core problem,
* I can explain the main intuition,
* I can derive or reconstruct the key equations at a high level,
* I know the assumptions and limitations,
* I can connect the paper to my project without forcing it,
* I have extracted at least one useful research question or implementation implication,
* all relevant repository files have been updated.

Until then, keep the paper status as `Reading` or `Revisit Needed`.
