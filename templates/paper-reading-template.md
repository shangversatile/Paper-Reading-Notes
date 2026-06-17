# Paper Reading Note Template

> This template is for research-level reading notes.
> The goal is not to summarize a paper mechanically, but to extract understanding,
> assumptions, limitations, transferable ideas, and research questions.

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
