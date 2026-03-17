---
tags: #research #ai #context-manager #temporal-gradient #salience #active-context #architecture #active
created: 2026-03-17
type: research-draft
status: active
author: Justin Shank
---

**Related:** [[Projects/Research/AI Optimal Synthesis — Architecture Brief|Architecture Brief]] | [[Systems/Core Theoretical Frameworks/Temporal Gradient/Temporal Gradient Hub|Temporal Gradient]] | [[00_PROJECTS_INDEX|Projects Index]]

---

# Minimal Context Manager v0

*Captured 2026-03-17, ~1:30 AM*
*This is the bridge between the Temporal Gradient as theory and as architecture.*

---

## The Core Shift

Current retrieval systems rank candidates by **semantic similarity**.

This system ranks candidates by **goal-conditioned salience**:

> Ψ(c | G, state) = H(c) × V(c | G)

Where:
- **H** = surprise — how much this candidate changes the current model/state (embedding distance from current context)
- **V** = value — how much this candidate helps achieve the objective (keyword overlap with goal, or LLM scoring)
- **Ψ** = salience — what actually deserves attention

**The critical distinction:** Ψ is not a module. It is the **control signal** governing the entire agent loop. The same function governs retrieval, memory writes, attention, tool triggering, and pruning.

This replaces ten incompatible local priority functions with one unified signal derived from first principles.

---

## The Algorithm

### Inputs
- **G** — Goal state
- **M** — Memory
- **R** — Retrieval candidates
- **T** — Tool outputs
- **H** — History

---

### Step 1: Candidate Generation
```
C = M ∪ R ∪ T ∪ H
```
All candidates pooled into a single set regardless of source.

---

### Step 2: Pre-Selection (Ψ₀)
```
Compute Ψ₀(c | G, state) for all c ∈ C
Select top-K under token budget
```
Every candidate scored against the current goal state. Only high-salience candidates enter context. Budget-constrained selection replaces passive injection.

---

### Step 3: Structuring
Organize selected candidates into four categories:
- **Facts** — verified, retrieved, high-confidence
- **Constraints** — boundaries on the solution space
- **Assumptions** — inferred, low-confidence, flagged
- **Open questions** — gaps that may require retrieval or tool use

Structure is not cosmetic. It determines how the model attends to the injected context.

---

### Step 4: Model Inference
Model runs on structured context only. Raw candidates never reach the model directly.

---

### Step 5: During Inference — Optional Hooks (Ψ₁, skip in v0)
```
Update Ψ₁:
- detect confusion signals
- trigger mid-inference retrieval
- trigger tool calls when Ψ gap detected
```
*Deferred to v1. V0 is pre-inference salience only.*

---

### Step 6: Post-Action Evaluation (Ψ₂)
```
Compute Ψ₂:
- which context actually mattered? (was cited, influenced output)
- which was ignored? (present but unused)
- which misled? (present, used, output was wrong)
```
This is the evaluation signal. Ψ₂ is the ground truth against which Ψ₀ is calibrated.

---

### Step 7: Update
```
- reinforce useful items (increase retrieval weight, memory priority)
- decay useless ones (decrease weight, accelerate forgetting)
- update retrieval index weights
- update memory priorities
```
The loop learns. Ψ₀ improves over time by comparing its predictions against Ψ₂ outcomes.

---

## Minimal Context Manager v0 — Implementation Spec

**Scope:** Ψ₀ only. No mid-inference hooks. No learning loop yet. Prove the ranking signal first.

### V proxy (value relative to goal)
Option A: keyword overlap between candidate and goal statement
Option B: LLM scoring — "does this help achieve [G]?" (0-1)
Start with Option A. Option B is more accurate but adds inference cost.

### H proxy (surprise / information gain)
Embedding distance from centroid of current context window.
High distance = high surprise = high H.

### Ψ₀ scoring
```python
def psi_0(candidate, goal, current_context):
    H = embedding_distance(candidate, centroid(current_context))
    V = keyword_overlap(candidate, goal)  # or llm_score(candidate, goal)
    return H * V
```

### Selection
```python
def select_context(candidates, goal, current_context, budget):
    scored = [(c, psi_0(c, goal, current_context)) for c in candidates]
    ranked = sorted(scored, key=lambda x: x[1], reverse=True)
    return fit_to_budget(ranked, budget)
```

### Baseline comparison
Run same task with:
- **Baseline:** standard cosine similarity retrieval
- **v0:** Ψ₀-ranked selection

---

## Evaluation

### Metrics
| Metric | What it measures |
|---|---|
| **Accuracy** | Task success rate vs. baseline |
| **Token efficiency** | Tokens used per successful task |
| **Context utilization rate** | % of injected context that influenced output |
| **Ψ₀ vs Ψ₂ calibration** | How well pre-selection predicts post-action usefulness |

### Evaluation tasks
- Factual QA (tests whether H × V outperforms cosine on exact-fact retrieval)
- Multi-step reasoning (tests whether goal-conditioned ranking prevents drift)
- Long-horizon tasks (tests whether token efficiency compounds over time)

---

## Why This Is Architecture-Level, Not Feature-Level

The brief describes 10 components with 10 different priority functions:
- Retrieval ranks by cosine similarity
- Memory decays by time
- Tool triggering thresholds by confidence
- Evaluation measures by accuracy

All 10 are solving the same problem — *what deserves attention right now* — with incompatible local solutions.

Ψ as unified control signal collapses this. Every component becomes a **consumer of Ψ** rather than a generator of its own relevance metric.

The components stay. The coordination mechanism unifies.

---

## Relationship to the Temporal Gradient

The Temporal Gradient describes how internal time (τ) accumulates at a rate modulated by salience: high-Ψ events consume more τ, persist longer, decay slower.

This system asks: what if the agent used the same logic to build its own context?

The Temporal Gradient describes how attention *naturally* allocates under constraint.
The Context Manager *implements* that allocation deliberately.

The equation didn't change. The placement in the system did.

---

## What to Build Next (v1)

1. Add Ψ₁ — mid-inference hooks for confusion detection and triggered retrieval
2. Add learning loop — compare Ψ₀ predictions against Ψ₂ outcomes, update weights
3. Replace V proxy with fine-tuned scoring model
4. Apply to memory writes — Ψ-gated memory reinforcement and decay
5. Apply to tool triggering — Ψ threshold replaces confidence threshold

---

## Open Questions

- What is the right V proxy for domains where "goal" is underspecified?
- How does Ψ interact with contradiction — should two conflicting high-Ψ candidates both enter context, or does contradiction resolution happen at the structuring step?
- Is H × V the right combination, or does domain matter (creative tasks may need higher H weight; factual tasks higher V)?
- At what Ψ threshold does a mid-inference hook trigger retrieval without interrupting reasoning?

---

*Draft. Not polished. Captured while the framing was live.*
*Next step: implement v0 prototype, run baseline comparison.*
