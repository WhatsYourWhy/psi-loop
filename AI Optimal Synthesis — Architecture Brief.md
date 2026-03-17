---
tags: #research #ai #architecture #synthesis #decision-document #active
created: 2026-03-17
type: decision-document
status: active
synthesizer: claude-sonnet-4-6 (subagent)
source: 10-agent research swarm
---

**Related:** [[00_VAULT_INDEX|Vault Index]] | [[00_PROJECTS_INDEX|Projects Index]]

---

# Decision-Grade Architecture Brief
*Synthesized from 10-agent research swarm | March 17, 2026*
*Synthesis model: claude-sonnet-4-6*

---

## 1. Executive Thesis

Current agentic AI systems fail not from insufficient intelligence but from absent architecture: no real memory, no reliable retrieval, no safe execution primitives, no honest uncertainty. The convergent finding across all 10 domains is that passive, context-window-centric designs produce compounding unreliability at every layer. The path forward is a **structured, event-driven, stateful agent runtime** with explicit memory CRUD, hybrid retrieval, decoupled execution sandboxes, HITL gates on write operations, and calibrated epistemic outputs. Build the plumbing first — the intelligence is already there; the scaffolding is not.

---

## 2. Capability Stack Ranked by Leverage

### Tier 1 — Foundational (nothing works without these)

| Capability | Why It's Load-Bearing |
|---|---|
| Task State Machine (DAG + serialized save state) | All other capabilities run on top of it |
| Dual Memory Store (vector DB episodic + KG semantic) | Substrate everything reads/writes |
| Hybrid Retrieval (BM25 + dense vector + intent router) | Primary grounding mechanism |
| Decoupled Execution Sandbox | LLM emits RPCs; deterministic external process executes |
| HITL Gate Layer | Blocks cascading failures before they start |

### Tier 2 — High Leverage (multiplies Tier 1 returns)

- Immutable Goal Anchor + Loop Detector
- Provenance + Lineage Tagging (source, timestamp, confidence, TTL on every node)
- Dynamic Model Routing (classifier → small or frontier model cascade; 60-80% cost reduction on routine steps)
- Epistemic Calibration Layer (isolates retrieved facts from synthesized conclusions, exposes logprobs)
- Dry-Run / Rollback Primitives

### Tier 3 — Secondary (high value, lower urgency)

Tiered memory compression, temporal decay on preferences, async event-driven orchestration, procedural benchmark generator, spatial UI / execution tree visualization, externalized KV-cache, domain-specific review surfaces.

---

## 3. What to Build First

**#1 — Task State JSON Schema + Serialization**
Canonical task state object: goal anchor (immutable), DAG of subtasks, current node, memory partition IDs, checkpoint timestamps, stagnation counter, last tool call + result. Serializes to disk at every state transition. This is the skeleton.

**#2 — HITL Gate Middleware**
Classify all tool calls: read-only vs. state-changing. State-changing requires explicit approval before execution. Dry-run mode returns diff/preview. Rollback hook required on every write. Non-negotiable — ship before any tool integration goes to production.

**#3 — Hybrid Retrieval Pipeline**
BM25 + dense vector in parallel. Intent router: exact-fact → BM25-weighted, conceptual → dense, temporal → recency-sorted, code → AST-aware. Contradiction surfacing (flag, don't merge). Abstain/null return when no confident match.

**#4 — Memory CRUD Tool Suite**
Explicit tools: `memory.read()`, `memory.write(content, source, ttl)`, `memory.delete(id)`, `memory.search(filter)`. Dual-store. Every node gets provenance metadata. User-facing memory interface (read + delete). Replaces passive RAG.

**#5 — Deterministic Loop Breaker + Halt Protocol**
Heuristic monitor (not LLM-based). Track: same tool called 3x with similar params, N steps with no state change. Trigger: structured "Halt and Catch Fire" report, suspend task, surface to user. Hard-coded. Not model-dependent.

---

## 4. What Not to Build Yet

| Item | Honest Reason |
|---|---|
| Async event-driven orchestration (Kafka/Redis) | Need working synchronous orchestration first. Infrastructure theater without task state machine. |
| Inference-time scaling / multi-path candidate generation | Multiplies cost immediately; requires evaluation harness you don't have; revisit at 90 days. |
| Complex user utility matrices / preference sliders | No longitudinal data to calibrate. Sycophancy is a training problem, not a UI problem. |
| Externalized KV-cache | Deep infrastructure work; dual memory store solves this at application layer with less complexity. |
| Spatial / execution tree UI | Build the data structure first. Structured text output is sufficient until Tier 1 is stable. |

---

## 5. Dependencies

```
Task State Machine
    ├── unlocks → Long-horizon execution, resumability, loop detection
    └── unlocks → Compartmentalized memory (task vs. user routing)

HITL Gate Layer
    ├── unlocks → Safe tool integration
    └── unlocks → Cascading failure prevention

Hybrid Retrieval
    ├── unlocks → Epistemic calibration (retrieved vs. synthesized separation)
    └── unlocks → Contradiction surfacing

Memory CRUD Suite
    ├── unlocks → Dual-store architecture, active forgetting, adversarial defense
    └── requires → Task State Machine (needs partition routing)

Epistemic Calibration Layer
    ├── requires → Hybrid retrieval + provenance tagging
    └── unlocks → Reliable user-facing confidence output

Dynamic Model Routing
    ├── requires → Task State Machine (routes by task type/complexity)
    └── unlocks → Cost efficiency at scale
```

---

## 6. Evaluation Plan

| Metric | How |
|---|---|
| Time-to-Success (TTS) | Instrument task start → completion. Primary utility metric. |
| Prompt-iteration count | Log re-prompts before task succeeds |
| Catastrophic commission rate | Failure taxonomy classifier: commission (wrong action) vs. omission (no action) |
| Multi-step degradation curve | Inject N-step tasks, measure accuracy at each depth |
| Retrieval precision @ intent | A/B: uniform vs. intent-routed retrieval, F1 on grounded answers |
| Memory poisoning resistance | Red-team injection harness |
| Cross-session coherence | Seed facts in session 1, probe in session 5 |
| Calibration (ECE) | Expected Calibration Error on held-out factual queries |

**How to run it:** Procedural test environment generator (no static benchmarks). Error injection harness. Failure taxonomy classifier on every failed task. Instrument telemetry before optimizing.

---

## 7. Top Failure Modes

| Rank | Failure Mode | Severity |
|---|---|---|
| 1 | Cascading write failures without HITL gate | Critical / Unrecoverable |
| 2 | Memory poisoning via adversarial injection | Critical / Persistent |
| 3 | Goal drift in long-horizon tasks | High / Compounding |
| 4 | Sycophantic self-correction flipping correct answers | High / Systematic |
| 5 | Indirect prompt injection via RAG | High / Covert |
| 6 | Hallucinated tool params + blind trust in HTTP 200 | Medium-High |
| 7 | Retrieval silence (confident wrong answer) | Medium |
| 8 | Stagnation loops consuming resources | Medium |

---

## 8. Final Recommended Architecture

```
┌─────────────────────────────────────────────────┐
│              USER / OPERATOR LAYER               │
│  Memory Inspector | Task Monitor | HITL Gates    │
├─────────────────────────────────────────────────┤
│              ORCHESTRATION LAYER                 │
│  Task State Machine (DAG) | Goal Anchor          │
│  Loop Detector | Halt Protocol | Context Router  │
├──────────────────────┬──────────────────────────┤
│    REASONING LAYER   │    EXECUTION LAYER        │
│  Dynamic Model Route │  Decoupled Sandbox        │
│  Epistemic Calibrat. │  Dry-Run Middleware       │
│  Falsifier Node      │  Semantic Output Valid.   │
├──────────────────────┴──────────────────────────┤
│              MEMORY LAYER                        │
│  Episodic Vector DB  |  Semantic Knowledge Graph │
│  Provenance Tagging  |  TTL + Decay Engine       │
│  CRUD Tool Suite     |  Partitioned Segments     │
├─────────────────────────────────────────────────┤
│              RETRIEVAL LAYER                     │
│  Intent Classifier → BM25 / Dense / SQL Router  │
│  Contradiction Detector | Abstain Protocol       │
│  Authority + Freshness Scoring                   │
└─────────────────────────────────────────────────┘
```

---

## 9. Roadmap

### 30 Days
- Task State JSON schema defined and serializing
- HITL gate middleware live — all write operations gated
- Hybrid retrieval baseline (BM25 + dense, intent router v1)
- Memory CRUD tool suite replacing passive RAG
- Loop breaker + Halt and Catch Fire protocol live
- Failure taxonomy defined, telemetry instrumented

### 90 Days
- Dual memory store fully operational (vector DB + knowledge graph)
- Provenance tagging on all memory nodes and retrieval results
- Epistemic calibration layer live (lineage separation, logprob exposure)
- Dry-run / rollback primitives on all write tools
- Falsifier node in evaluation pipeline
- Dynamic model routing v1 (cost optimization)
- First procedural evaluation suite running

### 1 Year
- Event-driven async orchestration (Kafka/Redis) replacing synchronous loops
- Inference-time scaling (multi-path candidate generation + scoring)
- Externalized KV-cache for long-session efficiency
- Longitudinal evaluation pipeline (cross-session coherence)
- Spatial UI / execution tree visualization
- Full adversarial red-team suite running continuously
- Calibrated uncertainty exposed at API level

---

## Source Agents

| Agent | Domain |
|---|---|
| 1 | Memory Systems |
| 2 | Retrieval and Grounding |
| 3 | Tool-Use and Execution |
| 4 | Long-Horizon Task Execution |
| 5 | User Utility Modeling |
| 6 | Epistemic Reliability and Calibration |
| 7 | Interface and UX |
| 8 | Systems Architecture |
| 9 | Evaluation and Benchmarking |
| 10 | Risk and Failure Modes (Adversarial/Safety) |

*Source file: `AI Optimal Synthesis.md` (97KB, 10 full agent memos)*
