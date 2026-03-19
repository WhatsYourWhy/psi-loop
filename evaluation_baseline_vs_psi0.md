# Baseline vs Psi0 Evaluation

**Current mainline (see [evaluation_run_log.md](evaluation_run_log.md) for experiment history):** Psi0 uses linear `V × H` with near-tie value-priority ranking. Bow and Dense both show no baseline wins; dense roadmap task is now a tie (Psi0 selects gold). Artifacts from the current system: `evaluation_results_baseline_vs_psi0_bow_near_tie_v_priority.json`, `evaluation_results_baseline_vs_psi0_dense_all-MiniLM-L6-v2_near_tie_v_priority.json`.

## Question

Does `Psi0` select better context than the similarity-only baseline under token constraints, and does a stronger representation backend materially improve that result?

## Setup

- Selector under test: `Psi0`
- Baseline: similarity-only ranking
- Benchmark fixture: `tests/fixtures/benchmark_tasks.json`
- Bow runner: `python scripts/run_baseline_vs_psi0.py --backend bow`
- Dense runner: `python scripts/run_baseline_vs_psi0.py --backend dense`
- Bow result artifact: `evaluation_results_baseline_vs_psi0_bow.json` (or `*_near_tie_v_priority.json` for current mainline)
- Dense result artifact: `evaluation_results_baseline_vs_psi0_dense_all-MiniLM-L6-v2.json` (or `*_near_tie_v_priority.json`)
- To avoid overwriting reference artifacts when replaying, use `--json-out <path>` (see [evaluation_run_log.md](evaluation_run_log.md)).
- Dense model: `all-MiniLM-L6-v2`
- Result metadata: each JSON artifact records `embedder_metadata.backend`, `embedder_metadata.class_name`, and `embedder_metadata.model_name`

## Benchmark Design

The benchmark contains 14 hand-authored tasks:

- 10 `synthetic_redundancy` tasks designed to isolate the redundancy-penalty hypothesis
- 4 `realistic_knowledge_work` tasks designed to approximate practical note-selection scenarios

Each task includes:

- a goal
- current context
- a token budget
- a candidate set
- gold useful candidates
- gold redundant candidates
- an expected winner for sanity checking

## Metrics

Task-level winner was determined by:

1. higher useful-hit count
2. if both sides found useful context, lower redundant-hit count
3. then useful precision
4. then lower selected token total

If both selectors found zero useful candidates, the task was scored as a tie.

Aggregate metrics:

- task wins / losses / ties
- useful-hit totals
- redundant-hit totals
- synthetic vs realistic split

## Results

### Comparison matrix (current mainline: near-tie value-priority)

| Backend | Psi0 wins | Baseline wins | Ties | Psi0 useful hits | Baseline useful hits | Psi0 redundant hits | Baseline redundant hits | Expected matches | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `BowEmbedder` | 4 | 0 | 10 | 5 | 1 | 7 | 12 | 4 / 14 | `refine_v` |
| `STEmbedder` | 2 | 0 | 12 | 3 | 1 | 6 | 13 | 2 / 14 | `refine_v` |

### By category

#### `BowEmbedder`

- `synthetic_redundancy`: `Psi0` 3, baseline 0, tie 7
- `realistic_knowledge_work`: `Psi0` 1, baseline 0, tie 3

#### `STEmbedder`

- `synthetic_redundancy`: `Psi0` 2, baseline 0, tie 8
- `realistic_knowledge_work`: `Psi0` 0, baseline 0, tie 4 (roadmap is tie; both select gold)

## Interpretation

The current mainline includes a **selection-time rule** (near-tie value-priority): when two candidates' Psi0 scores differ by less than 0.01, the one with higher value is ranked first before budget packing. That fixed the dense roadmap failure: the gold candidate had higher V but lost rank-1 by a tiny score gap; greedy budget then selected the high-surprise candidate. With near-tie V-priority, dense roadmap is now a tie and Psi0 useful hits on dense improved (2 → 3) with no baseline wins and no redundant-hit increase.

- **Bow:** unchanged; directional signal (4 wins, 0 baseline wins, useful 5 vs 1).
- **Dense:** no baseline wins; roadmap tie; +1 Psi0 useful hit; redundant hits unchanged at 6.

So the bottleneck on dense was partly **how ranking and budgeting operationalize** the score, not only the score function itself. The run log documents the experiments that led here (plan bonus, relation bonus, forensic, near-tie rule).

## Scientific Conclusion

For the current benchmark with near-tie value-priority:

- Psi0 + Bow has directional signal (4 wins, no baseline wins).
- Psi0 + dense now has no baseline wins; dense roadmap is tie; useful hits improved vs the pre-fix dense run.
- The dense roadmap failure was addressed by a policy-layer change (tie resolution under score uncertainty), not by further lexical V engineering.

## Recommended Next Steps

- Treat this as the dense-safe baseline. Further experiments are best in the **selection-policy** family (e.g. diversity-aware packing, reserve for top-V), not more cue buckets. See [evaluation_run_log.md](evaluation_run_log.md) for full history and replay commands.

## Reproduction

```powershell
python -m pip install -e .[dev]
python -m pip install -e .[dense]
pytest
python scripts/run_baseline_vs_psi0.py --backend bow
python scripts/run_baseline_vs_psi0.py --backend dense
```

To write results to a different file (e.g. when replaying without overwriting reference artifacts), use `--json-out <path>`. See [evaluation_run_log.md](evaluation_run_log.md) for replay commands.
