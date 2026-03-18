# Baseline vs Psi0 Evaluation

## Question

Does `Psi0` select better context than the similarity-only baseline under token constraints?

## Setup

- Selector under test: `Psi0`
- Baseline: similarity-only ranking
- Embedder: `BowEmbedder`
- Benchmark fixture: `tests/fixtures/benchmark_tasks.json`
- Runner: `python scripts/run_baseline_vs_psi0.py`
- Result artifact: `evaluation_results_baseline_vs_psi0.json`

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

### Aggregate

- Total tasks: 14
- `Psi0` wins: 3
- Baseline wins: 0
- Ties: 11
- `Psi0` win rate: 21.4%
- `Psi0` useful hits: 4
- Baseline useful hits: 1
- `Psi0` redundant hits: 9
- Baseline redundant hits: 12
- Expected-winner matches: 3 / 14
- Decision: `refine_v`

### By category

- `synthetic_redundancy`: `Psi0` 1, baseline 0, tie 9
- `realistic_knowledge_work`: `Psi0` 2, baseline 0, tie 2

## Interpretation

This is not yet the result you would want for a strong publishability claim.

The current `BowEmbedder` path shows directional promise:

- `Psi0` found more gold useful candidates than the baseline
- `Psi0` selected fewer gold redundant candidates than the baseline
- the baseline never beat `Psi0` on this benchmark

But it does not yet prove that `Psi0` materially changes outcomes:

- only 3 of 14 tasks were strict `Psi0` wins
- 11 tasks were ties
- the synthetic redundancy slice, which was supposed to isolate the main hypothesis, produced only 1 win out of 10

The most important takeaway is that the current bottleneck is not system architecture. It is the quality of the scoring signal. In particular, the current bag-of-words implementation appears too weak to consistently distinguish:

- repeated phrasing that matches the goal closely
- novel but still-goal-relevant candidates

One failure mode also appeared clearly: in `synthetic_incident_playbook`, `Psi0` preferred an unrelated candidate rather than the gold useful candidate. That suggests novelty alone can outrank relevance when the current `V` proxy is too weak.

## Scientific Conclusion

For the current `BowEmbedder` implementation:

- `Psi0` is better than baseline on some tasks
- `Psi0` is not yet strong enough to count as demonstrated proof
- the correct next step is to refine `V` and/or rerun the same benchmark with a stronger embedder backend

This benchmark does not justify expanding the broader system yet.

It does justify one more focused round of work on the control signal itself.

## Recommended Next Decision

Do not add more system features yet.

Instead choose one of these:

1. Refine `V` while keeping the benchmark fixed, then rerun the exact same evaluation.
2. Add a dense embedder behind the existing protocol seam, then rerun the exact same evaluation.

If neither change materially improves the benchmark, the current direction should be reconsidered rather than expanded.

## Reproduction

```powershell
python -m pip install -e .[dev]
python scripts/run_baseline_vs_psi0.py
pytest
```
