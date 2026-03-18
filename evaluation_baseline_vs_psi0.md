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
- `Psi0` wins: 4
- Baseline wins: 0
- Ties: 10
- `Psi0` win rate: 28.6%
- `Psi0` useful hits: 5
- Baseline useful hits: 1
- `Psi0` redundant hits: 7
- Baseline redundant hits: 12
- Expected-winner matches: 4 / 14
- Decision: `refine_v`

### By category

- `synthetic_redundancy`: `Psi0` 3, baseline 0, tie 7
- `realistic_knowledge_work`: `Psi0` 1, baseline 0, tie 3

## Interpretation

This is still not the result you would want for a strong publishability claim, but it is better than the previous flat-overlap run.

The current `BowEmbedder` path shows directional promise:

- `Psi0` found more gold useful candidates than the baseline
- `Psi0` selected fewer gold redundant candidates than the baseline
- the baseline never beat `Psi0` on this benchmark
- weighting `V` improved the benchmark over the prior run

But it does not yet prove that `Psi0` materially changes outcomes:

- only 4 of 14 tasks were strict `Psi0` wins
- 10 tasks were ties
- the synthetic redundancy slice improved, but still produced only 3 wins out of 10

Relative to the previous flat-overlap version, the weighted `V` change produced:

- `Psi0` wins: `3 -> 4`
- ties: `11 -> 10`
- useful hits: `4 -> 5`
- redundant hits: `9 -> 7`
- expected-match count: `3 -> 4`

That is a real directional gain, especially on the synthetic redundancy slice, but it is not yet decisive evidence.

The most important takeaway is still that the bottleneck is not system architecture. It is the quality of the scoring signal. The weighted `V` heuristic helped, but the current bag-of-words implementation still appears too weak to consistently distinguish:

- repeated phrasing that matches the goal closely
- novel but still-goal-relevant candidates

Two failure modes remain clear:

- in `synthetic_incident_playbook`, `Psi0` still preferred an unrelated candidate rather than the gold useful candidate
- in several synthetic tasks, both selectors still converged on the redundant candidate because lexical overlap remained too dominant

That suggests the scoring signal is improved, but still not robust enough to separate procedural usefulness from topical similarity in many cases.

## Scientific Conclusion

For the current `BowEmbedder` implementation:

- `Psi0` is better than baseline on some tasks
- weighted `V` improved results, especially on synthetic redundancy cases
- `Psi0` is still not strong enough to count as demonstrated proof
- the next credible move is either one more targeted scoring refinement or a rerun with a stronger embedder backend

This benchmark does not justify expanding the broader system yet.

It does justify one more focused round of work on the control signal itself.

## Recommended Next Decision

Do not add more system features yet.

Instead choose one of these:

1. If you want one more lexical pass, keep it tightly scoped and benchmark-driven.
2. Otherwise, the more credible next move is to add a dense embedder behind the existing protocol seam and rerun the exact same evaluation.

If neither change materially improves the benchmark, the current direction should be reconsidered rather than expanded.

## Reproduction

```powershell
python -m pip install -e .[dev]
python scripts/run_baseline_vs_psi0.py
pytest
```
