# Baseline vs Psi0 Evaluation

## Question

Does `Psi0` select better context than the similarity-only baseline under token constraints, and does a stronger representation backend materially improve that result?

## Setup

- Selector under test: `Psi0`
- Baseline: similarity-only ranking
- Benchmark fixture: `tests/fixtures/benchmark_tasks.json`
- Bow runner: `python scripts/run_baseline_vs_psi0.py --backend bow --json-out evaluation_results_baseline_vs_psi0.json`
- Dense runner: `python scripts/run_baseline_vs_psi0.py --backend dense --json-out evaluation_results_baseline_vs_psi0_dense.json`
- Bow result artifact: `evaluation_results_baseline_vs_psi0.json`
- Dense result artifact: `evaluation_results_baseline_vs_psi0_dense.json`
- Dense model: `sentence-transformers/all-MiniLM-L6-v2`

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

### Comparison matrix

| Backend | Psi0 wins | Baseline wins | Ties | Psi0 useful hits | Baseline useful hits | Psi0 redundant hits | Baseline redundant hits | Expected matches | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `BowEmbedder` | 3 | 0 | 11 | 4 | 1 | 9 | 12 | 3 / 14 | `refine_v` |
| `STEmbedder` | 2 | 0 | 12 | 3 | 1 | 6 | 13 | 2 / 14 | `refine_v` |

### By category

#### `BowEmbedder`

- `synthetic_redundancy`: `Psi0` 1, baseline 0, tie 9
- `realistic_knowledge_work`: `Psi0` 2, baseline 0, tie 2

#### `STEmbedder`

- `synthetic_redundancy`: `Psi0` 1, baseline 0, tie 9
- `realistic_knowledge_work`: `Psi0` 1, baseline 0, tie 3

## Interpretation

The dense backend did change behavior, but it did not strengthen the scientific conclusion.

What improved under dense embeddings:

- `Psi0` selected fewer gold redundant candidates: `9 -> 6`
- the baseline selected even more redundant candidates than before: `12 -> 13`

What got worse:

- `Psi0` wins dropped: `3 -> 2`
- useful hits dropped: `4 -> 3`
- expected-winner matches dropped: `3 -> 2`
- ties increased: `11 -> 12`

This means the current limitation is not simply that bag-of-words geometry is too weak. Better representation alone did not unlock the existing `Psi0` logic on the frozen benchmark.

The dominant failure pattern under dense embeddings was not “baseline beats `Psi0`.” The dominant pattern was:

- `Psi0` suppressed redundancy somewhat better
- but still often failed to land on the gold useful candidate
- and in several cases drifted to unrelated candidates instead

That is a sign that the current control signal is still under-specified. The geometry improved, but the selection logic still did not consistently turn that geometry into useful wins.

## Scientific Conclusion

For the current benchmark:

- `Psi0 + Bow` has directional signal but not proof
- `Psi0 + dense` did not materially outperform `Psi0 + Bow`
- dense embeddings alone do not unlock the architecture yet

The result is scientifically useful because it narrows the hypothesis:

- the ranking principle still appears to have some signal
- representation quality alone is not enough
- the remaining bottleneck is likely still in the scoring signal or its balance, not just the embedder backend

## Recommended Next Decision

Do not expand the system yet.

The dense experiment answered the question it was supposed to answer: swapping in a stronger embedder did not convert the benchmark into proof.

That means the next credible move is not “more architecture.” It is one of:

1. revise the scoring signal again in a tightly scoped way and rerun the frozen benchmark
2. tighten the benchmark or gold-label definitions if they are not exposing the distinction cleanly enough
3. stop if the control logic cannot be made to produce materially stronger wins under the fixed task set

Right now, the evidence does **not** justify claiming that better semantic geometry alone unlocks `Psi0`.

## Reproduction

```powershell
python -m pip install -e .[dev]
python -m pip install -e .[dense]
pytest
python scripts/run_baseline_vs_psi0.py --backend bow --json-out evaluation_results_baseline_vs_psi0.json
python scripts/run_baseline_vs_psi0.py --backend dense --json-out evaluation_results_baseline_vs_psi0_dense.json
```
