# Baseline vs Psi0 Evaluation

## Question

Does `Psi0` select better context than the similarity-only baseline under token constraints, and does a stronger representation backend materially improve that result?

## Setup

- Selector under test: `Psi0`
- Baseline: similarity-only ranking
- Benchmark fixture: `tests/fixtures/benchmark_tasks.json`
- Bow runner: `python scripts/run_baseline_vs_psi0.py --backend bow`
- Dense runner: `python scripts/run_baseline_vs_psi0.py --backend dense`
- Bow result artifact: `evaluation_results_baseline_vs_psi0_bow.json`
- Dense result artifact: `evaluation_results_baseline_vs_psi0_dense_all-MiniLM-L6-v2.json`
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

### Comparison matrix

| Backend | Psi0 wins | Baseline wins | Ties | Psi0 useful hits | Baseline useful hits | Psi0 redundant hits | Baseline redundant hits | Expected matches | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `BowEmbedder` | 4 | 0 | 10 | 5 | 1 | 7 | 12 | 4 / 14 | `refine_v` |
| `STEmbedder` | 2 | 0 | 12 | 3 | 1 | 8 | 13 | 2 / 14 | `refine_v` |

### By category

#### `BowEmbedder`

- `synthetic_redundancy`: `Psi0` 3, baseline 0, tie 7
- `realistic_knowledge_work`: `Psi0` 1, baseline 0, tie 3

#### `STEmbedder`

- `synthetic_redundancy`: `Psi0` 2, baseline 0, tie 8
- `realistic_knowledge_work`: `Psi0` 0, baseline 0, tie 4

## Interpretation

The dense backend still changed behavior, but the corrected run did not strengthen the scientific conclusion.

What changed under dense embeddings:

- the baseline selected more redundant candidates than Bow: `12 -> 13`
- `Psi0` redundant hits increased: `7 -> 8`
- `Psi0` wins dropped: `4 -> 2`
- `Psi0` useful hits dropped: `5 -> 3`
- expected-winner matches dropped: `4 -> 2`
- ties increased: `10 -> 12` (no baseline wins; dense run is all Psi0 wins or ties)

This means the current limitation is not simply that bag-of-words geometry is too weak. Better representation alone still did not unlock the existing `Psi0` logic on the frozen benchmark.

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
python scripts/run_baseline_vs_psi0.py --backend bow
python scripts/run_baseline_vs_psi0.py --backend dense
```
