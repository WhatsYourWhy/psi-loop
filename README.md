# psi-loop

`psi-loop` is a small Python library for redundancy-aware context selection. Instead of ranking candidates by similarity alone, it scores them by two signals: how relevant they are to the goal, and how novel they are relative to what is already in context. The core rule is `Psi0`: prefer context that is both useful and non-redundant.

The package is still intentionally narrow, but it is no longer just a hardcoded demo. It now exposes pluggable embedders, pluggable candidate sources, a thin `PsiLoop` orchestration layer, and a zero-dependency bag-of-words fallback so the ranking thesis can be exercised locally before introducing heavier retrieval or embedding backends.

## Why It Exists

Standard retrieval tends to return whatever looks semantically similar to the goal, even when that context is repetitive or stale. `psi-loop` explores a different ranking rule:

- `V`: value relative to the goal
- `H`: surprise relative to the current context
- `Psi0 = H * V`

The goal of this repo is not to ship a full agent system yet. The goal is to prove that this ranking rule is worth keeping, and to make it easy to plug into richer retrieval systems later.

## MVP Scope

- `Psi0` only
- Pluggable embedders and candidate sources
- Zero-dependency default behavior
- Budgeted context selection
- Baseline comparison against plain goal similarity
- Test fixtures and GitHub Actions CI

Not included yet:

- `Psi1` mid-inference hooks
- Learning/calibration loop from `Psi0` to `Psi2`
- Dual memory store, orchestration layer, or HITL middleware
- External model or embedding services

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

List the bundled sample tasks:

```powershell
psi-loop --list-tasks
```

Run the bundled demo task:

```powershell
psi-loop --task retry_backoff
```

You can still point the CLI at your own fixture file:

```powershell
psi-loop --fixture tests/fixtures/sample_tasks.json --task retry_backoff
```

Run the test suite:

```powershell
pytest
```

## Public API

The current package is organized around three main extension points:

- `Embedder`: text-to-vector protocol
- `CandidateSource`: candidate retrieval protocol
- `PsiLoop`: thin orchestration shell that fetches candidates, scores them, and fits them to a budget

The default install path remains zero-dependency:

- `BowEmbedder` is the fallback embedder
- `FixtureSource` is the default demo source

## How It Works

`Psi0` combines two simple signals:

- `V`: keyword overlap between a candidate and the goal
- `H`: surprise relative to the current context

The package ranks candidates by `H * V`, then fits the result into a shared token budget. A similarity-only baseline is included for comparison so fixtures can demonstrate where goal-conditioned salience beats naive retrieval.

By default, `H` is still computed through the bundled `BowEmbedder`, which preserves the current bag-of-words behavior. The scoring functions now also accept injected embedders, which is the seam intended for future dense-vector backends.

In the bundled example, the baseline prefers a note that repeats the existing fixed-delay retry policy, while `Psi0` prefers the more novel note about exponential backoff with jitter.

## Best Way To Test It

If you want to evaluate whether the project is doing anything useful yet, use this sequence:

1. Run `psi-loop --list-tasks` to confirm the bundled demo data is available.
2. Run `psi-loop --task retry_backoff` and compare the `Psi0` selection against the baseline.
3. Open `src/psi_loop/data/sample_tasks.json` and change the goal, current context, or candidates to see how ranking changes.
4. Run `pytest` to make sure your changes did not break the existing behavior.

The fastest way to learn the system is to tweak the fixture and rerun the CLI. That gives you immediate feedback on whether the ranking rule is behaving intuitively.

If you want to test the new protocol seam rather than just the demo:

1. Inject a fake embedder in tests to force a known vector geometry.
2. Use `FixtureSource` in tests or scripts to load candidate pools without going through CLI parsing.
3. Instantiate `PsiLoop(source=..., embedder=...)` and compare its ranked output to `select_context_baseline(...)`.

## Repo Layout

- `src/psi_loop/embedders.py`: `Embedder` protocol, `BowEmbedder`, and shared vector math
- `src/psi_loop/sources.py`: `CandidateSource` protocol and `FixtureSource`
- `src/psi_loop/scoring.py`: scoring primitives for `V`, `H`, and `Psi0`
- `src/psi_loop/pipeline.py`: `PsiLoop`, ranking helpers, and budget fitting
- `src/psi_loop/baseline.py`: similarity-only comparison path
- `src/psi_loop/cli.py`: fixture-backed CLI for local experiments
- `src/psi_loop/data/sample_tasks.json`: bundled demo data for first-run testing
- `tests/`: scoring, pipeline, and fixture-based regression tests

## Current Limits

- Tokenization and stemming are intentionally simple heuristics.
- The default embedder is still bag-of-words, not a dense embedding backend.
- `FixtureSource` is a demo source, not a production retrieval layer.
- The bundled fixture proves the concept on one narrow scenario, not a benchmark suite.

## Next Steps

- Add a dense embedder implementation behind the `Embedder` protocol
- Add richer source implementations behind `CandidateSource`
- Add `Psi1` hooks for triggered retrieval during reasoning
- Add post-action usefulness tracking and `Psi0`/`Psi2` calibration
- Expand fixtures into a repeatable evaluation harness

## License

MIT. See `LICENSE`.
