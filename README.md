# psi-loop

`psi-loop` is a minimal Python prototype for goal-conditioned context selection. It implements the `Psi0` ranking idea from `Minimal Context Manager v0.md` and the MVP guidance in `AI Optimal Synthesis — Architecture Brief.md`: rank candidate context by value relative to the goal and surprise relative to the current context, then select the best items under a budget.

This first release is intentionally narrow. It ships a library, a small CLI, fixture-driven tests, and a similarity-only baseline so the ranking signal can be evaluated locally before building the larger agent runtime described in the research docs.

## MVP Scope

- `Psi0` only
- Deterministic, local scoring
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

Run the sample task:

```bash
python -m psi_loop.cli --fixture tests/fixtures/sample_tasks.json --task retry_backoff
```

Run the test suite:

```bash
pytest
```

## How It Works

`Psi0` combines two simple signals:

- `V`: keyword overlap between a candidate and the goal
- `H`: surprise, approximated as bag-of-words distance from the current context

The package ranks candidates by `H * V`, then fits the result into a shared token budget. A similarity-only baseline is included for comparison so fixtures can demonstrate where goal-conditioned salience beats naive retrieval.

## Repo Layout

- `src/psi_loop/scoring.py`: scoring primitives for `V`, `H`, and `Psi0`
- `src/psi_loop/pipeline.py`: ranking and budgeted selection
- `src/psi_loop/baseline.py`: similarity-only comparison path
- `src/psi_loop/cli.py`: fixture runner for local experiments
- `tests/`: scoring, pipeline, and fixture-based regression tests

## Next Steps

- Add `Psi1` hooks for triggered retrieval during reasoning
- Add post-action usefulness tracking and `Psi0`/`Psi2` calibration
- Swap the deterministic surprise/value proxies for richer embedding or model-backed scorers
- Expand fixtures into a repeatable evaluation harness

## License

MIT. See `LICENSE`.
