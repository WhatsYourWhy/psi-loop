# AGENTS.md

## Cursor Cloud specific instructions

**psi-loop** is a pure Python library (zero runtime dependencies) for redundancy-aware context selection using the Psi0 ranking rule. The repo uses a `src/` layout with `setuptools` and `pyproject.toml`.

### Key commands

| Action | Command |
|---|---|
| Install (dev) | `python3 -m pip install -e ".[dev]"` |
| Run tests | `pytest` (or `pytest -v`) |
| CLI demo | `psi-loop --list-tasks` / `psi-loop --task retry_backoff` |
| Benchmark (BoW) | `python3 scripts/run_baseline_vs_psi0.py --backend bow` |
| Benchmark (dense, optional) | `pip install -e ".[dense]"` then `python3 scripts/run_baseline_vs_psi0.py --backend dense` |

### Non-obvious notes

- The `python` command is not available; use `python3` instead.
- Scripts installed by pip (e.g. `psi-loop`, `pytest`) land in `~/.local/bin`. Ensure `PATH` includes `$HOME/.local/bin`.
- There is no linter configured in the repository (no ruff, flake8, mypy, or pyright config). The CI only runs `pytest`.
- The `[dense]` extra installs `sentence-transformers` which is heavy; only install if specifically needed.
- No Docker, databases, or external services are required.
