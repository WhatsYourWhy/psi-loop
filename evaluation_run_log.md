# Evaluation Run Log

## 2025-03-17

Full evaluation run: test suite, Bow benchmark, and Dense benchmark. Psi0 scoring: `score = value * surprise` (linear, no V²).

### Prerequisites (fresh checkout)

The repo uses a `src/` layout. To reproduce the logged results:

- **Tests and Bow benchmark:** Install the package editable and dev dependencies so `psi_loop` is importable and pytest can collect tests. From repo root:
  ```bash
  python -m pip install -e .[dev]
  ```
  Alternatively set `PYTHONPATH=src` when running the commands below.
- **Dense benchmark:** The dense backend requires the `[dense]` extra (`sentence-transformers`). Install before running the dense run:
  ```bash
  python -m pip install -e .[dense]
  ```

---

### Test suite

From repo root (package installed editable or `PYTHONPATH=src` set):

```
pytest -v
```

- **Result:** 35 passed in 0.12s
- **Status:** All tests passed.

---

### Bow benchmark

From repo root (package installed editable or `PYTHONPATH=src` set):

```
PYTHONPATH=src python scripts/run_baseline_vs_psi0.py --backend bow
```

- **Fixture:** tests/fixtures/benchmark_tasks.json
- **Embedder:** BowEmbedder (backend=bow, model=n/a)

| Metric | Value |
|--------|-------|
| Wins | psi0=4, baseline=0, tie=10 |
| Useful hits | psi0=5, baseline=1 |
| Redundant hits | psi0=7, baseline=12 |
| Expected-match count | 4/14 |
| Decision | refine_v |

**Artifact:** `evaluation_results_baseline_vs_psi0_bow.json`  
**Status:** Matches checked-in artifact.

---

### Dense benchmark

Requires the `[dense]` extra (`pip install -e .[dense]`). From repo root:

```
PYTHONPATH=src python scripts/run_baseline_vs_psi0.py --backend dense
```

- **Fixture:** tests/fixtures/benchmark_tasks.json
- **Embedder:** STEmbedder (backend=dense, model=all-MiniLM-L6-v2)

| Metric | Value |
|--------|-------|
| Wins | psi0=2, baseline=1, tie=11 |
| Useful hits | psi0=2, baseline=1 |
| Redundant hits | psi0=6, baseline=13 |
| Expected-match count | 2/14 |
| Decision | refine_v |

**Notable:** `realistic_roadmap_planning` — winner=baseline; Psi0 selected `unrelated_visual_refresh`, baseline selected `novel_data_contracts`.

**Artifact:** `evaluation_results_baseline_vs_psi0_dense_all-MiniLM-L6-v2.json`  
**Status:** Matches checked-in artifact.

---

### Summary

| Backend | Psi0 wins | Baseline wins | Ties | Psi0 useful | Baseline useful | Psi0 redundant | Baseline redundant |
|---------|----------|---------------|------|-------------|-----------------|-----------------|---------------------|
| BowEmbedder | 4 | 0 | 10 | 5 | 1 | 7 | 12 |
| STEmbedder | 2 | 1 | 11 | 2 | 1 | 6 | 13 |

Outcome matches `evaluation_baseline_vs_psi0.md` and the committed JSON artifacts. No regeneration or doc updates required.
