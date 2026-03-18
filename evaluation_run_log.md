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

From repo root, ensure the package is importable: install editable with `pip install -e .[dev]` or set `PYTHONPATH=src` (see Prerequisites).

```
pytest -v
```

- **Result:** 35 passed in 0.12s
- **Status:** All tests passed.

---

### Bow benchmark

From repo root (package installed editable or `PYTHONPATH=src` set). Replay uses `--json-out` below so the checked-in reference artifact is not overwritten.

```
PYTHONPATH=src python scripts/run_baseline_vs_psi0.py --backend bow --json-out evaluation_results_baseline_vs_psi0_bow_rerun.json
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

**Reference artifact:** `evaluation_results_baseline_vs_psi0_bow.json` (checked-in). Replay writes to `evaluation_results_baseline_vs_psi0_bow_rerun.json`.  
**Status:** Aggregate metrics match the checked-in artifact. Byte-identical only on the same OS (e.g. Windows); on POSIX, `fixture_path` in the JSON uses forward slashes, so a rerun may diff in that field only.

---

### Dense benchmark

Requires the `[dense]` extra: `pip install -e .[dense]` (see Prerequisites). From repo root. Replay uses `--json-out` so the checked-in reference is not overwritten.

```
PYTHONPATH=src python scripts/run_baseline_vs_psi0.py --backend dense --json-out evaluation_results_baseline_vs_psi0_dense_all-MiniLM-L6-v2_rerun.json
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

**Reference artifact:** `evaluation_results_baseline_vs_psi0_dense_all-MiniLM-L6-v2.json` (checked-in). Replay writes to `..._rerun.json`.  
**Status:** Aggregate metrics match the checked-in artifact. Byte-identical only on the same OS; on POSIX, `fixture_path` may use forward slashes.

---

### Summary

| Backend | Psi0 wins | Baseline wins | Ties | Psi0 useful | Baseline useful | Psi0 redundant | Baseline redundant |
|---------|----------|---------------|------|-------------|-----------------|-----------------|---------------------|
| BowEmbedder | 4 | 0 | 10 | 5 | 1 | 7 | 12 |
| STEmbedder | 2 | 1 | 11 | 2 | 1 | 6 | 13 |

Outcome matches `evaluation_baseline_vs_psi0.md` and the committed JSON artifacts (aggregate metrics; path serialization in JSON may differ on POSIX). **Note:** README and `evaluation_baseline_vs_psi0.md` still show benchmark commands that write to the default artifact paths; to avoid overwriting the reference artifacts when replaying, use the `--json-out` commands in this log.

---

## 2025-03-17 — H-tempering experiment

**Change:** Psi0 scoring updated from linear `V × H` to **tempered surprise** `score = V × sqrt(H)`. Same fixture, same reporting. One variable only.

### Test suite

```
pytest -v
```

- **Result:** 35 passed.

### Bow (tempered H)

```
PYTHONPATH=src python scripts/run_baseline_vs_psi0.py --backend bow --json-out evaluation_results_baseline_vs_psi0_bow_tempered_h.json
```

| Metric | Tempered H (this run) | Baseline (linear V×H) |
|--------|------------------------|------------------------|
| Wins | psi0=2, baseline=0, tie=12 | psi0=4, baseline=0, tie=10 |
| Useful hits | psi0=3, baseline=1 | psi0=5, baseline=1 |
| Redundant hits | psi0=10, baseline=12 | psi0=7, baseline=12 |
| Expected-match count | 2/14 | 4/14 |

**Artifacts:** `evaluation_results_baseline_vs_psi0_bow_tempered_h.json`  
**Vs baseline:** Tempered H reduced Psi0 wins (4→2) and useful hits (5→3), increased ties (10→12) and Psi0 redundant hits (7→10). On this frozen benchmark, tempered H weakened the Bow result.

### Dense (tempered H)

```
PYTHONPATH=src python scripts/run_baseline_vs_psi0.py --backend dense --json-out evaluation_results_baseline_vs_psi0_dense_all-MiniLM-L6-v2_tempered_h.json
```

| Metric | Tempered H (this run) | Baseline (linear V×H) |
|--------|------------------------|------------------------|
| Wins | psi0=2, baseline=0, tie=12 | psi0=2, baseline=1, tie=11 |
| Useful hits | psi0=3, baseline=1 | psi0=2, baseline=1 |
| Redundant hits | psi0=8, baseline=13 | psi0=6, baseline=13 |
| Expected-match count | 2/14 | 2/14 |

**Notable:** `realistic_roadmap_planning` — with tempered H, **tie** (both select `novel_data_contracts`). Baseline (linear) had baseline win (Psi0 selected `unrelated_visual_refresh`).

**Artifacts:** `evaluation_results_baseline_vs_psi0_dense_all-MiniLM-L6-v2_tempered_h.json`  
**Vs baseline:** Tempered H removed the single baseline win (roadmap now tie), added one Psi0 useful hit (2→3), added two Psi0 redundant hits (6→8). Net: no baseline wins under tempered H; slight useful-hit gain and redundant-hit cost on dense.

### H-tempering summary

| Backend | Psi0 wins | Baseline wins | Ties | Psi0 useful | Psi0 redundant |
|---------|----------|---------------|------|-------------|----------------|
| Bow (tempered H) | 2 | 0 | 12 | 3 | 10 |
| Bow (linear baseline) | 4 | 0 | 10 | 5 | 7 |
| Dense (tempered H) | 2 | 0 | 12 | 3 | 8 |
| Dense (linear baseline) | 2 | 1 | 11 | 2 | 6 |

**Conclusion:** On the frozen benchmark, tempered H did not improve over linear V×H: Bow regressed (fewer wins, more redundant hits); Dense improved on roadmap (tie instead of baseline win) and useful hits but at higher Psi0 redundant hits. No clear win; linear V×H remains the stronger baseline for Bow on this fixture.
