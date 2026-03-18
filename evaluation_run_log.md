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

**Replay not reproducible on this branch:** `scripts/run_baseline_vs_psi0.py` has no scoring-mode flag and always uses linear psi_0 (V×H). Rerunning the script with `--json-out ..._tempered_h.json` would produce linear results, not tempered H, and would silently relabel them as tempered-H artifacts. Do not refresh tempered-H artifacts from this branch; the results in this section are from a prior run with tempered scoring enabled.

### Test suite

```
pytest -v
```

- **Result:** 35 passed.

### Bow (tempered H)

*(Replay not supported — script has no scoring-mode flag; running it would produce linear results. Tempered-H results below are from a prior run.)*

| Metric | Tempered H (this run) | Baseline (linear V×H) |
|--------|------------------------|------------------------|
| Wins | psi0=2, baseline=0, tie=12 | psi0=4, baseline=0, tie=10 |
| Useful hits | psi0=3, baseline=1 | psi0=5, baseline=1 |
| Redundant hits | psi0=10, baseline=12 | psi0=7, baseline=12 |
| Expected-match count | 2/14 | 4/14 |

**Artifacts:** `evaluation_results_baseline_vs_psi0_bow_tempered_h.json`  
**Vs baseline:** Tempered H reduced Psi0 wins (4→2) and useful hits (5→3), increased ties (10→12) and Psi0 redundant hits (7→10). On this frozen benchmark, tempered H weakened the Bow result.

### Dense (tempered H)

*(Replay not supported — script has no scoring-mode flag; running it would produce linear results. Tempered-H results below are from a prior run.)*

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

---

## 2025-03-17 — Planning-structure V bonus experiment

**Change:** Add a small deterministic plan-structure bonus to V: when the goal is planning-shaped (tokens: plan, roadmap, rollout, migration, migrate, timeline, phase), candidate value is `V' = clamp(V_base + α·S_plan, 0, 1)` with α=0.12 and S_plan from bucketed cue coverage (sequencing, dependency, planning/risk). Same fixture, same reporting.

### Test suite

```
pytest -v
```

- **Result:** 38 passed.

### Bow (plan bonus)

```
PYTHONPATH=src python scripts/run_baseline_vs_psi0.py --backend bow --json-out evaluation_results_baseline_vs_psi0_bow_plan_bonus.json
```

| Metric | Plan bonus (this run) | Linear baseline |
|--------|------------------------|-----------------|
| Wins | psi0=4, baseline=0, tie=10 | psi0=4, baseline=0, tie=10 |
| Useful hits | psi0=5, baseline=1 | psi0=5, baseline=1 |
| Redundant hits | psi0=7, baseline=12 | psi0=7, baseline=12 |
| Expected-match count | 4/14 | 4/14 |

**Artifacts:** `evaluation_results_baseline_vs_psi0_bow_plan_bonus.json`  
**Vs baseline:** No change. Bow metrics identical to linear baseline. `realistic_roadmap_planning` remains tie (both select `novel_data_contracts`). No regression.

### Dense (plan bonus)

```
PYTHONPATH=src python scripts/run_baseline_vs_psi0.py --backend dense --json-out evaluation_results_baseline_vs_psi0_dense_all-MiniLM-L6-v2_plan_bonus.json
```

| Metric | Plan bonus (this run) | Linear baseline |
|--------|------------------------|-----------------|
| Wins | psi0=2, baseline=1, tie=11 | psi0=2, baseline=1, tie=11 |
| Useful hits | psi0=2, baseline=1 | psi0=2, baseline=1 |
| Redundant hits | psi0=6, baseline=13 | psi0=6, baseline=13 |
| Expected-match count | 2/14 | 2/14 |

**Notable:** `realistic_roadmap_planning` — still **baseline win** (Psi0 selects `unrelated_visual_refresh`, baseline selects `novel_data_contracts`). The gold useful candidate text does not contain the current bucket cues (sequencing/dependency/risk), so S_plan=0 for it; bonus did not apply.

**Artifacts:** `evaluation_results_baseline_vs_psi0_dense_all-MiniLM-L6-v2_plan_bonus.json`  
**Vs baseline:** No change. Dense metrics identical to linear baseline.

### Plan-bonus summary

| Backend | Psi0 wins | Baseline wins | Ties | Psi0 useful | Psi0 redundant |
|---------|----------|---------------|------|-------------|----------------|
| Bow (plan bonus) | 4 | 0 | 10 | 5 | 7 |
| Bow (linear) | 4 | 0 | 10 | 5 | 7 |
| Dense (plan bonus) | 2 | 1 | 11 | 2 | 6 |
| Dense (linear) | 2 | 1 | 11 | 2 | 6 |

**Conclusion:** Plan-structure bonus produced no regression on Bow and no change on Dense. It did not improve the roadmap outcome on dense because the gold useful candidate does not contain the current bucketed cues. Success criteria (roadmap stable Psi0 win on both backends, dense useful +1 without redundant +1) were not met. The bonus is active and unit-tested; next step could be to extend bucket cues so planning-shaped useful notes (e.g. “investments”, “missing”, “concrete”) are rewarded, or to keep the change as a harmless nudge and iterate on cue design later.

### Tokenization fix (rerun)

Cue sets were originally raw strings (e.g. `depends`, `requires`, `blocked`), while candidates are matched using `tokenize()`, which stems (e.g. `depends` → `depend`, `blocked` → `block`). That caused dependency/blocker cues to rarely match. **Fix:** build all plan-bonus cue sets via `_normalized_cue_set(words)` so they contain the same normalized tokens as `tokenize()` output. Reran Bow and Dense with the fix.

- **Bow:** psi0=4, baseline=0, tie=10; useful 5, redundant 7. Unchanged vs pre-fix (no regression).
- **Dense:** psi0=2, baseline=1, tie=11; useful 2, redundant 6. Unchanged; `realistic_roadmap_planning` still baseline win (gold candidate has no sequencing/dependency/risk tokens, so S_plan=0 for it). The fix ensures cues like “depends”/“requires” are now detected when present; the roadmap gold text still does not contain them, so outcome unchanged.

### P2 guard + planning trigger (rerun)

Two fixes applied: (a) **Guard:** plan bonus is applied only when `v_base > 0`, so candidates with zero goal overlap no longer receive value purely from plan-structure cues. (b) **Planning trigger:** "planning" added to the goal trigger list so that tokenize("planning") yields "plann" and goals like "Select planning notes for …" activate the bonus. Reran full suite (40 passed), Bow and Dense benchmarks.

- **Bow:** psi0=4, baseline=0, tie=10; useful 5, redundant 7. Unchanged (no regression).
- **Dense:** psi0=2, baseline=1, tie=11; useful 2, redundant 6. Unchanged; `realistic_roadmap_planning` still baseline win. Roadmap outcome unchanged.
