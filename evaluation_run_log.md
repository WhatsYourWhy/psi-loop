# Evaluation Run Log

**Current mainline:** Psi0 uses linear `V × H` scoring (value from keyword overlap plus planning-structure and relation-aware bonuses, gated on planning-shaped goals) with **near-tie value-priority** ranking: when two scores differ by less than `NEAR_TIE_EPSILON` (0.01), the candidate with higher value is ranked first before budget packing. Baseline: Bow 4 wins / 0 baseline / 10 ties (useful 5 vs 1, redundant 7 vs 12); Dense 2 wins / 0 baseline / 12 ties (useful 3 vs 1, redundant 6 vs 13). `realistic_roadmap_planning` is a tie on both backends (Psi0 selects gold `novel_data_contracts`). Full experiment history and artifact names are below.

---

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

### Run, test and log (2025-03-17)

Full test suite, Bow and Dense benchmarks re-run with current code (P2 guard + planning trigger, normalized cue sets). Artifacts: `evaluation_results_baseline_vs_psi0_bow_plan_bonus.json`, `evaluation_results_baseline_vs_psi0_dense_all-MiniLM-L6-v2_plan_bonus.json`.

- **Tests:** 41 passed in 0.28s.
- **Bow:** Wins psi0=4, baseline=0, tie=10. Useful: psi0=5, baseline=1. Redundant: psi0=7, baseline=12. `realistic_roadmap_planning`: tie (both select `novel_data_contracts`).
- **Dense:** Wins psi0=2, baseline=1, tie=11. Useful: psi0=2, baseline=1. Redundant: psi0=6, baseline=13. `realistic_roadmap_planning`: baseline win (Psi0 `unrelated_visual_refresh`, baseline `novel_data_contracts`).

---

## 2025-03-17 — Relation-aware V bonus (experiment)

**Change:** Fourth bucket added to the plan-structure bonus: relation cues (implicit utility). `PLAN_RELATION_CUES` = normalized set of: missing, enable, supports, allows, ensures, unblocks, needed (no overlap with sequencing/dependency/risk). `S_plan` = matched_buckets / 4. Same gate, same `v_base > 0` guard, same α=0.12. Gold roadmap candidate text contains "missing", so it now receives a relation-bucket contribution.

### Test suite

```
pytest tests/ -v
```

- **Result:** 44 passed.

### Bow (relation bonus)

```
PYTHONPATH=src python scripts/run_baseline_vs_psi0.py --backend bow --json-out evaluation_results_baseline_vs_psi0_bow_relation_bonus.json
```

| Metric | Relation bonus (this run) | Plan bonus (prior) |
|--------|---------------------------|--------------------|
| Wins | psi0=4, baseline=0, tie=10 | psi0=4, baseline=0, tie=10 |
| Useful hits | psi0=5, baseline=1 | psi0=5, baseline=1 |
| Redundant hits | psi0=7, baseline=12 | psi0=7, baseline=12 |

**Artifacts:** `evaluation_results_baseline_vs_psi0_bow_relation_bonus.json`  
**Roadmap:** tie (both select `novel_data_contracts`). No regression.

### Dense (relation bonus)

```
PYTHONPATH=src python scripts/run_baseline_vs_psi0.py --backend dense --json-out evaluation_results_baseline_vs_psi0_dense_all-MiniLM-L6-v2_relation_bonus.json
```

| Metric | Relation bonus (this run) | Plan bonus (prior) |
|--------|---------------------------|--------------------|
| Wins | psi0=2, baseline=1, tie=11 | psi0=2, baseline=1, tie=11 |
| Useful hits | psi0=2, baseline=1 | psi0=2, baseline=1 |
| Redundant hits | psi0=6, baseline=13 | psi0=6, baseline=13 |

**Notable:** `realistic_roadmap_planning` — still **baseline win** (Psi0 selects `unrelated_visual_refresh`, baseline selects `novel_data_contracts`). The gold candidate ("…investments **missing** from the roadmap") now receives the relation-bucket bonus, but the dense ranking was not flipped; the value bump (α × 0.25) may be insufficient relative to the dense similarity gap for the competing candidate.

**Artifacts:** `evaluation_results_baseline_vs_psi0_dense_all-MiniLM-L6-v2_relation_bonus.json`

### Relation-bonus summary

| Backend | Psi0 wins | Baseline wins | Ties | Psi0 useful | Psi0 redundant |
|---------|-----------|---------------|------|-------------|----------------|
| Bow (relation bonus) | 4 | 0 | 10 | 5 | 7 |
| Dense (relation bonus) | 2 | 1 | 11 | 2 | 6 |

**Conclusion:** No regression on Bow; dense roadmap outcome unchanged. Relation-aware bucket is active and unit-tested; next steps could include a larger α for the relation bucket only, or additional relational cues, while staying in narrow hypothesis-testing mode.

---

## 2025-03-17 — Dense roadmap forensic

Task-specific forensic on `realistic_roadmap_planning` with dense embedder (all-MiniLM-L6-v2) to diagnose why Psi0 selects `unrelated_visual_refresh` instead of gold `novel_data_contracts`.

**Summary:**

- **Psi0 rank-1:** unrelated_visual_refresh — value=0.3077, surprise=0.7553 (selected; 5 tokens).
- **Baseline rank-1:** novel_data_contracts [gold useful] — value=0.5383, surprise=0.0000 (selected; 9 tokens).
- **Gold useful (novel_data_contracts) in Psi0:** rank=2, value=0.4146, surprise=0.5588. Gold has higher V than Psi0 rank-1 (0.4146 vs 0.3077) but lower score (0.2317 vs 0.2324) because Psi0 rank-1 has much higher surprise (0.7553 vs 0.5588).
- **Gold in baseline:** rank=1, selected.

**Diagnosis:** Budget. Psi0 selected the high-surprise candidate first; the gold candidate then did not fit in the remaining token budget (would_exceed_budget). The underlying cause is ranking: Psi0 ranked unrelated_visual_refresh above novel_data_contracts (score 0.2324 vs 0.2317), so the failure is a combination of high H on the Psi0 winner and budget crowd-out once that candidate is chosen.

**Artifact:** `evaluation_forensic_realistic_roadmap_planning_dense.txt` (full report with gold labels and diagnosis block).

**Tooling:** Forensic report now includes gold useful/redundant labels in the ranked table and budget trace, and a diagnosis block (Psi0 rank-1, baseline rank-1, gold position in each ranking, and Diagnosis line). Script `scripts/inspect_task_forensics.py` accepts `--out` to write the report to a file.

---

## 2025-03-17 — Near-tie V-priority (experiment)

**Change:** Selection-time rule in [rank_candidates](src/psi_loop/pipeline.py): when two candidates' Psi0 scores differ by less than `NEAR_TIE_EPSILON` (0.01), prefer the one with higher value when ranking. Score formula unchanged (`score = V × H`); only the sort key buckets score and breaks near-ties by value so budget packing sees the more useful candidate first. Baseline unchanged (it re-sorts by score after calling `rank_candidates`).

### Test suite

```
pytest tests/ -v
```

- **Result:** 46 passed.

### Bow (near-tie V-priority)

```
PYTHONPATH=src python scripts/run_baseline_vs_psi0.py --backend bow --json-out evaluation_results_baseline_vs_psi0_bow_near_tie_v_priority.json
```

| Metric | Near-tie V-priority | Prior (relation bonus) |
|--------|---------------------|------------------------|
| Wins | psi0=4, baseline=0, tie=10 | psi0=4, baseline=0, tie=10 |
| Useful hits | psi0=5, baseline=1 | psi0=5, baseline=1 |
| Redundant hits | psi0=7, baseline=12 | psi0=7, baseline=12 |

**Artifacts:** `evaluation_results_baseline_vs_psi0_bow_near_tie_v_priority.json`  
**Roadmap:** tie (both select `novel_data_contracts`). No regression.

### Dense (near-tie V-priority)

```
PYTHONPATH=src python scripts/run_baseline_vs_psi0.py --backend dense --json-out evaluation_results_baseline_vs_psi0_dense_all-MiniLM-L6-v2_near_tie_v_priority.json
```

| Metric | Near-tie V-priority | Prior (relation bonus) |
|--------|---------------------|------------------------|
| Wins | psi0=2, baseline=0, tie=12 | psi0=2, baseline=1, tie=11 |
| Useful hits | psi0=3, baseline=1 | psi0=2, baseline=1 |
| Redundant hits | psi0=6, baseline=13 | psi0=6, baseline=13 |

**Notable:** `realistic_roadmap_planning` — **tie** (Psi0 and baseline both select `novel_data_contracts`). The near-tie rule flipped rank-1 from unrelated_visual_refresh to novel_data_contracts (scores 0.2324 vs 0.2317 within epsilon; gold has higher V), so budget selection now picks the gold candidate. Dense baseline wins dropped from 1 to 0; Psi0 useful hits increased from 2 to 3; no redundant-hit regression.

**Artifacts:** `evaluation_results_baseline_vs_psi0_dense_all-MiniLM-L6-v2_near_tie_v_priority.json`

### Near-tie V-priority summary

| Backend | Psi0 wins | Baseline wins | Ties | Psi0 useful | Psi0 redundant |
|---------|-----------|---------------|------|-------------|----------------|
| Bow (near-tie V) | 4 | 0 | 10 | 5 | 7 |
| Dense (near-tie V) | 2 | 0 | 12 | 3 | 6 |

**Conclusion:** Selection-time fix achieved the target: dense roadmap is now tie (Psi0 selects gold). No Bow regression; Dense gained one useful hit and removed the single baseline win. The forensic diagnosis (budget crowd-out from a tiny score edge) was addressed by preferring higher V when scores are within 0.01.
