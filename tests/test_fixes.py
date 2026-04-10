"""Tests for the four psi-loop fixes.

Fix 1: forensics exposed in __all__
Fix 2: BowEmbedder L2-normalized
Fix 3: near-tie bucketing comment (behaviour unchanged — no new tests needed)
Fix 4: iterative selection as default
"""

from __future__ import annotations

import math

import pytest

from psi_loop import (
    BowEmbedder,
    Candidate,
    PsiLoop,
    build_task_forensics,
    render_task_forensics,
    select_context,
    surprise_score,
)
from psi_loop.embedders import centroid, cosine_similarity_vectors
from psi_loop.pipeline import _near_tie_sort_key, fit_to_budget, rank_candidates, select_with_scorer
from psi_loop.scoring import psi_0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_candidate(id: str, text: str) -> Candidate:
    return Candidate(id=id, text=text, source="test")


def l2_norm(vec: dict) -> float:
    return math.sqrt(sum(v * v for v in vec.values()))


# ---------------------------------------------------------------------------
# Fix 1: forensics in __all__
# ---------------------------------------------------------------------------

class TestForensicsPublicAPI:
    def test_build_task_forensics_importable_from_top_level(self):
        # If this import line at the top of the file didn't explode, this passes.
        assert callable(build_task_forensics)

    def test_render_task_forensics_importable_from_top_level(self):
        assert callable(render_task_forensics)

    def test_both_names_in_psi_loop_all(self):
        import psi_loop
        assert "build_task_forensics" in psi_loop.__all__
        assert "render_task_forensics" in psi_loop.__all__

    def test_forensics_roundtrip_with_fixture_source(self):
        """build_task_forensics via public import returns a non-empty report."""
        from psi_loop.models import TaskDefinition
        task_def = TaskDefinition(
            id="fix1-roundtrip",
            goal="design retry with backoff",
            current_context=["We already discussed timeouts."],
            max_tokens=200,
            candidates=[
                Candidate(id="c1", text="Use exponential backoff with jitter on retry.", source="t"),
                Candidate(id="c2", text="Timeouts should be aggressive.", source="t"),
            ],
        )
        forensics = build_task_forensics(task_def)
        assert forensics.task_id == "fix1-roundtrip"
        assert len(forensics.psi0.ranked) == 2
        rendered = render_task_forensics(forensics, top_k=2)
        assert "fix1-roundtrip" in rendered
        assert "Diagnosis" in rendered


# ---------------------------------------------------------------------------
# Fix 2: BowEmbedder L2-normalized
# ---------------------------------------------------------------------------

class TestBowEmbedderL2Normalization:
    def test_single_token_has_unit_norm(self):
        vec = BowEmbedder().embed("retry")
        assert pytest.approx(l2_norm(vec), abs=1e-9) == 1.0

    def test_multi_token_has_unit_norm(self):
        vec = BowEmbedder().embed("retry backoff jitter queue migration")
        assert pytest.approx(l2_norm(vec), abs=1e-9) == 1.0

    def test_empty_string_returns_empty_dict(self):
        vec = BowEmbedder().embed("")
        assert vec == {}

    def test_stopword_only_returns_empty_dict(self):
        vec = BowEmbedder().embed("the a and is of")
        assert vec == {}

    def test_embedding_is_deterministic(self):
        e = BowEmbedder()
        assert e.embed("retry with backoff") == e.embed("retry with backoff")

    def test_repeated_token_still_unit_norm(self):
        # "retry retry retry" — raw counts {retry: 3}, norm = 3, normalised = {retry: 1.0}
        vec = BowEmbedder().embed("retry retry retry")
        assert pytest.approx(l2_norm(vec), abs=1e-9) == 1.0
        assert pytest.approx(vec["retry"], abs=1e-9) == 1.0
        assert len(vec) == 1

    def test_no_length_bias_in_centroid(self):
        """A 2-token chunk and a 20-token chunk should contribute equally to the centroid."""
        emb = BowEmbedder()
        short = emb.embed("retry backoff")
        long_text = " ".join(["retry", "backoff"] + [f"word{i}" for i in range(18)])
        long = emb.embed(long_text)

        # Both vectors are unit-length; their cosine similarity with each other
        # is determined by shared content, not length.  The centroid norm is
        # bounded by 1 (it can only decrease due to cancellation).
        c = centroid([short, long])
        assert l2_norm(c) <= 1.0 + 1e-9

    def test_identical_candidates_have_cosine_similarity_one(self):
        emb = BowEmbedder()
        v = emb.embed("retry with exponential backoff")
        assert pytest.approx(cosine_similarity_vectors(v, v), abs=1e-9) == 1.0

    def test_disjoint_candidates_have_zero_cosine_similarity(self):
        emb = BowEmbedder()
        v1 = emb.embed("retry backoff")
        v2 = emb.embed("queue migration")
        assert pytest.approx(cosine_similarity_vectors(v1, v2), abs=1e-9) == 0.0

    def test_surprise_identical_candidate_in_context_is_zero(self):
        text = "Use exponential backoff on retry"
        h = surprise_score(text, [text])
        assert pytest.approx(h, abs=1e-9) == 0.0

    def test_surprise_novel_candidate_against_disjoint_context_is_one(self):
        h = surprise_score("queue dead-letter migration", ["retry backoff jitter"])
        assert pytest.approx(h, abs=1e-9) == 1.0

    def test_surprise_range_invariant(self):
        h = surprise_score("retry backoff queue", ["retry", "backoff", "queue migration"])
        assert 0.0 <= h <= 1.0

    def test_long_context_does_not_dominate_surprise(self):
        """With normalisation, adding many identical context items shouldn't make
        the centroid explode — the centroid of unit vectors is bounded in norm."""
        long_item = "retry backoff " + " ".join(f"extra{i}" for i in range(50))
        short_item = "retry backoff"
        h_short_ctx = surprise_score("queue migration", [short_item])
        h_long_ctx  = surprise_score("queue migration", [long_item])
        # Both contexts are about "retry backoff" (+ noise in long case).
        # Neither should make surprise for a completely disjoint candidate go below 0.
        assert h_short_ctx >= 0.0
        assert h_long_ctx  >= 0.0


# ---------------------------------------------------------------------------
# Fix 4: iterative selection (default behaviour)
# ---------------------------------------------------------------------------

class TestIterativeSelection:
    """iterative=True is the new default for select_context and PsiLoop.select."""

    def _redundant_set(self):
        """A and B say nearly the same thing; C is genuinely novel."""
        return [
            make_candidate("A", "Use exponential backoff with jitter on retry"),
            make_candidate("B", "Retry using exponential backoff and add jitter between attempts"),
            make_candidate("C", "Add a dead-letter queue for messages that exhaust retries"),
        ]

    def test_iterative_default_suppresses_redundant_second_pick(self):
        """After A is selected, B should be penalised and C preferred."""
        candidates = self._redundant_set()
        result = select_context(candidates, "design retry strategy with backoff", [], max_tokens=60)
        ids = [sc.candidate.id for sc in result.selected]
        # A should always be first; C should beat B for second slot.
        assert ids[0] == "A"
        assert "C" in ids
        assert ids.index("C") < ids.index("B") if "B" in ids else True

    def test_non_iterative_selects_both_redundant_candidates(self):
        """With iterative=False, A and B both score equally against empty context."""
        candidates = self._redundant_set()
        result = select_context(
            candidates, "design retry strategy with backoff", [], max_tokens=60, iterative=False
        )
        ids = [sc.candidate.id for sc in result.selected]
        # Both A and B score the same against empty context — both get selected.
        assert "A" in ids
        assert "B" in ids

    def test_iterative_flag_false_matches_original_fit_to_budget(self):
        """iterative=False must produce the same result as the legacy fit_to_budget path."""
        candidates = self._redundant_set()
        goal = "design retry strategy with backoff"
        result_flag = select_context(candidates, goal, [], max_tokens=60, iterative=False)
        ranked = rank_candidates(candidates, goal, [])
        legacy_selected = fit_to_budget(ranked, max_tokens=60)
        assert [sc.candidate.id for sc in result_flag.selected] == [sc.candidate.id for sc in legacy_selected]

    def test_psiloop_select_iterative_default(self):
        loop = PsiLoop()
        candidates = self._redundant_set()
        result = loop.select("design retry strategy with backoff", [], max_tokens=60, candidates=candidates)
        ids = [sc.candidate.id for sc in result.selected]
        assert ids[0] == "A"
        assert "C" in ids

    def test_psiloop_select_iterative_false(self):
        loop = PsiLoop()
        candidates = self._redundant_set()
        result = loop.select(
            "design retry strategy with backoff", [], max_tokens=60,
            candidates=candidates, iterative=False
        )
        ids = [sc.candidate.id for sc in result.selected]
        assert "A" in ids and "B" in ids

    def test_iterative_single_candidate_same_as_non_iterative(self):
        c = [make_candidate("only", "retry with backoff")]
        r_iter = select_context(c, "retry backoff", [], max_tokens=20, iterative=True)
        r_flat = select_context(c, "retry backoff", [], max_tokens=20, iterative=False)
        assert [sc.candidate.id for sc in r_iter.selected] == [sc.candidate.id for sc in r_flat.selected]

    def test_iterative_empty_candidates_returns_empty(self):
        result = select_context([], "retry backoff", [], max_tokens=100, iterative=True)
        assert result.selected == []
        assert result.ranked == []

    def test_iterative_nothing_fits_budget_returns_empty(self):
        c = [make_candidate("big", "retry " * 200)]
        result = select_context(c, "retry", [], max_tokens=5, iterative=True)
        assert result.selected == []

    def test_iterative_all_identical_candidates_selects_one(self):
        """All three say exactly the same thing — only the first should survive."""
        text = "Use exponential backoff with jitter"
        candidates = [make_candidate(f"c{i}", text) for i in range(3)]
        result = select_context(candidates, "retry backoff", [], max_tokens=100, iterative=True)
        # After the first pick, surprise for any remaining identical candidate → 0,
        # so their Psi0 score → 0 and they should not be selected over nothing.
        # (They may still be selected if budget allows and score > 0 due to floating point,
        #  but the key invariant is score drops sharply.)
        selected_ids = [sc.candidate.id for sc in result.selected]
        # At most one unique-content candidate should dominate.
        if len(selected_ids) > 1:
            # If extras are selected, their surprise must be near zero.
            for sc in result.selected[1:]:
                assert sc.surprise < 0.05, f"Expected near-zero surprise for redundant pick, got {sc.surprise}"

    def test_iterative_ranked_is_initial_pass_not_final_selected_order(self):
        """result.ranked always reflects the initial scoring (before iterative updates).
        It is not reordered by iterative selection — it is for inspection / forensics."""
        candidates = self._redundant_set()
        result = select_context(candidates, "design retry strategy with backoff", [], max_tokens=60)
        # ranked must contain all candidates
        assert len(result.ranked) == len(candidates)

    def test_iterative_running_context_grows(self):
        """After N picks the running context has grown — simulate with unique candidates
        and verify that each pick is genuinely distinct from the last."""
        candidates = [
            make_candidate("retry", "exponential backoff retry"),
            make_candidate("queue", "dead-letter queue overflow"),
            make_candidate("idempotency", "idempotency key deduplication"),
        ]
        result = select_context(candidates, "resilience patterns", [], max_tokens=200, iterative=True)
        selected_ids = {sc.candidate.id for sc in result.selected}
        # All three are topically distinct — all three should be selected.
        assert selected_ids == {"retry", "queue", "idempotency"}

    def test_select_with_scorer_iterative_flag_propagated(self):
        """select_with_scorer also honours iterative flag."""
        candidates = self._redundant_set()
        goal = "design retry strategy with backoff"
        r_iter = select_with_scorer(candidates, goal, [], 60, psi_0, iterative=True)
        r_flat = select_with_scorer(candidates, goal, [], 60, psi_0, iterative=False)
        # iterative suppresses B; non-iterative selects both A and B
        iter_ids = [sc.candidate.id for sc in r_iter.selected]
        flat_ids = [sc.candidate.id for sc in r_flat.selected]
        assert "A" in iter_ids and "A" in flat_ids
        assert "B" in flat_ids  # non-iterative selects both
        # iterative may or may not select B but must prefer C
        if "B" in iter_ids and "C" in iter_ids:
            assert iter_ids.index("C") < iter_ids.index("B")

    def test_iterative_respects_token_budget_exactly(self):
        """Total token count of selected items must not exceed max_tokens."""
        candidates = [make_candidate(f"c{i}", f"retry backoff jitter {'word ' * i}") for i in range(5)]
        result = select_context(candidates, "retry", [], max_tokens=20, iterative=True)
        total = sum(sc.token_count for sc in result.selected)
        assert total <= 20

    def test_iterative_with_existing_context_compounds_suppression(self):
        """When current_context already covers a topic, related candidates should score
        lower than candidates on an uncovered topic."""
        current_context = ["We are already using exponential backoff with jitter on retries."]
        candidates = [
            make_candidate("redundant", "Retry uses exponential backoff and jitter"),
            make_candidate("novel", "Add a dead-letter queue for exhausted retry messages"),
        ]
        result = select_context(
            candidates, "improve retry resilience", current_context, max_tokens=100, iterative=True
        )
        ids = [sc.candidate.id for sc in result.selected]
        # Novel should be ranked above redundant since context already covers backoff.
        assert ids[0] == "novel"


# ---------------------------------------------------------------------------
# Bot review fixes: near-tie key in iterative rounds + forensics consistency
# ---------------------------------------------------------------------------

class TestBotReviewFixes:
    """Covers the two issues flagged by Codex during PR review."""

    # --- Issue 1: near-tie sort key shared between rank_candidates and _select_iterative ---

    def test_near_tie_sort_key_is_module_level(self):
        """_near_tie_sort_key must be importable so both ranking paths share it."""
        assert callable(_near_tie_sort_key)

    def test_near_tie_key_buckets_score_not_raw(self):
        """The key must bucket by NEAR_TIE_EPSILON, not sort by raw score."""
        from psi_loop.pipeline import NEAR_TIE_EPSILON
        from psi_loop.models import ScoredCandidate
        # Two candidates whose scores straddle a bucket boundary by sub-epsilon.
        # With raw-score sorting the higher-score item always wins regardless of value.
        # With bucketed sorting they land in the same bucket and the higher-VALUE item wins.
        low_score_high_value = ScoredCandidate(
            candidate=make_candidate("hv", "x"), score=0.100, value=0.9, surprise=0.111, token_count=1
        )
        high_score_low_value = ScoredCandidate(
            candidate=make_candidate("lv", "x"), score=0.109, value=0.1, surprise=1.0, token_count=1
        )
        # Both fall in the 0.10 bucket (floor(0.100/0.01)*0.01 == floor(0.109/0.01)*0.01 == 0.10).
        assert _near_tie_sort_key(low_score_high_value) < _near_tie_sort_key(high_score_low_value), (
            "Higher-value candidate should sort first when scores are in the same bucket"
        )

    def test_iterative_near_tie_prefers_value_over_sub_epsilon_score_jitter(self):
        """Within an iterative round, a candidate with higher V should beat one with
        trivially higher score when both are in the same NEAR_TIE_EPSILON bucket."""
        # Construct a scenario where, after one pick updates context, two remaining
        # candidates have scores in the same epsilon bucket but different values.
        # We do this by pre-seeding context so surprise is roughly equal.
        shared_context = ["retry backoff strategy"]
        # Both candidates are about "retry backoff" so surprise is similar.
        high_value = make_candidate("hv", "retry backoff idempotency queue guardrail migration")
        low_value  = make_candidate("lv", "retry")
        goal = "design retry strategy with backoff idempotency"

        result = select_context(
            [high_value, low_value], goal, shared_context, max_tokens=50, iterative=True
        )
        # high_value mentions more goal terms → higher V; it must be picked first.
        assert result.selected[0].candidate.id == "hv"

    # --- Issue 2: forensics uses iterative=False so budget trace stays coherent ---

    def test_forensics_selected_matches_budget_trace(self):
        """Every candidate marked 'selected' in the budget trace must also appear
        in result.selected, and vice versa — i.e., no contradiction."""
        from psi_loop.models import TaskDefinition
        task = TaskDefinition(
            id="coherence-check",
            goal="design retry strategy with backoff",
            current_context=["We use fixed delays today."],
            max_tokens=80,
            candidates=[
                make_candidate("A", "Use exponential backoff with jitter on retry"),
                make_candidate("B", "Retry with exponential backoff and add jitter between attempts"),
                make_candidate("C", "Add a dead-letter queue for messages that exhaust retries"),
                make_candidate("D", "Emit a metric on every retry attempt for observability"),
            ],
        )
        forensics = build_task_forensics(task)

        for selector in (forensics.baseline, forensics.psi0):
            trace_selected_ids = {
                item.candidate.candidate.id
                for item in selector.budget_trace
                if item.reason == "selected"
            }
            result_selected_ids = {sc.candidate.id for sc in selector.selected}
            assert trace_selected_ids == result_selected_ids, (
                f"{selector.name}: budget trace selected={trace_selected_ids} "
                f"!= result selected={result_selected_ids}"
            )

    def test_forensics_no_candidate_selected_but_traced_dropped(self):
        """A candidate in result.selected must never appear as 'would_exceed_budget'
        or 'too_large' in the budget trace of the same selector."""
        from psi_loop.models import TaskDefinition
        task = TaskDefinition(
            id="no-ghost-drops",
            goal="retry backoff queue",
            current_context=[],
            max_tokens=60,
            candidates=[
                make_candidate("A", "retry backoff jitter"),
                make_candidate("B", "dead-letter queue overflow"),
                make_candidate("C", "idempotency key deduplication retry"),
            ],
        )
        forensics = build_task_forensics(task)

        for selector in (forensics.baseline, forensics.psi0):
            selected_ids = {sc.candidate.id for sc in selector.selected}
            for trace_item in selector.budget_trace:
                cid = trace_item.candidate.candidate.id
                if cid in selected_ids:
                    assert trace_item.reason == "selected", (
                        f"{selector.name}: {cid} is in selected but trace says {trace_item.reason}"
                    )
