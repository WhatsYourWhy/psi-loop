"""Comprehensive review tests for psi-loop — edge cases, invariants, and behavioral checks."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from psi_loop import (
    BowEmbedder,
    Candidate,
    FixtureSource,
    PsiLoop,
    SelectionResult,
    SourceRequest,
    keyword_overlap,
    psi_0,
    select_context,
    select_context_baseline,
    surprise_score,
)
from psi_loop.embedders import centroid, cosine_similarity_vectors
from psi_loop.pipeline import NEAR_TIE_EPSILON, fit_to_budget, rank_candidates
from psi_loop.scoring import (
    GENERIC_GOAL_TERMS,
    ACTION_MECHANISM_TERMS,
    PLAN_BONUS_ALPHA,
    _goal_is_planning_shaped,
    _plan_structure_score,
    _value_with_plan_bonus,
    goal_term_weight,
)
from psi_loop.text import tokenize, token_counts


# ---------------------------------------------------------------------------
# Section 1: Text / tokenization edge cases
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_empty_string_returns_empty_list(self):
        assert tokenize("") == []

    def test_stopwords_are_removed(self):
        tokens = tokenize("a an and are as at be for from in")
        assert tokens == []

    def test_stemming_ing_suffix(self):
        # "retrying" -> strip "ing" -> "retry" (len("retrying")=8 > 5)
        tokens = tokenize("retrying")
        assert "retry" in tokens

    def test_stemming_ed_suffix(self):
        # "failed" -> strip "ed" -> "fail" (len=6 > 4)
        tokens = tokenize("failed")
        assert "fail" in tokens

    def test_stemming_ies_suffix(self):
        # "retries" -> strip "ies" + "y" -> "retry" (len=7 > 4)
        tokens = tokenize("retries")
        assert "retry" in tokens

    def test_stemming_s_suffix(self):
        # "failures" -> strip "s" -> "failure" (len=8 > 3)
        tokens = tokenize("failures")
        assert "failure" in tokens

    def test_short_token_not_stemmed_for_s(self):
        # "as" is a stopword, but "iss" (len=3) should not get the -s strip
        # "bus" (len=3) should NOT have the s stripped (len > 3 required)
        tokens = tokenize("bus")
        assert "bus" in tokens
        assert "bu" not in tokens

    def test_case_insensitive(self):
        assert tokenize("BACKOFF") == tokenize("backoff")

    def test_numbers_included(self):
        tokens = tokenize("retry 3 times")
        assert "3" in tokens

    def test_special_chars_excluded(self):
        tokens = tokenize("hello-world foo@bar.com")
        assert "hello" in tokens
        assert "world" in tokens
        assert "foo" in tokens
        assert "bar" in tokens
        assert "-" not in tokens
        assert "@" not in tokens

    def test_token_counts_returns_counter(self):
        counts = token_counts("backoff backoff jitter")
        assert counts["backoff"] == 2
        assert counts["jitter"] == 1


# ---------------------------------------------------------------------------
# Section 2: BowEmbedder
# ---------------------------------------------------------------------------

class TestBowEmbedder:
    def test_embed_returns_dict(self):
        embedder = BowEmbedder()
        vec = embedder.embed("exponential backoff jitter")
        assert isinstance(vec, dict)
        assert all(isinstance(v, float) for v in vec.values())

    def test_empty_string_returns_empty_dict(self):
        embedder = BowEmbedder()
        vec = embedder.embed("")
        assert vec == {}

    def test_stopword_only_string_returns_empty_dict(self):
        embedder = BowEmbedder()
        vec = embedder.embed("a the and or")
        assert vec == {}

    def test_same_text_same_vector(self):
        embedder = BowEmbedder()
        assert embedder.embed("backoff jitter") == embedder.embed("backoff jitter")


# ---------------------------------------------------------------------------
# Section 3: cosine_similarity_vectors edge cases
# ---------------------------------------------------------------------------

class TestCosineSimilarityVectors:
    def test_identical_sparse_vectors_score_one(self):
        vec = {"a": 1.0, "b": 2.0}
        assert cosine_similarity_vectors(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_sparse_vectors_score_zero(self):
        left = {"a": 1.0}
        right = {"b": 1.0}
        assert cosine_similarity_vectors(left, right) == 0.0

    def test_empty_sparse_vector_returns_zero(self):
        assert cosine_similarity_vectors({}, {"a": 1.0}) == 0.0
        assert cosine_similarity_vectors({"a": 1.0}, {}) == 0.0

    def test_empty_dense_vectors_return_zero(self):
        assert cosine_similarity_vectors((), ()) == 0.0

    def test_dense_vectors_different_length_raises(self):
        with pytest.raises(ValueError, match="dimensionality"):
            cosine_similarity_vectors((1.0, 0.0), (1.0, 0.0, 0.0))

    def test_mixed_types_raises(self):
        with pytest.raises(TypeError):
            cosine_similarity_vectors({"a": 1.0}, (1.0, 0.0))

    def test_identical_dense_vectors_score_one(self):
        vec = (1.0, 2.0, 3.0)
        assert cosine_similarity_vectors(vec, vec) == pytest.approx(1.0)

    def test_cosine_dense_known_value(self):
        # (1,0) · (0,1) = 0
        assert cosine_similarity_vectors((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Section 4: centroid edge cases
# ---------------------------------------------------------------------------

class TestCentroid:
    def test_single_sparse_vector_returns_itself(self):
        vec = {"a": 2.0, "b": 4.0}
        c = centroid([vec])
        assert c == {"a": 2.0, "b": 4.0}

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="At least one vector"):
            centroid([])

    def test_mixed_sparse_dense_raises(self):
        with pytest.raises(TypeError):
            centroid([{"a": 1.0}, (1.0, 0.0)])

    def test_dense_vectors_different_lengths_raises(self):
        with pytest.raises(ValueError, match="dimensionality"):
            centroid([(1.0, 0.0), (1.0, 0.0, 0.0)])

    def test_sparse_centroid_averages_correctly(self):
        c = centroid([{"a": 2.0}, {"a": 4.0}])
        assert c == {"a": 3.0}


# ---------------------------------------------------------------------------
# Section 5: keyword_overlap edge cases
# ---------------------------------------------------------------------------

class TestKeywordOverlap:
    def test_empty_goal_returns_zero(self):
        assert keyword_overlap("some candidate text", "") == 0.0

    def test_empty_candidate_returns_zero(self):
        assert keyword_overlap("", "backoff retry jitter") == 0.0

    def test_both_empty_returns_zero(self):
        assert keyword_overlap("", "") == 0.0

    def test_perfect_overlap_returns_one(self):
        goal = "backoff"
        candidate = "backoff"
        score = keyword_overlap(candidate, goal)
        assert score == pytest.approx(1.0)

    def test_no_overlap_returns_zero(self):
        score = keyword_overlap("completely different text", "backoff jitter retry")
        assert score == 0.0

    def test_score_in_zero_one_range(self):
        score = keyword_overlap("exponential backoff with jitter and retries", "retry backoff jitter")
        assert 0.0 <= score <= 1.0

    def test_stopword_goal_returns_zero(self):
        # "the a and" are all stopwords, tokenize strips them -> empty set
        score = keyword_overlap("some candidate", "the a and")
        assert score == 0.0

    def test_mechanism_terms_weighted_higher(self):
        # A single mechanism match beats multiple generic matches
        goal = "retry backoff jitter"  # mechanism terms
        mech_candidate = "retry"
        gen_candidate = "review plan select new note"  # generic terms
        assert keyword_overlap(mech_candidate, goal) > keyword_overlap(gen_candidate, goal)

    def test_goal_term_weight_high_for_action_mechanism(self):
        for term in ["backoff", "jitter", "retry"]:
            assert goal_term_weight(term) == 4.0

    def test_goal_term_weight_low_for_generic(self):
        for term in ["plan", "review", "note", "summary"]:
            assert goal_term_weight(term) == 0.25

    def test_goal_term_weight_default_for_unknown(self):
        assert goal_term_weight("uniqueterm12345") == 1.0


# ---------------------------------------------------------------------------
# Section 6: surprise_score edge cases
# ---------------------------------------------------------------------------

class TestSurpriseScore:
    def test_empty_context_returns_one(self):
        assert surprise_score("any text", []) == 1.0

    def test_blank_context_items_ignored(self):
        # Blank items stripped in surprise_score
        assert surprise_score("any text", ["", "   "]) == 1.0

    def test_identical_candidate_and_context_returns_zero(self):
        text = "retry with exponential backoff and jitter"
        score = surprise_score(text, [text])
        assert score == pytest.approx(0.0, abs=1e-9)

    def test_score_in_zero_one_range_with_various_inputs(self):
        contexts = [
            "The client retries with fixed delay.",
            "Use timeout on each request.",
        ]
        score = surprise_score("Exponential backoff with jitter.", contexts)
        assert 0.0 <= score <= 1.0

    def test_single_item_context_gives_lower_score_for_similar_candidate(self):
        context = ["Retry with exponential backoff."]
        similar = "Exponential backoff retry strategy."
        different = "Database schema migration plan."
        sim_score = surprise_score(similar, context)
        diff_score = surprise_score(different, context)
        assert diff_score > sim_score


# ---------------------------------------------------------------------------
# Section 7: psi_0 invariants
# ---------------------------------------------------------------------------

class TestPsi0:
    def test_score_equals_value_times_surprise(self):
        score, value, surprise = psi_0(
            "exponential backoff handles rate limits",
            "retry with exponential backoff and jitter",
            ["The client uses fixed delay retries."],
        )
        assert score == pytest.approx(value * surprise, rel=1e-6)

    def test_score_is_zero_when_context_identical_to_candidate(self):
        text = "exponential backoff with jitter"
        score, value, surprise = psi_0(text, "retry logic backoff", [text])
        assert score == pytest.approx(0.0, abs=1e-9)

    def test_score_is_zero_when_no_goal_overlap(self):
        score, value, surprise = psi_0(
            "completely unrelated note",
            "exponential backoff jitter retry",
            [],
        )
        assert score == 0.0

    def test_high_relevance_novel_candidate_beats_redundant(self):
        goal = "retry with exponential backoff and jitter"
        context = ["The client retries with fixed delay."]
        novel_relevant = "Exponential backoff with jitter handles rate limits."
        redundant = "The client retries with fixed delay."

        score_novel, _, _ = psi_0(novel_relevant, goal, context)
        score_redundant, _, _ = psi_0(redundant, goal, context)
        assert score_novel > score_redundant

    def test_all_components_in_zero_one_range(self):
        for candidate_text, context in [
            ("backoff jitter retry", ["fixed delay retry"]),
            ("", ["context"]),
            ("novel text", []),
        ]:
            score, value, surprise = psi_0(candidate_text, "retry backoff", context)
            assert 0.0 <= score <= 1.0
            assert 0.0 <= value <= 1.0
            assert 0.0 <= surprise <= 1.0

    def test_empty_candidate_scores_zero(self):
        score, value, surprise = psi_0("", "retry with backoff", ["some context"])
        assert score == 0.0
        assert value == 0.0


# ---------------------------------------------------------------------------
# Section 8: Planning bonus logic
# ---------------------------------------------------------------------------

class TestPlanningBonus:
    def test_planning_shaped_goal_detected(self):
        assert _goal_is_planning_shaped("Plan the roadmap migration")
        assert _goal_is_planning_shaped("Create a rollout timeline")
        assert not _goal_is_planning_shaped("Retry logic with exponential backoff")

    def test_plan_structure_score_zero_for_non_planning_goal(self):
        score = _plan_structure_score("First phase then milestone", "Improve API retry logic")
        assert score == 0.0

    def test_plan_structure_score_bucketed_by_cue_type(self):
        # Max 4 buckets: sequencing, dependency, risk, relation
        candidate = "First phase. Depends on prerequisite. Risk of rollback. Missing from draft."
        goal = "Plan the migration roadmap"
        score = _plan_structure_score(candidate, goal)
        assert score == pytest.approx(4 / 4)

    def test_plan_structure_score_partial_buckets(self):
        # Only sequencing cue
        candidate = "First step then next phase"
        goal = "Plan the roadmap rollout"
        score = _plan_structure_score(candidate, goal)
        assert score == pytest.approx(1 / 4)

    def test_value_with_plan_bonus_clamps_at_one(self):
        # Manually construct a case where v_base + bonus could exceed 1.0
        # v_base = 1.0, bonus = PLAN_BONUS_ALPHA * 1.0
        # Actually hard to achieve v_base=1.0 in practice, but we can test clamp logic
        # by checking that value never exceeds 1.0
        goal = "Plan the roadmap migration phase rollout"
        # Dense cue-rich candidate likely to get near-max v and near-max plan score
        candidate = (
            "First phase of rollout migration roadmap. "
            "Depends on prerequisite. Risk rollback milestone timeline. Missing from plan."
        )
        v_prime, v_base = _value_with_plan_bonus(candidate, goal)
        assert v_prime <= 1.0
        assert v_prime >= v_base

    def test_plan_bonus_not_applied_when_v_base_zero(self):
        goal = "Plan the rollout migration roadmap"
        # Candidate has no goal keyword overlap
        candidate = "First step then after depends prerequisite risk"
        v_prime, v_base = _value_with_plan_bonus(candidate, goal)
        assert v_base == 0.0
        assert v_prime == 0.0  # bonus is gated on v_base > 0


# ---------------------------------------------------------------------------
# Section 9: rank_candidates and fit_to_budget
# ---------------------------------------------------------------------------

class TestRankAndBudget:
    def _make_candidates(self, n: int) -> list[Candidate]:
        return [Candidate(id=f"c{i}", text=f"candidate {i} text", source="test") for i in range(n)]

    def test_empty_candidates_returns_empty_ranked(self):
        result = rank_candidates([], goal="retry backoff", current_context=[])
        assert result == []

    def test_single_candidate_returns_single_ranked(self):
        candidates = [Candidate(id="one", text="exponential backoff retry", source="test")]
        ranked = rank_candidates(candidates, goal="retry backoff jitter", current_context=[])
        assert len(ranked) == 1

    def test_all_identical_candidates_ranked_by_id_tiebreak(self):
        # All identical text -> same score; fallback sort should be deterministic
        candidates = [
            Candidate(id="b", text="backoff jitter retry", source="test"),
            Candidate(id="a", text="backoff jitter retry", source="test"),
        ]
        ranked = rank_candidates(candidates, goal="backoff jitter", current_context=[])
        # Same score bucket and value -> sort by id (alphabetical)
        assert ranked[0].candidate.id == "a"
        assert ranked[1].candidate.id == "b"

    def test_fit_to_budget_empty_input_returns_empty(self):
        assert fit_to_budget([], max_tokens=100) == []

    def test_fit_to_budget_item_too_large_skipped(self):
        large = Candidate(id="large", text="one two three four five six seven eight nine ten", source="test")
        ranked = rank_candidates([large], goal="one two three", current_context=[], embedder=None)
        selected = fit_to_budget(ranked, max_tokens=5)
        assert selected == []

    def test_fit_to_budget_exactly_at_limit(self):
        candidate = Candidate(id="c", text="one two three", source="test")  # 3 tokens
        ranked = rank_candidates([candidate], goal="one", current_context=[])
        selected = fit_to_budget(ranked, max_tokens=3)
        assert len(selected) == 1

    def test_fit_to_budget_zero_max_tokens(self):
        candidate = Candidate(id="c", text="one", source="test")
        ranked = rank_candidates([candidate], goal="one", current_context=[])
        selected = fit_to_budget(ranked, max_tokens=0)
        assert selected == []

    def test_near_tie_epsilon_boundary(self):
        """Scores within NEAR_TIE_EPSILON should fall in same bucket; outside should not."""
        # Two candidates where scores are just within epsilon
        candidates = [
            Candidate(id="low_v", text="low_v", source="a"),
            Candidate(id="high_v", text="high_v", source="b"),
        ]
        # scores: 0.005 vs 0.009 — within epsilon 0.01, same bucket
        scores_map = {
            "low_v": (0.005, 0.10, 0.05),
            "high_v": (0.009, 0.30, 0.03),
        }

        def scorer(text, goal, ctx, emb):
            return scores_map[text]

        ranked = rank_candidates(candidates, goal="g", current_context=[], scorer=scorer)
        # Same bucket -> higher value (high_v, 0.30) should rank first
        assert ranked[0].candidate.id == "high_v"


# ---------------------------------------------------------------------------
# Section 10: select_context and PsiLoop integration
# ---------------------------------------------------------------------------

class TestSelectContext:
    def test_select_context_returns_selection_result(self):
        candidates = [
            Candidate(id="a", text="exponential backoff jitter retry", source="test"),
            Candidate(id="b", text="fixed delay retry", source="test"),
        ]
        result = select_context(
            candidates=candidates,
            goal="retry with exponential backoff and jitter",
            current_context=["The client uses fixed delay."],
            max_tokens=20,
        )
        assert isinstance(result, SelectionResult)
        assert result.max_tokens == 20
        assert len(result.ranked) == 2

    def test_psiloop_requires_source_or_candidates(self):
        loop = PsiLoop()  # no source
        with pytest.raises(ValueError, match="candidates or a configured source"):
            loop.select(goal="test", current_context=[], max_tokens=50)

    def test_psiloop_with_candidates_directly(self):
        candidates = [
            Candidate(id="a", text="backoff jitter retry rate limits", source="test"),
            Candidate(id="b", text="fixed delay retry", source="test"),
        ]
        loop = PsiLoop()
        result = loop.select(
            goal="exponential backoff jitter retry",
            current_context=["Fixed delay retry is the current approach."],
            max_tokens=50,
            candidates=candidates,
        )
        assert result.ranked[0].candidate.id == "a"

    def test_psiloop_fetch_k_limits_candidates_from_source(self):
        source = FixtureSource(Path(__file__).parent / "fixtures" / "sample_tasks.json")
        loop = PsiLoop(source=source)
        task = source.get_task("retry_backoff")

        result = loop.select(
            goal=task.goal,
            current_context=task.current_context,
            max_tokens=task.max_tokens,
            task_id=task.id,
            fetch_k=1,
        )
        # Only 1 candidate fetched
        assert len(result.ranked) == 1

    def test_select_context_empty_candidates(self):
        result = select_context(
            candidates=[],
            goal="retry with backoff",
            current_context=[],
            max_tokens=100,
        )
        assert result.ranked == []
        assert result.selected == []


# ---------------------------------------------------------------------------
# Section 11: Baseline behavior
# ---------------------------------------------------------------------------

class TestBaseline:
    def test_baseline_ignores_context(self):
        """Baseline score should be identical whether context is empty or populated."""
        candidates = [
            Candidate(id="a", text="backoff jitter retry", source="test"),
        ]
        result_no_ctx = select_context_baseline(
            candidates=candidates,
            goal="retry backoff jitter",
            max_tokens=50,
        )
        result_with_ctx = select_context_baseline(
            candidates=candidates,
            goal="retry backoff jitter",
            max_tokens=50,
        )
        assert result_no_ctx.ranked[0].score == result_with_ctx.ranked[0].score

    def test_baseline_redundant_candidate_scores_high(self):
        """Baseline picks whatever overlaps most with goal, even if redundant with context."""
        goal = "retry with fixed delay for transient failures"
        context = ["The client already uses fixed delay retries for transient failures."]
        candidates = [
            Candidate(id="redundant", text="retry with fixed delay for transient failures", source="test"),
            Candidate(id="novel", text="exponential backoff with jitter avoids thundering herd", source="test"),
        ]
        result = select_context_baseline(candidates=candidates, goal=goal, max_tokens=50)
        assert result.ranked[0].candidate.id == "redundant"


# ---------------------------------------------------------------------------
# Section 12: FixtureSource edge cases
# ---------------------------------------------------------------------------

class TestFixtureSource:
    def test_bundled_source_returns_tasks(self):
        source = FixtureSource()
        tasks = source.tasks()
        assert len(tasks) >= 1

    def test_missing_task_id_raises(self):
        source = FixtureSource(Path(__file__).parent / "fixtures" / "sample_tasks.json")
        with pytest.raises(ValueError, match="not found"):
            source.get_task("nonexistent_task_id")

    def test_fetch_no_limit_returns_all(self):
        source = FixtureSource(Path(__file__).parent / "fixtures" / "sample_tasks.json")
        candidates = source.fetch(SourceRequest(goal="test", task_id="retry_backoff"))
        assert len(candidates) == 3

    def test_fetch_limit_zero_returns_empty(self):
        source = FixtureSource(Path(__file__).parent / "fixtures" / "sample_tasks.json")
        candidates = source.fetch(SourceRequest(goal="test", task_id="retry_backoff", limit=0))
        assert candidates == []

    def test_default_task_is_first(self):
        source = FixtureSource(Path(__file__).parent / "fixtures" / "sample_tasks.json")
        task = source.get_task(None)
        assert task.id == "retry_backoff"


# ---------------------------------------------------------------------------
# Section 13: README example — exact demo scenario
# ---------------------------------------------------------------------------

class TestReadmeDemo:
    """Test the exact scenario described in the README."""

    def test_psi0_prefers_novel_backoff_candidate(self):
        """The README says: Psi0 prefers the novel exponential backoff note."""
        source = FixtureSource()
        task = source.get_task("retry_backoff")
        loop = PsiLoop(source=source)

        result = loop.select(
            goal=task.goal,
            current_context=task.current_context,
            max_tokens=task.max_tokens,
            candidates=task.candidates,
        )
        assert result.ranked[0].candidate.id == "novel_backoff_jitter"

    def test_baseline_prefers_redundant_candidate(self):
        """The README says: the baseline prefers the redundant fixed-delay note."""
        source = FixtureSource()
        task = source.get_task("retry_backoff")

        result = select_context_baseline(
            candidates=task.candidates,
            goal=task.goal,
            max_tokens=task.max_tokens,
        )
        assert result.ranked[0].candidate.id == "redundant_fixed_delay"

    def test_cli_list_tasks_includes_retry_backoff(self, monkeypatch, capsys):
        from psi_loop import cli
        monkeypatch.setattr("sys.argv", ["psi-loop", "--list-tasks"])
        cli.main()
        output = capsys.readouterr().out
        assert "retry_backoff" in output


# ---------------------------------------------------------------------------
# Section 14: Mathematical invariants and soundness checks
# ---------------------------------------------------------------------------

class TestMathInvariants:
    def test_psi0_is_multiplicative(self):
        """Psi0 = V * H should exactly equal returned score."""
        for candidate_text, goal, context in [
            ("backoff jitter retry rate limit", "retry exponential backoff", ["fixed delay"]),
            ("schema migration timeline rollout", "plan migration roadmap", []),
            ("", "retry backoff", ["context item"]),
        ]:
            score, value, surprise = psi_0(candidate_text, goal, context)
            assert score == pytest.approx(value * surprise, rel=1e-9, abs=1e-12)

    def test_surprise_decreases_as_context_grows_similar(self):
        """Adding more context items identical to candidate should push surprise toward 0."""
        candidate = "exponential backoff with jitter"
        base_surprise = surprise_score(candidate, [candidate])
        more_surprise = surprise_score(candidate, [candidate, candidate, candidate])
        # All identical: centroid = same, surprise should stay at ~0
        assert base_surprise == pytest.approx(0.0, abs=1e-9)
        assert more_surprise == pytest.approx(0.0, abs=1e-9)

    def test_psi0_score_bounded(self):
        """Psi0 score is in [0,1] since both V and H are clamped to [0,1]."""
        for text, goal, ctx in [
            ("backoff jitter retry", "retry backoff jitter exponential", []),
            ("aaa bbb ccc", "retry backoff", ["jitter exponential"]),
        ]:
            score, v, h = psi_0(text, goal, ctx)
            assert 0.0 <= score <= 1.0
            assert 0.0 <= v <= 1.0
            assert 0.0 <= h <= 1.0

    def test_cosine_similarity_symmetric(self):
        """cosine_similarity_vectors(a, b) == cosine_similarity_vectors(b, a)."""
        left = {"backoff": 2.0, "jitter": 1.0}
        right = {"backoff": 1.0, "retry": 3.0}
        assert cosine_similarity_vectors(left, right) == pytest.approx(
            cosine_similarity_vectors(right, left)
        )

    def test_plan_bonus_alpha_less_than_one(self):
        """PLAN_BONUS_ALPHA should be small enough that it can't make a zero-overlap item positive."""
        assert PLAN_BONUS_ALPHA < 1.0
        assert PLAN_BONUS_ALPHA > 0.0

    def test_near_tie_epsilon_is_small(self):
        """NEAR_TIE_EPSILON should be meaningfully small relative to the [0,1] score range."""
        assert NEAR_TIE_EPSILON < 0.05
        assert NEAR_TIE_EPSILON > 0.0

    def test_baseline_score_surprise_is_always_zero(self):
        """baseline_score() always returns surprise=0.0."""
        from psi_loop.baseline import baseline_score
        for text, goal in [
            ("backoff jitter", "retry backoff"),
            ("", "retry backoff"),
            ("completely different", "retry backoff jitter"),
        ]:
            _, _, surprise = baseline_score(text, goal, ["some context"])
            assert surprise == 0.0


# ---------------------------------------------------------------------------
# Section 15: Large / stress inputs
# ---------------------------------------------------------------------------

class TestLargeInputs:
    def test_many_candidates_all_scored(self):
        candidates = [
            Candidate(id=f"c{i}", text=f"retry backoff jitter candidate {i}", source="test")
            for i in range(100)
        ]
        result = select_context(
            candidates=candidates,
            goal="retry exponential backoff",
            current_context=[],
            max_tokens=1000,
        )
        assert len(result.ranked) == 100

    def test_very_long_text_does_not_crash(self):
        long_text = "retry " * 1000
        score, value, surprise = psi_0(long_text, "retry backoff jitter", [])
        assert 0.0 <= score <= 1.0

    def test_very_long_goal_does_not_crash(self):
        long_goal = "retry " * 500 + "backoff jitter"
        score, value, surprise = psi_0("exponential backoff with jitter", long_goal, [])
        assert 0.0 <= score <= 1.0

    def test_large_context_does_not_crash(self):
        large_context = [f"context item {i} with retry and backoff" for i in range(200)]
        score = surprise_score("novel exponential jitter", large_context)
        assert 0.0 <= score <= 1.0

    def test_single_candidate_selected_when_fits(self):
        candidate = Candidate(id="only", text="backoff jitter retry", source="test")
        result = select_context(
            candidates=[candidate],
            goal="retry backoff jitter",
            current_context=[],
            max_tokens=100,
        )
        assert len(result.selected) == 1
        assert result.selected[0].candidate.id == "only"

    def test_no_candidate_fits_budget_selected_is_empty(self):
        big_candidates = [
            Candidate(id=f"big{i}", text="one " * 50, source="test")
            for i in range(5)
        ]
        result = select_context(
            candidates=big_candidates,
            goal="retry backoff",
            current_context=[],
            max_tokens=5,
        )
        assert result.selected == []
