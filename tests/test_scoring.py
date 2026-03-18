from psi_loop.embedders import Embedder, centroid, cosine_similarity_vectors
from psi_loop.scoring import goal_similarity, goal_term_weight, keyword_overlap, psi_0, surprise_score


class FakeDenseEmbedder(Embedder):
    def __init__(self, vectors: dict[str, tuple[float, ...]]):
        self.vectors = vectors

    def embed(self, text: str) -> tuple[float, ...]:
        return self.vectors[text]


class DeterministicDenseEmbedder(Embedder):
    def embed(self, text: str) -> tuple[float, ...]:
        return (
            float(len(text)),
            float(sum(ord(char) for char in text) % 101),
            float(text.count(" ")),
        )


def test_keyword_overlap_counts_goal_terms():
    goal = "Design retry logic with exponential backoff and jitter"
    candidate = "Use exponential backoff with jitter for retries"

    score = keyword_overlap(candidate, goal)

    assert round(score, 3) == 0.963


def test_goal_term_weight_prioritizes_mechanism_terms():
    assert goal_term_weight("backoff") == 4.0
    assert goal_term_weight("design") == 0.25
    assert goal_term_weight("schema") == 1.0


def test_keyword_overlap_prefers_mechanism_overlap_over_generic_overlap():
    goal = "Design Python API client retry logic with exponential backoff and jitter"
    generic_candidate = "Design Python API client logic"
    mechanism_candidate = "Use backoff jitter retry"

    generic_score = keyword_overlap(generic_candidate, goal)
    mechanism_score = keyword_overlap(mechanism_candidate, goal)

    assert mechanism_score > generic_score


def test_surprise_score_drops_for_redundant_context():
    candidate = "Retry transient failures with fixed delay."
    context = ["The current client retries transient failures with fixed delay."]

    score = surprise_score(candidate, context)

    assert score < 0.5


def test_surprise_score_defaults_high_without_context():
    candidate = "Exponential backoff with jitter handles rate limits."

    assert surprise_score(candidate, []) == 1.0


def test_surprise_score_accepts_injected_embedder():
    embedder = FakeDenseEmbedder(
        {
            "candidate": (1.0, 0.0),
            "context_a": (1.0, 0.0),
            "context_b": (0.0, 1.0),
        }
    )

    score = surprise_score("candidate", ["context_a", "context_b"], embedder=embedder)

    assert round(score, 3) == 0.293


def test_surprise_score_clamps_negative_dense_cosine_to_one():
    embedder = FakeDenseEmbedder(
        {
            "candidate": (-1.0, 0.0),
            "context_a": (1.0, 0.0),
        }
    )

    score = surprise_score("candidate", ["context_a"], embedder=embedder)

    assert score == 1.0


def test_goal_similarity_accepts_dense_embedder():
    embedder = FakeDenseEmbedder(
        {
            "left": (1.0, 0.0),
            "right": (0.0, 1.0),
        }
    )

    assert goal_similarity("left", "right", embedder=embedder) == 0.0


def test_dense_vector_centroid_path_stays_compatible():
    vectors = [(1.0, 0.0), (0.0, 1.0)]

    center = centroid(vectors)
    similarity = cosine_similarity_vectors((1.0, 0.0), center)

    assert center == (0.5, 0.5)
    assert round(similarity, 3) == 0.707


def test_psi0_accepts_injected_embedder_without_changing_value_signal():
    embedder = FakeDenseEmbedder(
        {
            "Exponential backoff handles retries": (1.0, 0.0),
            "Retry with fixed delay": (1.0, 0.0),
        }
    )

    score, value, surprise = psi_0(
        "Exponential backoff handles retries",
        "Use exponential backoff for retries",
        ["Retry with fixed delay"],
        embedder=embedder,
    )

    assert value > 0.5
    assert surprise == 0.0
    assert score == 0.0


def test_psi0_score_is_value_times_surprise():
    candidate = "Data contracts and freshness alerts improve roadmap reliability"
    goal = "Select the best notes for a roadmap discussion on analytics reliability."
    embedder = FakeDenseEmbedder(
        {
            candidate: (1.0, 0.0),
            "current_context": (0.6, 0.8),
        }
    )

    score, value, surprise = psi_0(
        candidate,
        goal,
        ["current_context"],
        embedder=embedder,
    )

    assert value > 0.0
    assert round(score, 6) == round(value * surprise, 6)


def test_planning_bonus_activates_for_planning_goal_with_cues():
    """Planning-shaped goal + candidate with sequencing/dependency/risk cues yields V_prime > V_base."""
    goal = "Select the best notes for a roadmap discussion on analytics reliability."
    candidate_with_cues = "First phase: validate schema. Then milestone. Depends on migration timeline."
    v_base = keyword_overlap(candidate_with_cues, goal)
    _, value_prime, _ = psi_0(
        candidate_with_cues,
        goal,
        ["current context"],
        embedder=FakeDenseEmbedder({candidate_with_cues: (0.5, 0.5), "current context": (0.0, 1.0)}),
    )
    assert value_prime > v_base


def test_non_planning_goal_unchanged_value():
    """Non-planning goal: value from psi_0 equals keyword_overlap (no bonus)."""
    goal = "Improve API retry handling for transient failures and rate limits."
    candidate = "Use exponential backoff with jitter for retries."
    v_base = keyword_overlap(candidate, goal)
    _, value_prime, _ = psi_0(
        candidate,
        goal,
        ["Current client retries with fixed delay."],
        embedder=FakeDenseEmbedder({candidate: (0.3, 0.7), "Current client retries with fixed delay.": (0.9, 0.1)}),
    )
    assert round(value_prime, 6) == round(v_base, 6)


def test_deterministic_dense_embedder_can_drive_psi0():
    embedder = DeterministicDenseEmbedder()

    score, value, surprise = psi_0(
        "Exponential backoff handles retries",
        "Use exponential backoff for retries",
        ["Retry with fixed delay"],
        embedder=embedder,
    )

    assert value > 0.5
    assert 0.0 <= surprise <= 1.0
    assert 0.0 <= score <= 1.0
