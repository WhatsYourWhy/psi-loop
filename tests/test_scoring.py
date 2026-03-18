from psi_loop.embedders import Embedder
from psi_loop.scoring import goal_term_weight, keyword_overlap, psi_0, surprise_score


class FakeDenseEmbedder(Embedder):
    def __init__(self, vectors: dict[str, tuple[float, ...]]):
        self.vectors = vectors

    def embed(self, text: str) -> tuple[float, ...]:
        return self.vectors[text]


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


def test_psi0_value_signal_stays_embedder_independent():
    goal = "Design Python API client retry logic with exponential backoff and jitter"
    candidate = "Use backoff jitter retry"
    embedder = FakeDenseEmbedder(
        {
            candidate: (1.0, 0.0),
            "context": (0.0, 1.0),
        }
    )

    no_embedder_score = psi_0(candidate, goal, ["context"])[1]
    dense_embedder_score = psi_0(candidate, goal, ["context"], embedder=embedder)[1]

    assert no_embedder_score == dense_embedder_score
