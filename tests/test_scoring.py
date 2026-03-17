from psi_loop.scoring import keyword_overlap, surprise_score


def test_keyword_overlap_counts_goal_terms():
    goal = "Design retry logic with exponential backoff and jitter"
    candidate = "Use exponential backoff with jitter for retries"

    score = keyword_overlap(candidate, goal)

    assert score > 0.5


def test_surprise_score_drops_for_redundant_context():
    candidate = "Retry transient failures with fixed delay."
    context = ["The current client retries transient failures with fixed delay."]

    score = surprise_score(candidate, context)

    assert score < 0.5


def test_surprise_score_defaults_high_without_context():
    candidate = "Exponential backoff with jitter handles rate limits."

    assert surprise_score(candidate, []) == 1.0
