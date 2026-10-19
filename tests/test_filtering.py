from app.filtering import filter_hits


def hits(*scores):
    return [{"score": s, "text": str(s)} for s in scores]


def test_example_from_real_run_keeps_only_the_best():
    out = filter_hits(hits(0.8226, 0.5046, 0.4576, 0.4419))
    assert [h["score"] for h in out] == [0.8226]


def test_close_scores_are_all_kept():
    out = filter_hits(hits(0.70, 0.62, 0.55, 0.30))
    assert [h["score"] for h in out] == [0.70, 0.62, 0.55]


def test_nothing_relevant_returns_empty():
    assert filter_hits(hits(0.30, 0.25, 0.20)) == []


def test_empty_input():
    assert filter_hits([]) == []


def test_unsorted_input_is_handled():
    out = filter_hits(hits(0.5, 0.8, 0.6))
    assert out[0]["score"] == 0.8


def test_nothing_relevant_returns_empty():
    assert filter_hits(hits(0.12, 0.10, 0.05)) == []