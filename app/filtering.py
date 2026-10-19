from app.config import MIN_SCORE, RELATIVE_MARGIN


def filter_hits(hits: list[dict], min_score: float = MIN_SCORE,
                margin: float = RELATIVE_MARGIN) -> list[dict]:
    """Keep only hits that are relevant enough."""
    hits = sorted(hits, key=lambda h: h["score"], reverse=True)
    if not hits or hits[0]["score"] < min_score:
        return []
    best = hits[0]["score"]
    return [h for h in hits if h["score"] >= min_score and h["score"] >= best - margin]