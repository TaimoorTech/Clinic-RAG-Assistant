from app.config import COLLECTION_NAME, TOP_K
from app.ingest import get_client
from app.filtering import filter_hits

def retrieve(question: str, k: int = TOP_K, apply_filter: bool = True) -> list[dict]:
    """Return the k chunks most similar to the question, best first."""
    collection = get_client().get_collection(COLLECTION_NAME)
    res = collection.query(query_texts=[question], n_results=k)
    hits = []
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({
            "text": text,
            "source": meta["source"],
            "section": meta["section"],
            "score": round(1 - dist, 4),  # cosine similarity: higher = closer
        })
    return filter_hits(hits) if apply_filter else hits
