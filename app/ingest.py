"""Read data/docs/*.md, chunk them, embed them and store them in Chroma.

Run:  python -m app.ingest
Safe to re-run: it rebuilds the collection from scratch each time.
"""
import chromadb

from app.chunking import chunk_document
from app.config import (CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME,
                        DOCS_DIR)


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def build_index() -> int:
    client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection did not exist yet
    # No embedding function given -> Chroma's default local model (all-MiniLM-L6-v2).
    # It downloads once on first use, then runs locally with no API key.
    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    ids, texts, metas = [], [], []
    for path in sorted(DOCS_DIR.glob("*.md")):
        chunks = chunk_document(path.read_text(encoding="utf-8"), path.name,
                                CHUNK_SIZE, CHUNK_OVERLAP)
        for c in chunks:
            ids.append(f"{c.source}::{c.index}")
            texts.append(c.text)
            metas.append({"source": c.source, "section": c.section})

    if not ids:
        raise SystemExit(f"No .md files found in {DOCS_DIR}")
    collection.add(ids=ids, documents=texts, metadatas=metas)
    return len(ids)


if __name__ == "__main__":
    n = build_index()
    print(f"Indexed {n} chunks from {DOCS_DIR}")
