from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "docs"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "clinic_docs"

# Chunking: sizes are in words (a simple stand-in for tokens).
# These are the first knobs you'll tune in Week 2 and report in the README.
CHUNK_SIZE = 120
CHUNK_OVERLAP = 25

# How many chunks to retrieve per question.
TOP_K = 4

MIN_SCORE = 0.15        # only removes clearly unrelated chunks; the LLM judges answerability
RELATIVE_MARGIN = 0.25  # drop hits more than this far below the best hit