"""Split markdown documents into overlapping chunks that keep their section title."""
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str   # file name, e.g. "services.md"
    section: str  # nearest heading, e.g. "Blood tests"
    index: int    # position of the chunk within the file


_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def split_sections(markdown: str) -> list[tuple[str, str]]:
    """Return [(heading, body), ...]. Text before the first heading gets heading ''."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in markdown.splitlines():
        m = _HEADING.match(line)
        if m:
            sections.append((m.group(2).strip(), []))
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(body).strip()) for h, body in sections if "".join(body).strip()]


def window_words(text: str, size: int, overlap: int) -> list[str]:
    """Slide a window of `size` words over the text, stepping by size - overlap."""
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and < size")
    words = text.split()
    if not words:
        return []
    step = size - overlap
    out = []
    for start in range(0, len(words), step):
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return out


def chunk_document(markdown: str, source: str, size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading, body in split_sections(markdown):
        for piece in window_words(body, size, overlap):
            # Prepend the heading so the chunk is meaningful on its own
            # (and the heading words help retrieval).
            text = f"{heading}: {piece}" if heading else piece
            chunks.append(Chunk(text=text, source=source, section=heading, index=len(chunks)))
    return chunks
