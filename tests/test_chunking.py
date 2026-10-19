import pytest

from app.chunking import chunk_document, split_sections, window_words


def test_split_sections_keeps_headings():
    md = "# Title\nintro text\n## Hours\nOpen 9 to 5\n## Fees\nFree"
    assert split_sections(md) == [("Title", "intro text"), ("Hours", "Open 9 to 5"), ("Fees", "Free")]


def test_split_sections_text_before_first_heading():
    assert split_sections("hello\n# A\nbody") == [("", "hello"), ("A", "body")]


def test_window_short_text_single_chunk():
    assert window_words("a b c", size=10, overlap=2) == ["a b c"]


def test_window_overlap():
    words = " ".join(str(i) for i in range(10))
    chunks = window_words(words, size=4, overlap=1)
    assert chunks[0] == "0 1 2 3"
    assert chunks[1].startswith("3 ")          # 1 word of overlap
    assert chunks[-1].endswith("9")            # nothing is lost at the end
    assert len(chunks) == 3


def test_window_never_drops_or_repeats_tail():
    words = " ".join(str(i) for i in range(8))
    chunks = window_words(words, size=4, overlap=0)
    assert chunks == ["0 1 2 3", "4 5 6 7"]


def test_window_rejects_bad_params():
    with pytest.raises(ValueError):
        window_words("a b", size=3, overlap=3)
    with pytest.raises(ValueError):
        window_words("a b", size=0, overlap=0)


def test_chunk_document_metadata_and_heading_prefix():
    md = "## Opening hours\nMonday to Friday 8am to 6pm"
    chunks = chunk_document(md, "hours.md", size=50, overlap=5)
    assert len(chunks) == 1
    assert chunks[0].source == "hours.md"
    assert chunks[0].section == "Opening hours"
    assert chunks[0].text.startswith("Opening hours:")


def test_chunk_indexes_are_sequential_across_sections():
    md = "## A\none two\n## B\nthree four"
    assert [c.index for c in chunk_document(md, "x.md", 10, 2)] == [0, 1]
