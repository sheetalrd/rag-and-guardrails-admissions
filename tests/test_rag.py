import pytest

from admissions_agent.rag.chunker import chunk_pages, split_text
from admissions_agent.rag.context_builder import build_context
from admissions_agent.rag.models import Page, RetrievedChunk


def test_split_text_overlaps_chunks():
    text = " ".join(str(number) for number in range(10))

    assert split_text(text, chunk_size=4, overlap=1) == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]


def test_split_text_keeps_short_text_whole():
    assert split_text("just a few words", chunk_size=50, overlap=10) == ["just a few words"]


def test_split_text_rejects_overlap_as_big_as_chunk():
    with pytest.raises(ValueError):
        split_text("some text", chunk_size=5, overlap=5)


def test_chunk_pages_keeps_source_and_page():
    pages = [Page(text="one two three four five", source="tudelft_admission_application_rag_dataset.pdf", page=3)]

    chunks = chunk_pages(pages, chunk_size=3, overlap=1)

    assert [chunk.id for chunk in chunks] == ["tudelft_admission_application_rag_dataset.pdf-p3-c0", "tudelft_admission_application_rag_dataset.pdf-p3-c1"]
    assert all(chunk.source == "tudelft_admission_application_rag_dataset.pdf" and chunk.page == 3 for chunk in chunks)


def test_build_context_numbers_chunks_with_sources(): 
    chunks = [
        RetrievedChunk(text="Important application dates.", source="tudelft_admission_application_rag_dataset.pdf", page=2, distance=0.1),
        RetrievedChunk(text="Official TU Delft sources.", source="tudelft_admission_application_rag_dataset.pdf", page=3, distance=0.2),
    ]

    context = build_context(chunks)

    assert context == (
        "[1] Source: tudelft_admission_application_rag_dataset.pdf, page 2\nImportant application dates.\n\n"
        "[2] Source: tudelft_admission_application_rag_dataset.pdf, page 3\nOfficial TU Delft sources."
    )

 