from admissions_agent.rag.models import Chunk, Page


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into chunks of `chunk_size` words, each sharing `overlap` words with the previous one."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    step = chunk_size - overlap
    pieces = []
    for start in range(0, len(words), step):
        pieces.append(" ".join(words[start : start + chunk_size]))
        if start + chunk_size >= len(words):
            break
    return pieces


def chunk_pages(pages: list[Page], chunk_size: int, overlap: int) -> list[Chunk]:
    chunks = []
    for page in pages:
        for index, piece in enumerate(split_text(page.text, chunk_size, overlap)):
            chunks.append(
                Chunk(
                    id=f"{page.source}-p{page.page}-c{index}",
                    text=piece,
                    source=page.source,
                    page=page.page,
                )
            )
    return chunks
