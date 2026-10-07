from admissions_agent.rag.models import Chunk
from admissions_agent.rag.vector_store import reset_collection


def store_chunks(chunks: list[Chunk]) -> int:
    """Embed the chunks and store them in ChromaDB, replacing anything ingested before."""
    collection = reset_collection()
    collection.add(
        ids=[chunk.id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        metadatas=[{"source": chunk.source, "page": chunk.page} for chunk in chunks],
    )
    return collection.count()
