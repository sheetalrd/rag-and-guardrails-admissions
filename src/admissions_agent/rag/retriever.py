from admissions_agent.rag.models import RetrievedChunk
from admissions_agent.rag.vector_store import get_collection


def retrieve(query: str, top_k: int) -> list[RetrievedChunk]:
    """Return the `top_k` chunks closest to the query (smaller distance = more similar)."""
    results = get_collection().query(query_texts=[query], n_results=top_k)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        RetrievedChunk(text=text, source=meta["source"], page=meta["page"], distance=distance)
        for text, meta, distance in zip(documents, metadatas, distances)
    ]
